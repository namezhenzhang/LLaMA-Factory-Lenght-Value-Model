# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from types import MethodType
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)

EPS = 1e-8

class LengthValueTrainer(Trainer):
    r"""Trainer for length value regression."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False
        self.finetuning_args = finetuning_args
        self.can_return_loss = True
        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))
        
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        
        # Diagnostics accumulators for gradient accumulation-safe logging
        # self._diag_ga_sum_main = 0.0
        self._diag_ga_sum_lam0 = 0.0
        self._diag_ga_sum_lam1 = 0.0
        # self._diag_ga_sum_lam05 = 0.0
        # Relative loss diagnostics (|y - ret| / |ret|)
        self._diag_ga_sum_rel_lam0 = 0.0
        self._diag_ga_sum_rel_lam1 = 0.0
        self._diag_ga_count = 0
        self._diag_ga_step = 0

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def _forward_value(self, model: "PreTrainedModel", inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            value_preds = outputs.get("value", outputs.get("values"))
        elif isinstance(outputs, tuple):
            if len(outputs) >= 3:
                value_preds = outputs[2]
            else:
                raise ValueError("Model outputs do not contain value predictions.")
        else:
            raise ValueError("Unsupported model output type for length value trainer.")

        if value_preds.dim() == 3:
            value_preds = value_preds.squeeze(-1)

        return value_preds

    @staticmethod
    def _value_to_length(v: torch.Tensor, gamma: float, eps: float = EPS) -> torch.Tensor:
        r"""Convert discounted return value v in (-1, 0) to remaining length.

        Paper: l = ln(1 + v) / ln(gamma).
        """
        gamma_t = torch.tensor(gamma, dtype=v.dtype, device=v.device)
        denom = torch.log(gamma_t).clamp_max(-eps)  # ln(gamma) < 0, avoid divide-by-0 when gamma≈1
        v_clamped = v.clamp(min=-1.0 + eps, max=0.0 - eps)  # keep 1+v in (0,1)
        return torch.log1p(v_clamped) / denom

    # def _apply_decay_factor(self, value_labels: torch.Tensor, value_mask: torch.Tensor) -> torch.Tensor:
    #     r"""Apply decay factor to future token values.
        
    #     Args:
    #         value_labels: Tensor of shape (batch_size, seq_len) containing the number of future tokens
    #         value_mask: Tensor of shape (batch_size, seq_len) indicating valid positions
            
    #     Returns:
    #         Modified value_labels with decay applied
    #     """
    #     if self.finetuning_args.lvm_decay_factor == 1.0:
    #         return value_labels
            
    #     # Create decay weights: for each position, apply decay_factor^(n-1) where n is the distance
    #     batch_size, seq_len = value_labels.shape
    #     device = value_labels.device
        
    #     # Create position indices for each sequence
    #     positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
    #     # Calculate decay weights: decay_factor^(n-1) where n is the distance from current token
    #     # For value_labels[i, j], the distance is value_labels[i, j], so decay is decay_factor^(value_labels[i, j]-1)
    #     decay_weights = torch.pow(self.finetuning_args.lvm_decay_factor, value_labels)
        
    #     # Apply decay weights to value_labels
    #     decayed_labels = 1 - decay_weights
        
    #     # Only apply decay where mask is valid
    #     decayed_labels = decayed_labels * value_mask
        
    #     return decayed_labels

    def _compute_gae_advantage_return(
        self,
        y_hat: torch.Tensor,
        value_mask: torch.Tensor,
        value_labels: torch.Tensor,
        reward: Union[float, torch.Tensor],
        gamma: float,
        lam: Union[float, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute token-level GAE advantages and λ-return targets.

        This implementation follows the paper:
          δ_t = r_t + γ v_{t+1}^{old} - v_t^{old}
          A_t = δ_t + γλ A_{t+1}
          v_t^{tgt} = v_t^{old} + A_t

        Key detail for cutoff truncation:
        - When the trajectory continues beyond the observed window (no token t+1 in the batch),
          we bootstrap v_{t+1}^{old} using the deterministic Monte-Carlo return derived from
          the ground-truth remaining length `value_labels` (see `value_regression.py`).

        Notes:
        - `value_mask` selects valid regression positions (prompt last token + response tokens, excluding EOS).
        - `value_labels` is the remaining number of tokens to go until EOS for each position.
        - `reward` is the per-step reward for non-EOS steps (often ±(1-γ) depending on sign convention).
        """
        dtype = y_hat.dtype
        device = y_hat.device

        # We only regress on mask=1 positions; everything else is ignored by the loss.
        value_mask_f = value_mask.to(dtype)
        remaining = value_labels.to(dtype)

        # stopgrad baseline
        v_old = y_hat.detach()

        lam_t = torch.as_tensor(lam, dtype=dtype, device=device)
        reward_t = torch.as_tensor(reward, dtype=dtype, device=device)

        # Deterministic MC return from remaining length:
        # If reward = (1-γ)  => G_t =  (1 - γ^n)
        # If reward = -(1-γ) => G_t = -(1 - γ^n)
        # where n = remaining steps until EOS (i.e., `remaining`).
        def mc_value_from_remaining(n: torch.Tensor) -> torch.Tensor:
            if gamma == 1.0:
                # Degenerate case; in practice gamma∈(0,1).
                return torch.zeros_like(n)
            gamma_t = torch.tensor(gamma, dtype=dtype, device=device)
            # base ∈ [0, 1] (or [0,1) if finite)
            base = 1.0 - torch.pow(gamma_t, n)
            # scale is +1 or -1 depending on reward sign convention (reward should be ±(1-γ))
            scale = 1 if reward_t / (1.0 - gamma_t) > 0 else -1
            return scale * base

        B, T = v_old.shape
        advantages = torch.zeros((B, T), dtype=dtype, device=device)
        lastgaelam = torch.zeros((B,), dtype=dtype, device=device)

        for t in reversed(range(T)):
            mask_t = value_mask_f[:, t]
            if t < T - 1:
                v_next_model = v_old[:, t + 1]
                # If the next token is outside the observed window for this sequence
                # (e.g., cutoff truncation or padding), bootstrap from deterministic MC value.
                need_det = (remaining[:, t] > 1.0) & (value_mask_f[:, t + 1] == 0.0)
                v_next_det = mc_value_from_remaining(torch.clamp(remaining[:, t] - 1.0, min=0.0))
                v_next = torch.where(need_det, v_next_det, v_next_model)
                # If the true next step is EOS (remaining==1), its value is deterministic 0.
                v_next = torch.where(remaining[:, t] <= 1.0, torch.zeros_like(v_next), v_next)
            else: # t == T - 1
                # No t+1 token in the observed window. Bootstrap from the deterministic return
                # of the next state using ground-truth remaining length.
                # Next state's remaining length is (remaining-1).
                v_next = torch.where(
                    remaining[:, t] <= 1.0,
                    torch.zeros((B,), dtype=dtype, device=device),
                    mc_value_from_remaining(torch.clamp(remaining[:, t] - 1.0, min=0.0)),
                )

            delta = reward_t + gamma * v_next - v_old[:, t]
            gae_t = delta + gamma * lam_t * lastgaelam

            # Only update recursion and store values on valid positions.
            # This also prevents leakage into padded/ignored tokens.
            lastgaelam = gae_t * mask_t + lastgaelam * (1.0 - mask_t)
            advantages[:, t] = lastgaelam

        returns = v_old + advantages
        return advantages, returns
    def aggregate_loss(self, loss: torch.Tensor, value_mask: torch.Tensor, method: str = "token-mean") -> torch.Tensor:
        if method == "token-mean":
            value_mask = value_mask.to(loss.dtype)
            total = (loss * value_mask).sum()
            denom = value_mask.sum().clamp_min(torch.tensor(1.0, device=loss.device, dtype=loss.dtype))
            return total / denom
        elif method == "seq-mean-token-sum":
            value_mask = value_mask.to(loss.dtype)
            per_seq_sum = (loss * value_mask).sum(dim=-1)
            valid_seq_mask = (value_mask.sum(dim=-1) > 0).to(loss.dtype)
            denom = valid_seq_mask.sum().clamp_min(torch.tensor(1.0, device=loss.device, dtype=loss.dtype))
            return (per_seq_sum * valid_seq_mask).sum() / denom
        elif method == "seq-mean-token-mean":
            value_mask_f = value_mask.to(loss.dtype)
            per_seq_token_count = value_mask_f.sum(dim=-1).clamp_min(torch.tensor(1.0, device=loss.device, dtype=loss.dtype))
            per_seq_mean = (loss * value_mask_f).sum(dim=-1) / per_seq_token_count
            valid_seq_mask = (value_mask.sum(dim=1) > 0).to(loss.dtype)
            denom = valid_seq_mask.sum().clamp_min(torch.tensor(1.0, device=loss.device, dtype=loss.dtype))
            return (per_seq_mean * valid_seq_mask).sum() / denom
        elif method == "seq-sum-token-sum":
            value_mask = value_mask.to(loss.dtype)
            # Sum over tokens within each sequence, then sum over all sequences.
            # This is an unnormalized total loss over all valid tokens.
            return (loss * value_mask).sum()
        elif method == "seq-mean-token-mean-max":
            value_mask = value_mask.to(loss.dtype)
            per_seq_sum = (loss * value_mask).sum(dim=-1) / 1000.0
            valid_seq_mask = (value_mask.sum(dim=-1) > 0).to(loss.dtype)
            denom = valid_seq_mask.sum().clamp_min(torch.tensor(1.0, device=loss.device, dtype=loss.dtype))
            return (per_seq_sum * valid_seq_mask).sum() / denom
        else:
            raise ValueError(f"Invalid method: {method}")

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, torch.Tensor], return_outputs: bool = False, **kwargs
    ):
        # TODO: 看一下这里是不是需要把 value_labels 转换为 float64
        value_labels = inputs.pop("value_labels").to(torch.float64) # remaining number of tokens
        value_mask = inputs.pop("value_mask").to(torch.float64) # mask of response tokens
        # TODO: 这里 max 是不是对的
        response_len = value_labels.max(dim=-1).values

        # TODO: 这里_forward_value是不是太复杂了，需要写成一个函数吗？
        value_preds = self._forward_value(model, inputs).to(torch.float64)

        gamma = self.finetuning_args.lvm_gamma
        lam = self.finetuning_args.lvm_lam
        if self.finetuning_args.lvm_alpha > 0:
            lam = 1 - 1/(self.finetuning_args.lvm_alpha * response_len)
            lam = torch.clamp(lam, min=0.0)

        delta = self.finetuning_args.lvm_huber_loss_delta
        agg_method = self.finetuning_args.lvm_agg_method
        # Paper: r_t = -(1 - gamma) for all non-EOS steps (we do not regress on EOS itself).
        reward = -(1.0 - gamma)

        y_hat = torch.sigmoid(value_preds)
        # Paper: \hat v(s_t) = -sigma(W h_t) in (-1, 0).
        y_hat = -y_hat

        with torch.no_grad():
            advantages, returns = self._compute_gae_advantage_return(y_hat, value_mask, value_labels, reward, gamma, lam)
            # advantages = verl_F.masked_whiten(advantages, value_mask)

        value_loss = torch.nn.functional.huber_loss(
            y_hat, returns, reduction="none", delta=delta
        )
        # value_loss = 0.5 * (y_hat-returns) ** 2

        # Optionally convert to relative loss so that positions with
        # larger target values do not dominate the overall loss.
        if self.finetuning_args.lvm_relative_loss:
            value_loss = value_loss / (returns.abs() + EPS)

        value_loss = self.aggregate_loss(value_loss, value_mask, method = agg_method)

        with torch.no_grad():
            # Compute and report losses for lam=0 and lam=1 (diagnostics only)
            _, returns_lam0 = self._compute_gae_advantage_return(y_hat, value_mask, value_labels, reward, gamma, 0.0)
            # loss_lam0 = F.huber_loss(y_hat, returns_lam0, reduction="none", delta=delta)
            loss_lam0 = 0.5 * (y_hat-returns_lam0) ** 2
            loss_lam0 = self.aggregate_loss(loss_lam0, value_mask, method = "seq-sum-token-sum")
            # relative loss: |y_hat - returns_lam0| / (|returns_lam0| + eps)
            rel_loss_lam0 = torch.abs(y_hat - returns_lam0) / (torch.abs(returns_lam0) + EPS)
            rel_loss_lam0 = self.aggregate_loss(rel_loss_lam0, value_mask, method = "seq-sum-token-sum")
            # Precompute length-space prediction once (used for lam=1 metric below).
            len_pred = self._value_to_length(y_hat, gamma)

            

            _, returns_lam1 = self._compute_gae_advantage_return(y_hat, value_mask, value_labels, reward, gamma, 1.0)
            # loss_lam1 = F.huber_loss(y_hat, returns_lam1, reduction="none", delta=delta)
            loss_lam1 = 0.5 * (y_hat-returns_lam1) ** 2
            loss_lam1 = self.aggregate_loss(loss_lam1, value_mask, method = "seq-sum-token-sum")
            # relative loss: |y_hat - returns_lam1| / (|returns_lam1| + eps)
            rel_loss_lam1 = torch.abs(y_hat - returns_lam1) / (torch.abs(returns_lam1) + EPS)
            rel_loss_lam1 = self.aggregate_loss(rel_loss_lam1, value_mask, method = "seq-sum-token-sum")
            len_tgt_lam1 = self._value_to_length(returns_lam1, gamma)
            rel_len_loss_lam1 = torch.abs(len_pred - len_tgt_lam1) / (torch.abs(len_tgt_lam1) + EPS)
            rel_len_loss_lam1 = self.aggregate_loss(rel_len_loss_lam1, value_mask, method = "seq-sum-token-sum")

            # Accumulate diagnostics across micro-steps (gradient accumulation)
            # We store the *sum* of losses and the *sum* of valid tokens,
            # then divide by total tokens when logging.
            ga_steps = max(1, getattr(self.args, "gradient_accumulation_steps", 1))
            batch_token_count = float(value_mask.sum().detach())
            # self._diag_ga_sum_main += float(value_loss.detach())  # already normalized loss
            self._diag_ga_sum_lam0 += float(loss_lam0.detach())
            self._diag_ga_sum_lam1 += float(loss_lam1.detach())
            # accumulate relative losses
            self._diag_ga_sum_rel_lam0 += float(rel_loss_lam0.detach())
            self._diag_ga_sum_rel_lam1 += float(rel_loss_lam1.detach())
            # accumulate length-space relative losses
            self._diag_ga_sum_rel_len_lam1 = getattr(self, "_diag_ga_sum_rel_len_lam1", 0.0) + float(rel_len_loss_lam1.detach())
            self._diag_ga_count += batch_token_count
            self._diag_ga_step = (self._diag_ga_step + 1) % ga_steps
            # Only log once per optimizer step and in sync with logging_steps
            if self._diag_ga_step == 0 and self.state is not None and self.args is not None:
                if self.args.logging_steps > 0 and (self.state.global_step % self.args.logging_steps == 0):
                    # Reduce across processes (DDP) before computing averages
                    device = value_loss.device
                    # sum_main = torch.tensor(self._diag_ga_sum_main, device=device, dtype=torch.float32)
                    sum_l0 = torch.tensor(self._diag_ga_sum_lam0, device=device, dtype=torch.float32)
                    sum_l1 = torch.tensor(self._diag_ga_sum_lam1, device=device, dtype=torch.float32)
                    # sum_l05 = torch.tensor(self._diag_ga_sum_lam05, device=device, dtype=torch.float32)
                    sum_rel_l0 = torch.tensor(self._diag_ga_sum_rel_lam0, device=device, dtype=torch.float32)
                    sum_rel_l1 = torch.tensor(self._diag_ga_sum_rel_lam1, device=device, dtype=torch.float32)
                    sum_rel_len_l1 = torch.tensor(getattr(self, "_diag_ga_sum_rel_len_lam1", 0.0), device=device, dtype=torch.float32)
                    cnt = torch.tensor(self._diag_ga_count, device=device)  # total tokens
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        # torch.distributed.all_reduce(sum_main, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(sum_l0, op=torch.distributed.ReduceOp.SUM)
                        # torch.distributed.all_reduce(sum_l05, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(sum_l1, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(sum_rel_l0, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(sum_rel_l1, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(sum_rel_len_l1, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(cnt, op=torch.distributed.ReduceOp.SUM)

                    if self.accelerator.is_main_process:
                        # avg_main = (sum_main / cnt.clamp_min(1.0)).item()
                        avg_l0 = (sum_l0 / cnt.clamp_min(1.0)).item()
                        avg_l1 = (sum_l1 / cnt.clamp_min(1.0)).item()
                        avg_rel_l0 = (sum_rel_l0 / cnt.clamp_min(1.0)).item()
                        avg_rel_l1 = (sum_rel_l1 / cnt.clamp_min(1.0)).item()
                        avg_rel_len_l1 = (sum_rel_len_l1 / cnt.clamp_min(1.0)).item()
                        # print(f"value_loss(lam={lam}): {avg_main} | value_loss_lam0: {avg_l0} | value_loss_lam1: {avg_l1}")
                        self.log(
                            {
                                # f"train/value_loss_lam{lam}": avg_main,
                                "train/value_loss_lam0": avg_l0,
                                "train/value_loss_lam1": avg_l1,
                                "train/value_rel_loss_lam0": avg_rel_l0,
                                "train/value_rel_loss_lam1": avg_rel_l1,
                                "train/len_rel_loss_lam1": avg_rel_len_l1,
                            }
                        )
                # reset accumulators after optimizer step
                # self._diag_ga_sum_main = 0.0
                self._diag_ga_sum_lam0 = 0.0
                self._diag_ga_sum_lam1 = 0.0
                self._diag_ga_sum_rel_lam0 = 0.0
                self._diag_ga_sum_rel_lam1 = 0.0
                self._diag_ga_sum_rel_len_lam1 = 0.0
                self._diag_ga_count = 0

        if return_outputs:
            preds_for_metric = value_preds
            combined_labels = torch.stack((value_labels.detach(), value_mask.detach()), dim=-1)
            return value_loss, (preds_for_metric.detach(), combined_labels)

        return value_loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        return 0, None, None
        has_labels = "value_labels" in inputs and "value_mask" in inputs
        inputs = self._prepare_inputs(inputs)

        value_labels = inputs.pop("value_labels", None)
        value_mask = inputs.pop("value_mask", None)

        with torch.no_grad():
            value_preds = self._forward_value(model, inputs)

        loss = None
        combined_labels = None

        if has_labels and value_labels is not None and value_mask is not None:

            gamma = self.finetuning_args.lvm_gamma
            assert 0.0 < gamma <= 1.0, "gamma must be (0.0, 1.0]"
            lam = self.finetuning_args.lvm_lam
            assert 0.0 <= lam <= 1.0, "lam must be [0.0, 1.0]"

            reward = 1 - gamma

            y_hat = torch.sigmoid(value_preds)

            with torch.no_grad():
                advantages, returns = self._compute_gae_advantage_return(y_hat, value_mask, value_labels, reward, gamma, lam)
                # advantages = verl_F.masked_whiten(advantages, value_mask)

            value_loss = torch.nn.functional.huber_loss(
                y_hat, returns, reduction="none", delta=0.5
            )
            # value_loss = 0.5 * (y_hat-returns) ** 2
            value_loss = (value_loss * value_mask).sum() / (value_mask.shape[1]*value_mask.shape[0])

            combined_labels = torch.stack((value_labels, value_mask), dim=-1)

        if prediction_loss_only:
            return loss, None, None

        value_preds = value_preds.detach()
        if combined_labels is not None:
            combined_labels = combined_labels.detach()

        return loss, value_preds, combined_labels
