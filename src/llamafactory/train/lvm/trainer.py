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
from typing import TYPE_CHECKING, Optional

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

    def _compute_gae_advantage_return(self, y_hat: torch.Tensor, value_mask: torch.Tensor, reward: float, gamma: float, lam: float) -> torch.Tensor:
        value_mask = value_mask.to(y_hat.dtype)
        token_level_rewards = reward
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = y_hat.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards + gamma * nextvalues - y_hat[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = y_hat[:, t] * value_mask[:, t] + (1 - value_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * value_mask[:, t] + (1 - value_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + y_hat

        # 简单版解码：
        # - 整体左移一位：当前 token 的 return 等于下一个 token 的旧 return
        # - 最后一位补 0
        # - 再乘上 value_mask，保证只有 mask=1 的位置生效
        shifted_returns = torch.zeros_like(returns)
        shifted_returns[:, :-1] = returns[:, 1:]
        returns = shifted_returns * value_mask
        advantages = returns - y_hat

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
        value_labels = inputs.pop("value_labels").to(torch.float64) # remaining number of tokens
        value_mask = inputs.pop("value_mask").to(torch.float64) # mask of response tokens
        response_len = value_mask.sum(dim=-1)


        value_preds = self._forward_value(model, inputs).to(torch.float64)

        gamma = self.finetuning_args.lvm_gamma
        if self.finetuning_args.lvm_alpha > 0:
            lam = 1 - 1/(self.finetuning_args.lvm_alpha * response_len)
            lam = torch.clamp(lam, min=0.0)
        else:
            lam = self.finetuning_args.lvm_lam
        delta = self.finetuning_args.lvm_huber_loss_delta
        agg_method = self.finetuning_args.lvm_agg_method
        reward = 1.0 - gamma

        y_hat = torch.sigmoid(value_preds)

        with torch.no_grad():
            advantages, returns = self._compute_gae_advantage_return(y_hat, value_mask, reward, gamma, lam)
            # advantages = verl_F.masked_whiten(advantages, value_mask)

        # value_loss = torch.nn.functional.huber_loss(
        #     y_hat, returns, reduction="none", delta=delta
        # )
        value_loss = 0.5 * (y_hat-returns) ** 2
        value_loss = self.aggregate_loss(value_loss, value_mask, method = agg_method)
        # if self.accelerator.is_main_process:
        #     import pdb; pdb.set_trace()
        with torch.no_grad():
            # Compute and report losses for lam=0 and lam=1 (diagnostics only)
            _, returns_lam0 = self._compute_gae_advantage_return(y_hat, value_mask, reward, gamma, 0.0)
            # loss_lam0 = F.huber_loss(y_hat, returns_lam0, reduction="none", delta=delta)
            loss_lam0 = 0.5 * (y_hat-returns_lam0) ** 2
            loss_lam0 = self.aggregate_loss(loss_lam0, value_mask, method = "seq-sum-token-sum")
            # relative loss: |y_hat - returns_lam0| / (|returns_lam0| + eps)
            eps = 1e-8
            rel_loss_lam0 = torch.abs(y_hat - returns_lam0) / (torch.abs(returns_lam0) + eps)
            rel_loss_lam0 = self.aggregate_loss(rel_loss_lam0, value_mask, method = "seq-sum-token-sum")

            # _, returns_lam05 = self._compute_gae_advantage_return(y_hat, value_mask, reward, gamma, 0.5)
            # loss_lam05 = F.huber_loss(y_hat, returns_lam05, reduction="none", delta=delta)
            # # loss_lam05 = 0.5 * (y_hat-returns_lam05) ** 2
            # loss_lam05 = self.aggregate_loss(loss_lam05, value_mask, method = agg_method)
            
            _, returns_lam1 = self._compute_gae_advantage_return(y_hat, value_mask, reward, gamma, 1.0)
            # loss_lam1 = F.huber_loss(y_hat, returns_lam1, reduction="none", delta=delta)
            loss_lam1 = 0.5 * (y_hat-returns_lam1) ** 2
            loss_lam1 = self.aggregate_loss(loss_lam1, value_mask, method = "seq-sum-token-sum")
            # relative loss: |y_hat - returns_lam1| / (|returns_lam1| + eps)
            rel_loss_lam1 = torch.abs(y_hat - returns_lam1) / (torch.abs(returns_lam1) + eps)
            rel_loss_lam1 = self.aggregate_loss(rel_loss_lam1, value_mask, method = "seq-sum-token-sum")
            
            # Accumulate diagnostics across micro-steps (gradient accumulation)
            # We store the *sum* of losses and the *sum* of valid tokens,
            # then divide by total tokens when logging.
            ga_steps = max(1, getattr(self.args, "gradient_accumulation_steps", 1))
            batch_token_count = float(value_mask.sum().detach())
            # self._diag_ga_sum_main += float(value_loss.detach())  # already normalized loss
            self._diag_ga_sum_lam0 += float(loss_lam0.detach())
            self._diag_ga_sum_lam1 += float(loss_lam1.detach())
            # self._diag_ga_sum_lam05 += float(loss_lam05.detach())
            # accumulate relative losses
            self._diag_ga_sum_rel_lam0 += float(rel_loss_lam0.detach())
            self._diag_ga_sum_rel_lam1 += float(rel_loss_lam1.detach())
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
                    cnt = torch.tensor(self._diag_ga_count, device=device, dtype=torch.float32)  # total tokens
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        # torch.distributed.all_reduce(sum_main, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(sum_l0, op=torch.distributed.ReduceOp.SUM)
                        # torch.distributed.all_reduce(sum_l05, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(sum_l1, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(sum_rel_l0, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(sum_rel_l1, op=torch.distributed.ReduceOp.SUM)
                        torch.distributed.all_reduce(cnt, op=torch.distributed.ReduceOp.SUM)

                    if self.accelerator.is_main_process:
                        # avg_main = (sum_main / cnt.clamp_min(1.0)).item()
                        avg_l0 = (sum_l0 / cnt.clamp_min(1.0)).item()
                        # avg_l05 = (sum_l05 / cnt.clamp_min(1.0)).item()
                        avg_l1 = (sum_l1 / cnt.clamp_min(1.0)).item()
                        avg_rel_l0 = (sum_rel_l0 / cnt.clamp_min(1.0)).item()
                        avg_rel_l1 = (sum_rel_l1 / cnt.clamp_min(1.0)).item()
                        # print(f"value_loss(lam={lam}): {avg_main} | value_loss_lam0: {avg_l0} | value_loss_lam1: {avg_l1}")
                        self.log(
                            {
                                # f"train/value_loss_lam{lam}": avg_main,
                                "train/value_loss_lam0": avg_l0,
                                # "train/value_loss_lam05": avg_l05,
                                "train/value_loss_lam1": avg_l1,
                                "train/value_rel_loss_lam0": avg_rel_l0,
                                "train/value_rel_loss_lam1": avg_rel_l1,
                            }
                        )
                # reset accumulators after optimizer step
                # self._diag_ga_sum_main = 0.0
                self._diag_ga_sum_lam0 = 0.0
                # self._diag_ga_sum_lam05 = 0.0
                self._diag_ga_sum_lam1 = 0.0
                self._diag_ga_sum_rel_lam0 = 0.0
                self._diag_ga_sum_rel_lam1 = 0.0
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
                advantages, returns = self._compute_gae_advantage_return(y_hat, value_mask, reward, gamma, lam)
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
