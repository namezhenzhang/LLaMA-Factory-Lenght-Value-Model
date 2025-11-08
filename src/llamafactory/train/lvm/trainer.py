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

    def _apply_decay_factor(self, value_labels: torch.Tensor, value_mask: torch.Tensor) -> torch.Tensor:
        r"""Apply decay factor to future token values.
        
        Args:
            value_labels: Tensor of shape (batch_size, seq_len) containing the number of future tokens
            value_mask: Tensor of shape (batch_size, seq_len) indicating valid positions
            
        Returns:
            Modified value_labels with decay applied
        """
        if self.finetuning_args.lvm_decay_factor == 1.0:
            return value_labels
            
        # Create decay weights: for each position, apply decay_factor^(n-1) where n is the distance
        batch_size, seq_len = value_labels.shape
        device = value_labels.device
        
        # Create position indices for each sequence
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Calculate decay weights: decay_factor^(n-1) where n is the distance from current token
        # For value_labels[i, j], the distance is value_labels[i, j], so decay is decay_factor^(value_labels[i, j]-1)
        decay_weights = torch.pow(self.finetuning_args.lvm_decay_factor, value_labels)
        
        # Apply decay weights to value_labels
        decayed_labels = 1 - decay_weights
        
        # Only apply decay where mask is valid
        decayed_labels = decayed_labels * value_mask
        
        return decayed_labels

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, torch.Tensor], return_outputs: bool = False, **kwargs
    ):
        value_labels = inputs.pop("value_labels")
        value_mask = inputs.pop("value_mask")
        # Cast labels and masks to float32 to avoid mixed-precision dtype mismatches in loss computation
        value_labels = value_labels
        value_mask = value_mask
        value_preds = self._forward_value(model, inputs)

        # Apply decay factor to future token values
        if self.finetuning_args.lvm_decay_factor != 1.0:
            value_labels = self._apply_decay_factor(value_labels, value_mask)

        # Prepare labels for loss (optionally in log-space)
        if self.finetuning_args.lvm_use_log1p:
            labels_for_loss = torch.log1p(value_labels.clamp_min(0.0))
        else:
            labels_for_loss = value_labels
        
        if self.finetuning_args.lvm_use_sigmoid:
            value_preds = torch.sigmoid(value_preds)
        else:
            value_preds = value_preds
        # Compute token-wise loss
        if getattr(self.finetuning_args, "lvm_loss", "mse") == "smooth_l1":
            per_token_loss = F.smooth_l1_loss(
                value_preds, labels_for_loss, beta=self.finetuning_args.lvm_smooth_l1_beta, reduction="none"
            )
        else:
            per_token_loss = torch.pow(value_preds - labels_for_loss, 2)/2

        mask_sum = value_mask.sum().clamp_min(1.0)
        loss = (per_token_loss * value_mask).sum() / mask_sum
        # rank 0
        # import os
        # if int(os.getenv("LOCAL_RANK", "0")) == 0:
        #     import pdb; pdb.set_trace()
        if return_outputs:
            # For metrics, return predictions in original scale if trained in log-space
            if self.finetuning_args.lvm_use_log1p:
                preds_for_metric = torch.expm1(value_preds).clamp_min(0.0)
            else:
                preds_for_metric = value_preds
            combined_labels = torch.stack((value_labels.detach(), value_mask.detach()), dim=-1)
            return loss, (preds_for_metric.detach(), combined_labels)

        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        has_labels = "value_labels" in inputs and "value_mask" in inputs
        inputs = self._prepare_inputs(inputs)

        value_labels = inputs.pop("value_labels", None)
        value_mask = inputs.pop("value_mask", None)

        with torch.no_grad():
            value_preds = self._forward_value(model, inputs)

        loss = None
        combined_labels = None
        if has_labels and value_labels is not None and value_mask is not None:
            # Cast labels and masks to float32 for consistent loss computation
            value_labels = value_labels
            value_mask = value_mask
            
            # Apply decay factor to future token values
            if self.finetuning_args.lvm_decay_factor != 1.0:
                value_labels = self._apply_decay_factor(value_labels, value_mask)
            
            if self.finetuning_args.lvm_use_log1p:
                labels_for_loss = torch.log1p(value_labels.clamp_min(0.0))
            else:
                labels_for_loss = value_labels

            if self.finetuning_args.lvm_use_sigmoid:
                value_preds = torch.sigmoid(value_preds)
            else:
                value_preds = value_preds

            if getattr(self.finetuning_args, "lvm_loss", "mse") == "smooth_l1":
                per_token_loss = F.smooth_l1_loss(
                    value_preds, labels_for_loss, beta=self.finetuning_args.lvm_smooth_l1_beta, reduction="none"
                )
            else:
                per_token_loss = torch.pow(value_preds - labels_for_loss, 2)/2

            loss = (per_token_loss * value_mask).sum() / value_mask.sum().clamp_min(1.0)
            combined_labels = torch.stack((value_labels, value_mask), dim=-1)

        if prediction_loss_only:
            return loss, None, None

        # Return predictions in original scale for metrics/outputs
        if self.finetuning_args.lvm_use_log1p:
            value_preds = torch.expm1(value_preds).clamp_min(0.0)
        value_preds = value_preds.detach()
        if combined_labels is not None:
            combined_labels = combined_labels.detach()

        return loss, value_preds, combined_labels
