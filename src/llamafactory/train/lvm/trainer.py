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

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, torch.Tensor], return_outputs: bool = False, **kwargs
    ):
        value_labels = inputs.pop("value_labels")
        value_mask = inputs.pop("value_mask")
        value_preds = self._forward_value(model, inputs)

        value_labels = value_labels.to(value_preds.dtype)
        value_mask = value_mask.to(value_preds.dtype)

        mse = F.mse_loss(value_preds.float(), value_labels, reduction="none")
        mask_sum = value_mask.sum().clamp_min(1.0)
        loss = (mse * value_mask).sum() / mask_sum

        if return_outputs:
            combined_labels = torch.stack((value_labels.detach(), value_mask.detach()), dim=-1)
            return loss, (value_preds.detach(), combined_labels)

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
            value_labels = value_labels.to(value_preds.dtype)
            value_mask = value_mask.to(value_preds.dtype)
            mse = F.mse_loss(value_preds.float(), value_labels, reduction="none")
            loss = (mse * value_mask).sum() / value_mask.sum().clamp_min(1.0)
            combined_labels = torch.stack((value_labels, value_mask), dim=-1)

        if prediction_loss_only:
            return loss, None, None

        value_preds = value_preds.detach()
        if combined_labels is not None:
            combined_labels = combined_labels.detach()

        return loss, value_preds, combined_labels
