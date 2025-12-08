# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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
from torch import nn

from ...data import LengthValueDataCollator, get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import fix_valuehead_checkpoint
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeLengthMetrics
from .trainer import LengthValueTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def _enable_value_head_high_precision_mode(model: "torch.nn.Module", dtype: torch.dtype = torch.float32) -> None:
    r"""Force value head to run in float32 (inputs and outputs), while keeping the rest of the model unchanged."""
    if not (hasattr(model, "v_head") and hasattr(model.v_head, "summary")):
        raise ValueError("Value head not found in model")

    v_head = model.v_head
    # Ensure parameters of the linear layer are float32
    v_head.summary = v_head.summary.to(dtype)

    def _cast_nested_to_float32(x):
        if isinstance(x, torch.Tensor):
            return x.to(dtype)
        if isinstance(x, (list, tuple)):
            xs = [ _cast_nested_to_float32(xx) for xx in x ]
            return type(x)(xs)
        if isinstance(x, dict):
            return {k: _cast_nested_to_float32(v) for k, v in x.items()}
        return x

    def _pre_hook(module, inputs):
        # Cast all tensor inputs to float32 before the linear layer
        return tuple(_cast_nested_to_float32(i) for i in inputs)

    def _post_hook(module, inputs, output):
        # Ensure outputs stay in float32
        return _cast_nested_to_float32(output)

    # Register hooks on the actual linear layer used as value head
    v_head.summary.register_forward_pre_hook(_pre_hook)
    v_head.summary.register_forward_hook(_post_hook)


def run_lvm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="lvm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

    # Disable dropout in value head
    if hasattr(model, "v_head") and hasattr(model.v_head, "dropout"):
        model.v_head.dropout = nn.Identity()
    else:
        raise ValueError("Value head not found in model")

    # Disable bias in value head
    if hasattr(model, "v_head") and hasattr(model.v_head, "summary"):
        if hasattr(model.v_head.summary, "bias") and model.v_head.summary.bias is not None:
            model.v_head.summary.bias = None

    # Force value head (its linear layer) to run strictly in float32:
    #   - inputs to the linear layer are cast to float32
    #   - the linear transformation itself uses float32 weights
    #   - outputs from the linear layer stay in float32
    _enable_value_head_high_precision_mode(model, dtype=torch.float32)
    
    data_collator = LengthValueDataCollator(
        template=template, model=model, pad_to_multiple_of=8, compute_dtype=model_args.compute_dtype, **tokenizer_module
    )

    # Initialize our Trainer
    trainer = LengthValueTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeLengthMetrics(),
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_mae", f"eval_{key}_rmse"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_mae", "eval_rmse"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
