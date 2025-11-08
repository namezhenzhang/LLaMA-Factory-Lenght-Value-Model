from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class LengthValueDatasetProcessor(DatasetProcessor):
    def _encode_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[bool]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages, system, tools)

        if self.template.efficient_eos:
            response_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        source_len, target_len = infer_seqlen(len(prompt_ids), len(response_ids), self.data_args.cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        response_ids = response_ids[:target_len]

        input_ids = prompt_ids + response_ids
        seq_len = len(input_ids)
        # compute length-to-go targets per token (including prompt final token)
        targets: list[int] = []
        masks: list[bool] = []

        valid_start = max(source_len - 1, 0)
        for idx in range(seq_len):
            remaining = seq_len - idx
            targets.append(int(remaining))
            masks.append(True if idx >= valid_start and idx < seq_len - 1 else False) # ignore the last token <eos>
        return input_ids, targets, masks

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, value_targets, value_mask = self._encode_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append([IGNORE_INDEX] * len(input_ids))
            model_inputs["value_labels"].append(value_targets)
            model_inputs["value_mask"].append(value_mask)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("value_labels:\n{}".format(example.get("value_labels")))
        print("value_mask:\n{}".format(example.get("value_mask")))

