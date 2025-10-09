# Copyright 2025 the LlamaFactory team.
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction


@dataclass
class ComputeLengthMetrics:
    r"""Compute regression metrics for length value model."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"mae": [], "rmse": [], "seq_mae": [], "seq_rmse": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds = numpify(eval_preds.predictions)
        combined_labels = numpify(eval_preds.label_ids)

        if combined_labels.ndim != preds.ndim + 1 or combined_labels.shape[-1] != 2:
            raise ValueError("Labels must contain stacked values and masks for length metrics.")

        values = combined_labels[..., 0]
        masks = combined_labels[..., 1]

        if preds.shape != values.shape:
            raise ValueError("Predictions and value labels must have the same shape for length metrics.")

        masks = masks.astype(np.float32)
        abs_diff = np.abs(preds - values) * masks
        sq_diff = (preds - values) ** 2 * masks

        denom = np.clip(masks.sum(), a_min=1.0, a_max=None)
        mae = np.sum(abs_diff) / denom
        rmse = np.sqrt(np.sum(sq_diff) / denom)

        # per-sequence statistics (average over valid tokens of each sequence, then mean over samples)
        if preds.ndim == 1:
            seq_denoms = np.array([denom])
            seq_mae_vals = np.array([mae])
            seq_rmse_vals = np.array([rmse])
        else:
            seq_axes = tuple(range(1, preds.ndim))
            seq_denoms = np.clip(masks.sum(axis=seq_axes), a_min=1.0, a_max=None)
            seq_mae_vals = np.sum(abs_diff, axis=seq_axes) / seq_denoms
            seq_rmse_vals = np.sqrt(np.sum(sq_diff, axis=seq_axes) / seq_denoms)
        seq_mae = float(np.mean(seq_mae_vals))
        seq_rmse = float(np.mean(seq_rmse_vals))

        self.score_dict["token_mean_mae"].append(mae)
        self.score_dict["token_mean_rmse"].append(rmse)
        self.score_dict["token_mean_seq_mean_mae"].append(seq_mae)
        self.score_dict["token_mean_seq_mean_rmse"].append(seq_rmse)

        if compute_result:
            return self._dump()
