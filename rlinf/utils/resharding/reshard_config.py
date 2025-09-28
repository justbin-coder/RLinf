# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Callable

from megatron.core.transformer import TransformerConfig

from rlinf.utils.convertor.utils import get_mg2hf_convertor

from .utils import get_pp_reshard_fn, get_tp_reshard_fn


@dataclass
class ReshardConfig:
    model_arch: str
    """Supported model arch, valid options are `qwen2.5` and `llama2`."""

    model_config: TransformerConfig

    reshard_weights_format: str = "sglang"
    """Resharding weights format, support sglang, mcore (megatron core)."""

    reshard_tp_size: int = 1
    """Resharding tp size."""

    reshard_pp_size: int = 1
    """Resharding pp size."""

    convert_fn: Callable = None
    """Function to convert the model weights from megatron format to HuggingFace format."""

    tp_reshard_fn: Callable = None
    """Resharding function to use for resharding the model parallelism from tensor_model_parallel_size to reshard_tp_size."""

    pp_reshard_fn: Callable = None
    """Resharding function to use for resharding the model parallelism from pipeline_model_parallel_size to reshard_pp_size."""

    def __post_init__(self):
        if self.model_config.tensor_model_parallel_size < self.reshard_tp_size:
            raise ValueError(
                "Model tp size must be greater than or equal to resharding tp size."
            )
        if self.model_config.tensor_model_parallel_size % self.reshard_tp_size != 0:
            raise ValueError("Model tp size must be divisible by resharding tp size.")

        if self.model_arch is None:
            raise ValueError(
                "Please specify the model_arch, valid options are `qwen2.5` and `llama2`."
            )

        if self.convert_fn is None and self.reshard_weights_format != "mcore":
            self._convertor = get_mg2hf_convertor(self.model_arch, self, strict=True)
            self.convert_fn = self._convertor.convert

        if self.tp_reshard_fn is None:
            self.tp_reshard_fn = get_tp_reshard_fn(self.model_arch)

        if self.pp_reshard_fn is None:
            self.pp_reshard_fn = get_pp_reshard_fn(self.model_arch)
