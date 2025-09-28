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

import logging
from enum import Enum, auto
from typing import Dict, List, overload

from omegaconf import DictConfig

from rlinf.scheduler import (
    Cluster,
    FlexiblePlacementStrategy,
    PackedPlacementStrategy,
    PlacementStrategy,
)


class PlacementMode(Enum):
    COLLOCATED = auto()
    DISAGGREGATED = auto()
    HYBRID = auto()


class ComponentPlacement:
    """Base component placement for parsing config."""

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Parsing component placement configuration.

        Args:
            config (DictConfig): The configuration dictionary for the component placement.
        """
        self._config = config
        self._placement_config: DictConfig = config.cluster.component_placement
        self._cluster_num_gpus = cluster.num_accelerators_in_cluster
        self._components: List[str] = []
        self._component_gpu_map: Dict[str, List[int]] = {}

        # Each line of component placement config looks like:
        # actor,inference: 0-4, which means both the actor and inference groups occupy GPU 0 to 4
        # Alternatively, "all" can be used to specify all GPUs
        for components in self._placement_config.keys():
            components_gpus: str = self._placement_config[components]
            components = components.split(",")
            components = [c.strip() for c in components]
            components_gpus = self._parse_gpu_ids(components_gpus, components)

            for component in components:
                self._components.append(component)
                self._component_gpu_map[component] = components_gpus

            self._placements: Dict[str, PlacementStrategy] = {}
            self._placement_mode: PlacementMode = None

    def _parse_gpu_ids(
        self, components_gpus: str, component_names: List[str]
    ) -> List[int]:
        """Parse a string of GPU IDs into a list of integers.

        Args:
            components_gpus (str): A string representing GPU IDs. The string can either be "all", representing all GPUs, or a comma-separated list of GPU IDs and ranges (e.g., "0,1,2-4").
            component_names (List[str]): The names of the components for error reporting.

        Returns:
            List[int]: A list of GPU IDs as integers.
        """
        gpu_ids: List[int] = []
        if components_gpus == "all":
            gpu_ids = list(range(0, self._cluster_num_gpus))
        else:
            # If the GPU placement is a single number
            # Omegaconf will parse it as an integer instead of a string
            components_gpus = str(components_gpus)
            # First split by comma
            gpu_id_ranges = components_gpus.split(",")
            for gpu_id_range in gpu_id_ranges:
                gpu_id_range = gpu_id_range.strip()
                if gpu_id_range == "":
                    continue
                # Then split by hyphen to get the start and end of the range
                gpu_id_range = gpu_id_range.split("-")
                try:
                    if len(gpu_id_range) == 1:
                        start_gpu = int(gpu_id_range[0])
                        end_gpu = start_gpu
                    elif len(gpu_id_range) == 2:
                        start_gpu = int(gpu_id_range[0])
                        end_gpu = int(gpu_id_range[1])
                    else:
                        raise ValueError
                except (ValueError, IndexError):
                    raise ValueError(
                        f'Invalid GPU placement format for components {component_names}: {components_gpus}, expected format: "a,b,c-d" or "all"'
                    )
                assert end_gpu >= start_gpu, (
                    f"Start GPU ID {start_gpu} must be less than or equal to end GPU ID {end_gpu}."
                )
                assert start_gpu < self._cluster_num_gpus, (
                    f"Start GPU ID {start_gpu} must be less than total number of GPUs {self._cluster_num_gpus}."
                )
                assert end_gpu < self._cluster_num_gpus, (
                    f"End GPU ID {end_gpu} must be less than total number of GPUs {self._cluster_num_gpus}."
                )
                gpu_ids.extend(list(range(start_gpu, end_gpu + 1)))
        return gpu_ids

    @property
    def placement_mode(self):
        """Get the placement mode for the component.

        Returns:
            PlacementMode: The placement mode for the component.
        """
        return self._placement_mode

    def get_world_size(self, component_name: str):
        """Get the world size for a specific component.

        Args:
            component_name (str): The name of the component.

        Returns:
            int: The world size for the specified component.
        """
        assert component_name in self._component_gpu_map, (
            f"Unknown component name: {component_name}"
        )
        return len(self._component_gpu_map[component_name])

    @overload
    def _generate_placements(self):
        raise NotImplementedError

    def get_strategy(self, component_name: str):
        """Get the placement strategy for a component based on the configuration.

        Args:
            component_name (str): The name of the component to retrieve the placement strategy for.

        Returns:
            PackedPlacementStrategy: The placement strategy for the specified component.
        """
        if len(self._placements.keys()) == 0:
            self._generate_placements()
        assert component_name in self._placements, (
            f"Component {component_name} does not exist in {type(self)} with placement mode {self._placement_mode}"
        )
        return self._placements[component_name]


class HybridComponentPlacement(ComponentPlacement):
    """Hybrid component placement that allows components to run on any sets of GPUs."""

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Initialize HybridComponentPlacement

        Args:
            config (DictConfig): The configuration dictionary.
        """
        super().__init__(config, cluster)
        self._placement_mode = PlacementMode.HYBRID

    def _generate_placements(self):
        for component_name, component_gpus in self._component_gpu_map.items():
            self._placements[component_name] = FlexiblePlacementStrategy(
                [[gpu_id] for gpu_id in component_gpus]
            )


class ModelParallelComponentPlacement(ComponentPlacement):
    """Component placement for model-parallel components.

    The components must be actor, rollout, and optionally inference, whose GPUs must be continuous.

    This placement supports both collocated and disaggregated modes.

    In the collocated mode, all components share the same set of GPUs. In particular, the rollout group is specially placed in a strided manner to enable fast cudaIPC-based weight sync.
    In the disaggregated mode, each component has its own dedicated set of GPUs.

    In the collocated mode, only actor and rollout exist. While in the disaggregated mode, actor, rollout, and inference should all exist.
    """

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Initialize ModelParallelComponentPlacement

        Args:
            config (DictConfig): The configuration dictionary for the component placement.
        """
        super().__init__(config, cluster)

        self._actor_gpus = self._component_gpu_map.get("actor", None)
        self._inference_gpus = self._component_gpu_map.get("inference", None)
        self._rollout_gpus = self._component_gpu_map.get("rollout", None)
        assert self._actor_gpus is not None, (
            "Actor GPUs must be specified in the component_placement config."
        )
        assert self._rollout_gpus is not None, (
            "Rollout GPUs must be specified in the component_placement config."
        )
        assert self._actor_gpus == list(
            range(self._actor_gpus[0], self._actor_gpus[-1] + 1)
        ), f"Actor GPUs {self._actor_gpus} must be continuous."
        assert self._rollout_gpus == list(
            range(self._rollout_gpus[0], self._rollout_gpus[-1] + 1)
        ), f"Rollout GPUs {self._rollout_gpus} must be continuous."
        if self._inference_gpus is not None:
            assert self._inference_gpus == list(
                range(self._inference_gpus[0], self._inference_gpus[-1] + 1)
            ), f"Inference GPUs {self._inference_gpus} must be continuous."

        self._actor_num_gpus = len(self._actor_gpus)
        self._inference_num_gpus = (
            len(self._inference_gpus) if self._inference_gpus else 0
        )
        self._rollout_num_gpus = len(self._rollout_gpus)

        if self._is_collocated():
            assert self._inference_gpus is None, (
                "Inference GPUs must not be specified in collocated mode."
            )
            self._placement_mode = PlacementMode.COLLOCATED
            logging.info("Running in collocated mode")
        elif self._is_disaggregated():
            if self._inference_gpus is not None:
                assert self.inference_tp_size <= self.inference_world_size, (
                    f"Inference TP size {self.inference_tp_size} must be less than or equal to Inference world size {self.inference_world_size}."
                )
                assert self._config.algorithm.recompute_logprobs, (
                    f"algorithm.recompute_logprobs has been set to false, which disables inference. So inference GPUs {self._inference_gpus} must not be specified."
                )
            self._placement_mode = PlacementMode.DISAGGREGATED
            logging.info("Running in disaggregated mode")
        else:
            raise ValueError(
                f"The specified placement does not match either the collocated mode (all the components use the same GPUs) or the disaggregated mode (all the components use completely different GPUs), but got {self._component_gpu_map}"
            )

        # Sanity checking
        assert self.actor_tp_size <= self.actor_world_size, (
            f"Actor TP size {self.actor_tp_size} must be less than or equal to Actor world size {self.actor_world_size}."
        )
        assert self.rollout_tp_size <= self.rollout_world_size, (
            f"Rollout TP size {self.rollout_tp_size} must be less than or equal to Rollout world size {self.rollout_world_size}."
        )

    def _is_collocated(self):
        if self._actor_gpus == self._rollout_gpus:
            return True
        return False

    def _is_disaggregated(self):
        actor_gpu_set = set(self._actor_gpus)
        rollout_gpu_set = set(self._rollout_gpus)
        inference_gpu_set = (
            [] if self._inference_gpus is None else set(self._inference_gpus)
        )
        return (
            actor_gpu_set.isdisjoint(rollout_gpu_set)
            and actor_gpu_set.isdisjoint(inference_gpu_set)
            and rollout_gpu_set.isdisjoint(inference_gpu_set)
        )

    def _generate_placements(self):
        if self._placement_mode == PlacementMode.COLLOCATED:
            self._placements["actor"] = PackedPlacementStrategy(
                self._actor_gpus[0], self._actor_gpus[-1]
            )

            actor_tp_size = self._config.actor.model.tensor_model_parallel_size
            rollout_tp_size = self._config.rollout.tensor_parallel_size
            if actor_tp_size > rollout_tp_size:
                assert actor_tp_size % rollout_tp_size == 0, (
                    f"Actor TP size ({actor_tp_size}) must be divisible by Rollout TP size ({rollout_tp_size})"
                )
            stride = (
                self.actor_tp_size // self.rollout_tp_size
                if self.actor_tp_size > self.rollout_tp_size
                else 1
            )
            stride = actor_tp_size // rollout_tp_size
            self._placements["rollout"] = PackedPlacementStrategy(
                self._rollout_gpus[0],
                self._rollout_gpus[-1],
                num_accelerators_per_process=rollout_tp_size,
                stride=stride,
            )
        elif self._placement_mode == PlacementMode.DISAGGREGATED:
            # Generate continuous placement strategies for components in a cluster.
            num_gpus_per_rollout_dp = len(self._rollout_gpus) // self.rollout_dp_size
            self._placements["rollout"] = PackedPlacementStrategy(
                self._rollout_gpus[0],
                self._rollout_gpus[-1],
                num_accelerators_per_process=num_gpus_per_rollout_dp,
            )
            if self._inference_gpus is not None:
                self._placements["inference"] = PackedPlacementStrategy(
                    self._inference_gpus[0], self._inference_gpus[-1]
                )
            self._placements["actor"] = PackedPlacementStrategy(
                self._actor_gpus[0], self._actor_gpus[-1]
            )

    @property
    def is_disaggregated(self):
        return self._placement_mode == PlacementMode.DISAGGREGATED

    @property
    def has_dedicated_inference(self):
        return (
            self._placement_mode == PlacementMode.DISAGGREGATED
            and self._inference_gpus is not None
        )

    @property
    def actor_dp_size(self) -> int:
        return self._actor_num_gpus // (
            self._config.actor.model.get("tensor_model_parallel_size", 1)
            * self._config.actor.model.get("context_parallel_size", 1)
            * self._config.actor.model.get("pipeline_model_parallel_size", 1)
        )

    @property
    def actor_tp_size(self) -> int:
        return self._config.actor.model.get("tensor_model_parallel_size", 1)

    @property
    def actor_pp_size(self) -> int:
        return self._config.actor.model.get("pipeline_model_parallel_size", 1)

    @property
    def actor_world_size(self) -> int:
        return self._actor_num_gpus

    @property
    def inference_tp_size(self) -> int:
        if (
            hasattr(self._config, "inference")
            and hasattr(self._config.inference, "model")
            and hasattr(self._config.inference.model, "tensor_model_parallel_size")
        ):
            return self._config.inference.model.get("tensor_model_parallel_size", 1)
        else:
            return self.actor_tp_size

    @property
    def inference_pp_size(self) -> int:
        if (
            hasattr(self._config, "inference")
            and hasattr(self._config.inference, "model")
            and hasattr(self._config.inference.model, "pipeline_model_parallel_size")
        ):
            return self._config.inference.model.get("pipeline_model_parallel_size", 1)
        else:
            return self.actor_pp_size

    @property
    def inference_dp_size(self) -> int:
        return self._inference_num_gpus // (
            self.inference_tp_size * self.inference_pp_size
        )

    @property
    def inference_world_size(self) -> int:
        return self._inference_num_gpus

    @property
    def rollout_dp_size(self) -> int:
        return self._rollout_num_gpus // (
            self._config.rollout.get("tensor_parallel_size", 1)
            * self._config.rollout.get("pipeline_parallel_size", 1)
        )

    @property
    def rollout_tp_size(self) -> int:
        return self._config.rollout.get("tensor_parallel_size", 1)

    @property
    def rollout_world_size(self) -> int:
        return self._rollout_num_gpus
