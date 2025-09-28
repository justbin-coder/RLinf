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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from vllm.outputs import CompletionOutput
    from vllm.outputs import RequestOutput as VllmRequestOutput

from rlinf.data.datasets import batch_pad_to_fixed_len
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    split_list,
)


def get_batch_size(
    batch: Dict[str, torch.Tensor], batch_tensor_key: str = "input_ids"
) -> int:
    """Get the batch size from the batch dictionary."""
    return batch[batch_tensor_key].size(0)


def get_seq_length(
    batch: Dict[str, torch.Tensor], batch_tensor_key: str = "input_ids"
) -> int:
    """Get the sequence length from the batch dictionary."""
    return batch[batch_tensor_key].size(1)


@dataclass
class RolloutRequest:
    """
    Attr
    input_ids: List of input token IDs for rollout
    n: Number of completions to generate for each input
    idx: List of unique identifiers for the requests, used for tracking
    input_lengths: List of lengths of the input sequences, corresponding to input_ids
    image_data: list of image data (bytes or URLs) for multimodal inputs
    answers: Optional list of answers for the requests, if available
    """

    n: int
    input_ids: List[List[int]]
    answers: List[str]
    image_data: Union[List[List[bytes]], List[List[str]]]

    def repeat_and_split(
        self, rollout_batch_size: Optional[int] = None
    ) -> List["RolloutRequest"]:
        input_ids, answers, image_data = zip(
            *[
                (input_id, answer, image_data)
                for input_id, answer, image_data in zip(
                    self.input_ids, self.answers, self.image_data
                )
                for _ in range(self.n)
            ]
        )
        input_ids, answers, image_data = (
            list(input_ids),
            list(answers),
            list(image_data),
        )

        # Split input ids based on rollout_batch_size_per_gpu
        if rollout_batch_size is None:
            num_batches = 1
        else:
            assert len(input_ids) % rollout_batch_size == 0, (
                f"Input IDs length {len(input_ids)} is not divisible by rollout batch size {rollout_batch_size}"
            )
            num_batches = len(input_ids) // rollout_batch_size

        splitted_requests = []
        input_ids_split_list = split_list(input_ids, num_batches)
        answers_split_list = split_list(answers, num_batches)
        image_data_split_list = split_list(image_data, num_batches)

        for input_ids_batch, answers_batch, image_data_batch in zip(
            input_ids_split_list, answers_split_list, image_data_split_list
        ):
            request = RolloutRequest(
                n=self.n,
                input_ids=input_ids_batch,
                answers=answers_batch,
                image_data=image_data_batch,
            )
            splitted_requests.append(request)

        return splitted_requests


class CompletionInfo:
    def __init__(self, logger=None):
        self.input_ids: Dict[int, List[int]] = {}  # hash -> input token IDs
        self.complete_num: Dict[int, int] = {}  # hash -> completion count
        self.results: Dict[int, List[Dict]] = {}  # hash -> list of results

        self.num_requests: int = 0
        self.num_completed: int = 0
        self._num_returned: int = 0  # Number of results returned

        self.n_result_each_request: int = 0

        self.logger = logger

    def hash(self, token_ids: List[int]) -> int:
        """Generate a hash for the token IDs."""
        return hash(tuple(token_ids))

    def clear(self):
        self.complete_num.clear()
        self.input_ids.clear()
        self.results.clear()
        self.num_requests = 0
        self.num_completed = 0
        self._num_returned = 0

    def add_request(self, req: RolloutRequest):
        """Add a new request to the completion info."""
        if self.n_result_each_request != 0:
            assert self.n_result_each_request == req.n
        else:
            self.n_result_each_request = req.n

        self.num_requests += len(req.input_ids)

        for ids in req.input_ids:
            hash_id = self.hash(ids)
            if hash_id not in self.input_ids:
                self.input_ids[hash_id] = ids
                self.complete_num[hash_id] = 0
                self.results[hash_id] = []
            else:
                assert self.input_ids[hash_id] == ids, (
                    "Input IDs mismatch for existing hash ID"
                )

    def clear_and_set(self, req: RolloutRequest):
        self.clear()
        self.add_request(req)

    def is_empty(self) -> bool:
        return len(self.complete_num) == 0 and len(self.results) == 0

    def record_result(self, token_ids: List[int], result: Dict) -> int:
        hash_id = self.hash(token_ids)

        self.complete_num[hash_id] += 1
        self.results[hash_id].append(result)

        if self.complete_num[hash_id] == self.n_result_each_request:
            self.num_completed += 1
            if self.logger is not None:
                self.logger.debug(f"Completed all rollouts for hash: {hash_id}")

        return self.complete_num[hash_id]

    def is_completed(self, hash_id: int) -> bool:
        return self.complete_num[hash_id] == self.n_result_each_request

    def get_results(self, hash_id: int) -> List[Dict]:
        """Get the results for the given token IDs."""
        assert hash_id in self.results, "Hash ID not found in results"
        assert self.complete_num[hash_id] == self.n_result_each_request, (
            "Not all results for this hash ID are completed"
        )
        value = self.results.pop(hash_id)
        return value

    def record_returned(self):
        """Record that a result has been returned."""
        self._num_returned += 1
        if self.logger is not None:
            self.logger.debug(
                f"Returned / Completed: {self._num_returned} / {self.num_completed}"
            )

    def all_returned(self) -> bool:
        """Check if all results have been returned."""
        return self._num_returned == self.num_requests


@dataclass(kw_only=True)
class RolloutResult:
    """
    Rollout Result
    """

    num_sequence: int
    group_size: int
    prompt_lengths: List[int]
    prompt_ids: List[List[int]]
    response_lengths: List[int]
    response_ids: List[List[int]]
    is_end: List[bool]
    rewards: Optional[List[float] | torch.Tensor] = None
    advantages: Optional[List[float] | torch.Tensor] = None
    prompt_texts: Optional[List[str]] = None
    response_texts: Optional[List[str]] = None
    answers: Optional[List[str | dict]] = None
    image_data: Optional[Union[List[List[bytes]], List[List[str]]]] = None
    # Inference
    # Only set when recompute_logprobs is False
    rollout_logprobs: Optional[List[List[float]]] = None
    prev_logprobs: Optional[torch.Tensor] = None
    ref_logprobs: Optional[torch.Tensor] = None

    @property
    def batch_size(self):
        return self.num_sequence // self.group_size

    @staticmethod
    def _get_attention_masks_and_position_ids(
        prompt_lengths: torch.Tensor,
        response_lengths: torch.Tensor,
        max_prompt_len: int,
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = prompt_lengths.size(0)

        # =========================
        # Attention Mask
        # =========================
        arange_ids = (
            torch.arange(total_len).unsqueeze(0).expand(B, -1)
        )  # [B, total_len]

        # Compute the start and end positions of the prompt and response tokens
        prompt_start = max_prompt_len - prompt_lengths  # [B]
        response_end = max_prompt_len + response_lengths  # [B]

        # Broadcast [B, total_len]
        prompt_start = prompt_start.unsqueeze(1)
        response_end = response_end.unsqueeze(1)

        attention_mask = (arange_ids >= prompt_start) & (arange_ids < response_end)

        # =========================
        # Position IDs
        # =========================
        position_ids = torch.zeros_like(arange_ids)

        for i in range(B):
            ps = prompt_start[i].item()
            position_ids[i, ps:] = torch.arange(total_len - ps)

        return attention_mask, position_ids

    @staticmethod
    def from_vllm_results(
        group_size: int,
        results: List["VllmRequestOutput"],
        answers: Optional[List[List[int]]] = None,
        return_logprobs: bool = False,
    ) -> "RolloutResult":
        def get_logprobs(
            response_ids: List[int], output: "CompletionOutput"
        ) -> List[float]:
            logprobs = []
            returned_logprobs = output.logprobs
            assert logprobs is not None, (
                "vllm returned None logprobs, while return_logprobs is set."
            )
            for i, logprob in enumerate(returned_logprobs):
                logprobs.append(logprob[response_ids[i]].logprob)
            return logprobs

        num_sequences = len(results)

        prompt_lengths = []
        prompt_ids = []
        response_lengths = []
        response_ids = []
        logprobs = []
        is_end = []
        for _, res in enumerate(results):
            if res.prompt_token_ids is not None:
                prompt_ids.append(res.prompt_token_ids)
                prompt_lengths.append(len(res.prompt_token_ids))
            else:
                return NotImplementedError("vllm should return tokenized prompt.")
            response_id = list(res.outputs[0].token_ids)
            response_ids.append(response_id)
            response_lengths.append(len(response_id))
            is_end.append(res.finished)
            if return_logprobs:
                logprobs.append(get_logprobs(response_id, res.outputs[0]))
        result: RolloutResult = RolloutResult(
            group_size=group_size,
            num_sequence=num_sequences,
            answers=answers,
            prompt_ids=prompt_ids,
            prompt_lengths=prompt_lengths,
            response_ids=response_ids,
            response_lengths=response_lengths,
            is_end=is_end,
        )
        if return_logprobs:
            result.rollout_logprobs = logprobs
        return result

    @staticmethod
    def from_sglang_results(
        results: List[Dict],
        group_size: int,
        input_ids: List[List[int]],
        answers: Optional[List[List[int]]] = None,
        image_data: Optional[Union[List[List[bytes]], List[List[str]]]] = None,
        return_logprobs: bool = False,
    ) -> "RolloutResult":
        """Create a MathRolloutResult from the given results and input IDs.

        Args:
            results (List[Dict]): The rollout results from the model.
            input_ids (List[List[int]]): The input IDs for the prompts.
            return_logprobs (bool): Whether to return log probabilities.
        """
        assert len(results) == len(input_ids), (
            f"Results length {len(results)} does not match input_ids length {len(input_ids)}"
        )
        assert isinstance(results, list) and all(
            isinstance(res, dict) for res in results
        ), "Results should be a list of dictionaries."
        assert isinstance(input_ids, list) and all(
            isinstance(id_list, list) for id_list in input_ids
        ), "Input IDs should be a list of lists."
        result = RolloutResult(
            num_sequence=len(results),
            group_size=group_size,
            prompt_lengths=[len(input_id) for input_id in input_ids],
            prompt_ids=input_ids,
            response_lengths=[len(res["output_ids"]) for res in results],
            response_ids=[res["output_ids"] for res in results],
            answers=answers,
            image_data=image_data,
            is_end=[
                res["meta_info"]["finish_reason"]["type"] == "stop" for res in results
            ],
        )
        if return_logprobs:
            logprobs = [
                [item[0] for item in res["meta_info"]["output_token_logprobs"]]
                for res in results
            ]
            result.rollout_logprobs = logprobs
        return result

    @staticmethod
    def merge_result_list(
        rollout_results: List["RolloutResult"],
    ) -> "RolloutResult":
        assert len(rollout_results) > 0, "No rollout results to merge."
        if len(rollout_results) == 1:
            return rollout_results[0]
        merged_result = RolloutResult(
            num_sequence=sum(res.num_sequence for res in rollout_results),
            group_size=rollout_results[0].group_size,
            prompt_lengths=[],
            prompt_ids=[],
            response_lengths=[],
            response_ids=[],
            is_end=[],
        )

        def merge_tensor(dst_tensor: torch.Tensor, src_tensor: torch.Tensor):
            assert dst_tensor is None or torch.is_tensor(dst_tensor), (
                f"Expected tensor, got {type(dst_tensor)}"
            )
            assert torch.is_tensor(src_tensor), (
                f"Expected tensor, got {type(src_tensor)}"
            )
            if dst_tensor is None:
                return src_tensor
            else:
                return torch.cat([dst_tensor, src_tensor], dim=0)

        def merge_list(dst_list: List, src_list: List):
            assert dst_list is None or isinstance(dst_list, list), (
                f"Expected list, got {type(dst_list)}"
            )
            assert isinstance(src_list, list), f"Expected list, got {type(src_list)}"
            if dst_list is None:
                return src_list
            else:
                dst_list.extend(src_list)
                return dst_list

        for res in rollout_results:
            merged_result.prompt_lengths.extend(res.prompt_lengths)
            merged_result.prompt_ids.extend(res.prompt_ids)
            merged_result.response_lengths.extend(res.response_lengths)
            merged_result.response_ids.extend(res.response_ids)
            merged_result.is_end.extend(res.is_end)
            if res.answers is not None:
                merged_result.answers = merge_list(merged_result.answers, res.answers)
            if res.advantages is not None:
                if isinstance(res.advantages, list):
                    merged_result.advantages = merge_list(
                        merged_result.advantages, res.advantages
                    )
                elif isinstance(res.advantages, torch.Tensor):
                    merged_result.advantages = merge_tensor(
                        merged_result.advantages, res.advantages
                    )
                else:
                    raise ValueError(
                        f"Wrong type of advantages {type(merged_result.advantages)}"
                    )
            if res.rewards is not None:
                if isinstance(res.rewards, list):
                    merged_result.rewards = merge_list(
                        merged_result.rewards, res.rewards
                    )
                elif isinstance(res.rewards, torch.Tensor):
                    merged_result.rewards = merge_tensor(
                        merged_result.rewards, res.rewards
                    )
                else:
                    raise ValueError(
                        f"Wrong type of rewards {type(merged_result.rewards)}"
                    )
            if res.rollout_logprobs is not None:
                merged_result.rollout_logprobs = merge_list(
                    merged_result.rollout_logprobs, res.rollout_logprobs
                )
            if res.prev_logprobs is not None:
                merged_result.prev_logprobs = merge_tensor(
                    merged_result.prev_logprobs, res.prev_logprobs
                )
            if res.ref_logprobs is not None:
                merged_result.ref_logprobs = merge_tensor(
                    merged_result.ref_logprobs, res.ref_logprobs
                )

        return merged_result

    @staticmethod
    def split_result_list_by_group(
        rollout_results: List["RolloutResult"],
    ) -> List["RolloutResult"]:
        """
        Split RolloutResult objects by group_size.

        If input has only one RolloutResult, split it into multiple RolloutResult objects by group_size.
        If input has multiple RolloutResult objects, split each one and merge the results.

        Args:
            rollout_results: List of input RolloutResult objects

        Returns:
            List of RolloutResult objects grouped by group_size
        """
        assert len(rollout_results) > 0, "No rollout results to split."

        all_split_results = []

        for rollout_result in rollout_results:
            split_results = RolloutResult._split_single_result_by_group(rollout_result)
            all_split_results.extend(split_results)

        return all_split_results

    @staticmethod
    def _split_single_result_by_group(
        rollout_result: "RolloutResult",
    ) -> List["RolloutResult"]:
        """
        Split a single RolloutResult into multiple RolloutResult objects by group_size.

        Args:
            rollout_result: The RolloutResult to be split

        Returns:
            List of split RolloutResult objects
        """
        group_size = rollout_result.group_size
        num_sequence = rollout_result.num_sequence

        assert num_sequence % group_size == 0, (
            f"num_sequence ({num_sequence}) must be divisible by group_size ({group_size})"
        )

        num_groups = num_sequence // group_size
        split_results = []

        # Split list fields
        prompt_lengths_split = split_list(rollout_result.prompt_lengths, num_groups)
        prompt_ids_split = split_list(rollout_result.prompt_ids, num_groups)
        response_lengths_split = split_list(rollout_result.response_lengths, num_groups)
        response_ids_split = split_list(rollout_result.response_ids, num_groups)
        is_end_split = split_list(rollout_result.is_end, num_groups)

        # Handle optional fields
        answers_split = None
        if rollout_result.answers is not None:
            answers_split = split_list(rollout_result.answers, num_groups)

        image_data_split = None
        if rollout_result.image_data is not None:
            image_data_split = split_list(rollout_result.image_data, num_groups)

        prompt_texts_split = None
        if rollout_result.prompt_texts is not None:
            prompt_texts_split = split_list(rollout_result.prompt_texts, num_groups)

        response_texts_split = None
        if rollout_result.response_texts is not None:
            response_texts_split = split_list(rollout_result.response_texts, num_groups)

        rollout_logprobs_split = None
        if rollout_result.rollout_logprobs is not None:
            rollout_logprobs_split = split_list(
                rollout_result.rollout_logprobs, num_groups
            )

        # Handle tensor fields
        rewards_split = None
        if rollout_result.rewards is not None:
            if isinstance(rollout_result.rewards, torch.Tensor):
                rewards_split = torch.chunk(rollout_result.rewards, num_groups, dim=0)
            else:
                rewards_split = split_list(rollout_result.rewards, num_groups)

        advantages_split = None
        if rollout_result.advantages is not None:
            if isinstance(rollout_result.advantages, torch.Tensor):
                advantages_split = torch.chunk(
                    rollout_result.advantages, num_groups, dim=0
                )
            else:
                advantages_split = split_list(rollout_result.advantages, num_groups)

        prev_logprobs_split = None
        if rollout_result.prev_logprobs is not None:
            prev_logprobs_split = torch.chunk(
                rollout_result.prev_logprobs, num_groups, dim=0
            )

        ref_logprobs_split = None
        if rollout_result.ref_logprobs is not None:
            ref_logprobs_split = torch.chunk(
                rollout_result.ref_logprobs, num_groups, dim=0
            )

        # Create split RolloutResult objects
        for i in range(num_groups):
            split_result = RolloutResult(
                num_sequence=group_size,
                group_size=group_size,
                prompt_lengths=prompt_lengths_split[i],
                prompt_ids=prompt_ids_split[i],
                response_lengths=response_lengths_split[i],
                response_ids=response_ids_split[i],
                is_end=is_end_split[i],
                answers=answers_split[i] if answers_split is not None else None,
                image_data=image_data_split[i]
                if image_data_split is not None
                else None,
                prompt_texts=prompt_texts_split[i]
                if prompt_texts_split is not None
                else None,
                response_texts=response_texts_split[i]
                if response_texts_split is not None
                else None,
                rollout_logprobs=rollout_logprobs_split[i]
                if rollout_logprobs_split is not None
                else None,
                rewards=rewards_split[i] if rewards_split is not None else None,
                advantages=advantages_split[i]
                if advantages_split is not None
                else None,
                prev_logprobs=prev_logprobs_split[i]
                if prev_logprobs_split is not None
                else None,
                ref_logprobs=ref_logprobs_split[i]
                if ref_logprobs_split is not None
                else None,
            )
            split_results.append(split_result)

        return split_results

    def to_actor_batch(
        self,
        data_seq_length: int,
        training_seq_length: int,
        pad_token: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Transform the rollout result into a format suitable for the actor.

        Args:
            data_seq_length (int): Maximum prompt length, e.g., 1024.
            training_seq_length (int): Total sequence length for training, e.g., 8192.
                The maximum response length is calculated as `training_seq_length - data_seq_length`.
            pad_token (int): Token used for padding, e.g., `tokenizer.pad_token_id`.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys:

            input_ids (torch.Tensor):
                Concatenated prompt and response token IDs,
                shape ``[batch_size, training_seq_length]``.

            attention_mask (torch.Tensor):
                Attention mask for the input sequence,
                shape ``[batch_size, training_seq_length]``.

            is_end (torch.Tensor):
                Boolean tensor indicating whether the sequence ends,
                shape ``[batch_size]``.

            position_ids (torch.Tensor):
                Position IDs for the input sequence,
                shape ``[batch_size, training_seq_length]``.

            prompt_lengths (torch.Tensor):
                Lengths of the prompt sequences,
                shape ``[batch_size]``.

            response_lengths (torch.Tensor):
                Lengths of the response sequences,
                shape ``[batch_size]``.

            advantages (torch.Tensor), optional:
                Advantage values for the responses,
                shape ``[batch_size, training_seq_length - data_seq_length]``.
        """

        # len = training_seq_length: input_ids, attention_mask, position_ids
        #           [prompt_padding, prompt_ids,    response_ids, ... ,response_padding]
        #           |<-- padding -->|<-- pmp len -->|<-- resp len --->|<-- padding --->|
        #           |<---- cfg.data.seq_length ---->|
        #           |<------------------ cfg.runner.seq_length --------------------->|

        # len = training_seq_length - data_seq_length: advantage, prev_logprobs, ref_logprobs
        # each row: [response_ids, ...,                , response_padding]
        #           |<----- true response length ----->|<--- padding --->|
        #           |<-- cfg.runner.seq_length - cfg.data.seq_length ->|

        max_response_len = training_seq_length - data_seq_length

        prompt_lengths = torch.tensor(self.prompt_lengths)
        response_lengths = torch.tensor(self.response_lengths)
        is_end = torch.tensor(self.is_end, dtype=torch.bool)

        attention_mask, position_ids = self._get_attention_masks_and_position_ids(
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            max_prompt_len=data_seq_length,
            total_len=training_seq_length,
        )

        prompt_ids = batch_pad_to_fixed_len(
            [torch.as_tensor(ids, dtype=torch.long) for ids in self.prompt_ids],
            max_batch_len=data_seq_length,
            pad_token=pad_token,
            left_pad=True,
        )

        response_ids = batch_pad_to_fixed_len(
            [torch.as_tensor(ids, dtype=torch.long) for ids in self.response_ids],
            max_batch_len=max_response_len,
            pad_token=pad_token,
        )
        input_ids = torch.cat(
            [prompt_ids, response_ids], dim=1
        )  # [B, training_seq_length]

        batch = {
            "input_ids": input_ids.cuda(),
            "attention_mask": attention_mask.cuda(),
            "is_end": is_end.cuda(),
            "position_ids": position_ids.cuda(),
            "prompt_lengths": prompt_lengths.cuda(),
            "response_lengths": response_lengths.cuda(),
        }

        if self.advantages is not None:
            if isinstance(self.advantages, torch.Tensor):
                batch["advantages"] = self.advantages.cuda()
            else:
                response_attention_mask = attention_mask[
                    :, -max_response_len:
                ]  # [B, max_response_len]
                advantages = torch.tensor(self.advantages, dtype=torch.float32).reshape(
                    -1, 1
                )  # [B, 1]
                advantages = response_attention_mask.float().cuda() * advantages.cuda()
                batch["advantages"] = advantages.cuda()

        if self.prev_logprobs is not None:
            batch["prev_logprobs"] = self.prev_logprobs.cuda()

        if self.ref_logprobs is not None:
            batch["ref_logprobs"] = self.ref_logprobs.cuda()

        if self.rewards is not None:
            batch["rewards"] = self.rewards.cuda()

        if self.rollout_logprobs is not None:
            logprobs = batch_pad_to_fixed_len(
                [
                    torch.as_tensor(logprobs, dtype=torch.float)
                    for logprobs in self.rollout_logprobs
                ],
                max_batch_len=max_response_len,
                pad_token=pad_token,
            )
            batch["prev_logprobs"] = logprobs.cuda()

        return batch

    @staticmethod
    def merge_batches(
        batches: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Merge two batches into one."""
        merged_batch = {}
        if len(batches) == 0:
            return merged_batch
        if len(batches) == 1:
            return batches[0]
        for key in batches[0].keys():
            assert torch.is_tensor(batches[0][key]), (
                f"Expected tensor for key {key} in batches, got {type(batches[0][key])}"
            )
            assert torch.is_tensor(batches[0][key]), (
                f"Expected tensor for key {key} in batches, got {type(batches[0][key])}"
            )
            merged_batch[key] = torch.cat([batch[key] for batch in batches], dim=0)
        return merged_batch


class BatchResizingIterator:
    """The iterator for handling getting a batch and split it as a batch iterator with optional dynamic batch size."""

    def __init__(
        self,
        cfg: DictConfig,
        get_batch_fn: Callable,
        micro_batch_size: int,
        total_batch_size: int,
        num_global_batches: int,
        forward_only: bool,
        batch_tensor_key: str = "input_ids",
    ):
        """Initialize the BatchResizingIterator.

        Args:
            cfg (DictConfig): The configuration object.
            get_batch_fn (Callable): The function to get the batch.
            micro_batch_size (int): The size of the micro batch.
            global_batch_size_per_dp (int): The global batch size per data parallel. Here a global batch means the data required for running a single step of inference/training.
            batch_tensor_key (str): The key for retrieving a sample batch tensor, which will be used to measure the batch size and sequence length. By default, this is "input_ids", which means the input_ids tensor's shape will be used to determine batch size and sequence length.
        """
        self.cfg = cfg
        self.get_batch_fn = get_batch_fn
        self.micro_batch_size = micro_batch_size
        self.num_global_batches = num_global_batches
        self.total_batch_size = total_batch_size
        self.global_batch_size = total_batch_size // num_global_batches
        self.forward_only = forward_only
        self.batch_tensor_key = batch_tensor_key

        # Iterator states
        self.consumed_batch_size = 0
        self.micro_batch_iter = iter([])
        self.global_batch_iter = iter([])
        self.prefetch_micro_batch = None  # Used for computing batch info
        self.global_batch_done = False
        self.batches = []

    def check_finished_global_batch(self):
        assert self.global_batch_done, (
            f"Batch iterator has not finished for this global batch, only consumed {self.consumed_batch_size} sequences, expected {self.global_batch_size}"
        )

    def get_all_batches(self):
        """Retrieve all the batches (merged) iterated after the last call to get_all_batches."""
        batch = RolloutResult.merge_batches(self.batches)
        self.batches = []
        return batch

    def prefetch_one_batch(self):
        """Get the total sequence length, number of microbatches, and indices based on the batch information and dynamic batch sizing.

        Args:
            forward_micro_batch_size: The size of the forward micro batch.
            forward_only: Whether to only consider the forward pass.
        """
        if self.prefetch_micro_batch is None:
            self.prefetch_micro_batch = next(self)

        return self.prefetch_micro_batch

    def _get_next_micro_batch(self):
        """Retrieve the next micro batch from the current microbatch iterator."""
        if self.prefetch_micro_batch is not None:
            # If a microbatch has already been prefetched for batch info computation
            # Return the prefetched microbatch
            micro_batch = self.prefetch_micro_batch
            self.prefetch_micro_batch = None
        else:
            micro_batch: Dict[str, torch.Tensor] = next(self.micro_batch_iter)
            self.global_batch_done = False
            self.consumed_batch_size += micro_batch[self.batch_tensor_key].shape[0]
            self.batches.append(micro_batch)
            if self.consumed_batch_size == self.global_batch_size:
                # A global batch has been consumed, store the global batch step history
                self.consumed_batch_size = 0
                self.global_batch_done = True
            else:
                assert self.consumed_batch_size < self.global_batch_size, (
                    f"Recevied batches with a total size of {self.consumed_batch_size}, which exceeds the global batch size per dp {self.global_batch_size}. This suggests that the configured global batch size cannot be divided by the actual batch size."
                )
        return micro_batch

    def _get_global_batches(self):
        """Split a batch into multiple global batches, each of which will be used for one step of inference/training."""
        batch, result = self.get_batch_fn()
        batch_size = result.num_sequence
        if batch_size % self.global_batch_size != 0:
            # If the batch size is smaller than the global batch size per data parallel group,
            # we can return the batch as is
            return iter([batch])
        num_splits = batch_size // self.global_batch_size
        return get_iterator_k_split(
            batch,
            num_splits=num_splits,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

    def __iter__(self):
        """Return the iterator object itself."""
        return self

    def __next__(self):
        """Retrieve the next micro batch from the current microbatch iterator."""
        try:
            return self._get_next_micro_batch()
        except StopIteration:
            try:
                global_batch = next(self.global_batch_iter)
            except StopIteration:
                # If both the current micro and global batch iterators are exhausted, fetch a new batch
                self.global_batch_iter = self._get_global_batches()
                global_batch = next(self.global_batch_iter)

            global_batch_size = get_batch_size(global_batch, self.batch_tensor_key)
            self.micro_batch_iter = get_iterator_k_split(
                global_batch,
                num_splits=global_batch_size // self.micro_batch_size,
                shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
                shuffle_seed=self.cfg.actor.seed,
            )

            return self._get_next_micro_batch()
