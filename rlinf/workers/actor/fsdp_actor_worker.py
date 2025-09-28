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

import gc
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.multiprocessing.reductions import reduce_tensor
from tqdm import tqdm

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import actor_loss, calculate_adv_and_returns
from rlinf.algorithms.rewards import get_reward_class
from rlinf.algorithms.utils import (
    kl_penalty,
    preprocess_advantages_inputs,
    preprocess_loss_inputs,
)
from rlinf.data.io_struct import RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.models.embodiment.model_utils import custom_forward
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    masked_mean,
    seq_mean_token_mean,
    seq_mean_token_sum,
)
from rlinf.workers.rollout.utils import RankMapper


class FSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)
        super().__init__(cfg.actor)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        world_size = self._world_size
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_data_io_rank = True

        if self.cfg.algorithm.loss_agg_func == "token-mean":
            self.loss_agg_func = masked_mean
        elif self.cfg.algorithm.loss_agg_func == "seq-mean-token-sum":
            self.loss_agg_func = seq_mean_token_sum
        elif self.cfg.algorithm.loss_agg_func == "seq-mean-token-mean":
            self.loss_agg_func = seq_mean_token_mean
        else:
            raise NotImplementedError(
                f"algorithm.loss_agg_func={self.cfg.algorithm.loss_agg_func} is not supported!"
            )

        # Reward configurations
        if not self.cfg.reward.use_reward_model:
            assert self.cfg.reward.reward_type in ["math", "vqa"], (
                "only support math and vqa reward!"
            )
            reward_cls = get_reward_class(self.cfg.reward.reward_type)
            self.reward = reward_cls(self.cfg.reward)

    def init_worker(self):
        self.setup_model_and_optimizer()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self):
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self):
        if hasattr(self, "rollou_state_dict"):
            del self.rollou_state_dict

    def sync_model_to_rollout(self):
        if next(self.model.parameters()).is_cpu:
            self.load_fsdp_param_and_grad(self.device)

        self.rollou_state_dict = self.get_model_state_dict()

        if self._weight_dst_rank_in_rollout is not None:

            def transform_key(k):
                if k.startswith("model.language_model."):
                    return "model." + k[21:]
                elif k.startswith("model."):
                    return k[6:]
                else:
                    return k

            handle = {
                transform_key(k): reduce_tensor(v)
                for k, v in self.rollou_state_dict.items()
            }

            self.send(
                handle, self._rollout_group_name, self._weight_dst_rank_in_rollout
            )
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def compute_logprobs(self):
        self.model.eval()
        self.rollout_batch["logprob"] = self.rollout_batch["prev_logprobs"]

    def get_batch(
        self, channel: Channel
    ) -> Tuple[Dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def put_result(self, result: RolloutResult, channel: Channel):
        if channel.is_local:
            # Local channel, every process will put its own data locally
            # No need to broadcast
            channel.put(result)
        else:
            if self.is_data_io_rank:
                channel.put(result)

    def _load_weight_and_optimizer(self, channel: Channel):
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with channel.device_lock:
            if self.cfg.actor.get("enable_offload", False):
                self.load_fsdp_param_and_grad(self.device)
                self.load_fsdp_optimizer(self.device)

    def run_training(self, input_channel: Channel):
        # Get all batches for this DP
        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        batch = RolloutResult.merge_batches(batches)

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer(input_channel)

        global_batches = get_iterator_k_split(
            batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for global_batch in global_batches:
                train_global_batch_size = global_batch["input_ids"].shape[0]

                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size=}"
                )

                self.gradient_accumulation = (
                    train_global_batch_size // self.cfg.actor.micro_batch_size
                )
                # split batch into micro_batches
                train_micro_batches = get_iterator_k_split(
                    global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                metrics = {}
                for _, m_batch in enumerate(train_micro_batches):
                    for k, v in m_batch.items():
                        m_batch[k] = v.cuda() if isinstance(v, torch.Tensor) else v

                    multi_modal_inputs = {}
                    if "multi_modal_inputs" in m_batch.keys():
                        for key in m_batch["multi_modal_inputs"][0].keys():
                            multi_modal_inputs[key] = torch.cat(
                                [
                                    inputs[key]
                                    for inputs in m_batch["multi_modal_inputs"]
                                ],
                                dim=0,
                            ).cuda()

                    input_ids = m_batch["input_ids"]
                    attention_mask = m_batch["attention_mask"]
                    position_ids = m_batch["position_ids"]
                    prev_logprobs = m_batch["prev_logprobs"]
                    advantages = m_batch["advantages"]
                    ref_logprobs = None
                    if "ref_logprobs" in m_batch:
                        ref_logprobs = m_batch["ref_logprobs"]

                    loss_mask = m_batch["attention_mask"][:, -self.response_len :]

                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                    )

                    logits = output.logits

                    logits.div_(self.cfg.algorithm.sampling_params.temperature)

                    responses = input_ids[:, -self.response_len :]
                    logits = logits[
                        :, -self.response_len - 1 : -1, :
                    ]  # (bsz, response_length, vocab_size)
                    logprobs = compute_logprobs_from_logits(
                        logits, responses, task_type=self.cfg.runner.task_type
                    )
                    if self.calculate_entropy:
                        entropy = compute_entropy_from_logits(
                            logits, task_type=self.cfg.runner.task_type
                        )  # (bsz, response_length)

                    clip_ratio = self.cfg.algorithm.ratio_clip_eps
                    clip_ratio_low = (
                        self.cfg.algorithm.clip_ratio_low
                        if self.cfg.algorithm.clip_ratio_low is not None
                        else clip_ratio
                    )
                    clip_ratio_high = (
                        self.cfg.algorithm.clip_ratio_high
                        if self.cfg.algorithm.clip_ratio_high is not None
                        else clip_ratio
                    )
                    clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

                    loss, mbs_metrics_data = actor_loss(
                        loss_type=self.cfg.algorithm.loss_type,
                        loss_agg_func=self.loss_agg_func,
                        logprobs=logprobs,
                        old_logprobs=prev_logprobs,
                        advantages=advantages,
                        clip_ratio_low=clip_ratio_low,
                        clip_ratio_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_mask=loss_mask,
                    )

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if self.calculate_entropy:
                        entropy = output["entropy"][
                            :, -self.response_len - 1 : -1
                        ].contiguous()
                        entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                        if self.calculate_entropy_loss:
                            loss = (
                                loss - self.cfg.algorithm.entropy_bonus * entropy_loss
                            )

                    kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if self.kl_beta > 0 and ref_logprobs is not None:
                        kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                        kl_loss = self.loss_agg_func(kld, loss_mask)
                        loss = loss + kl_loss * self.kl_beta

                    # add to log
                    mbs_metrics_data.update(
                        {
                            "final_loss": loss.detach().cpu(),
                            "entropy_loss": entropy_loss.detach().cpu(),
                            "kl_loss": kl_loss.detach().cpu(),
                        }
                    )

                    append_to_dict(metrics, mbs_metrics_data)

                mean_metric_dict = {
                    key: np.mean(value) for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )
                training_metrics_list.append(mean_metric_dict)

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len, self._world_size
        )

        return rollout_metrics, training_metrics_list

    def save_checkpoint(self, save_base_path, step):
        torch.distributed.barrier()
        model_state = self.get_model_state_dict()
        optim_state = self.get_optimizer_state_dict()
        if self._rank == 0:
            os.makedirs(save_base_path, exist_ok=True)
            torch.save(model_state, os.path.join(save_base_path, "model.pt"))
            torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))
        torch.distributed.barrier()

    def _compute_batch_rewards(
        self, batch: Dict[str, torch.Tensor], answers: List[str]
    ):
        """Reward computation using non-model based reward."""
        texts = []
        for response, response_len in zip(
            batch["input_ids"],
            batch["response_lengths"],
        ):
            response = response[
                self.cfg.data.max_prompt_length : self.cfg.data.max_prompt_length
                + response_len
            ]
            texts.append(
                self.tokenizer.decode(response.tolist(), skip_special_tokens=True)
            )
        reward_scores = self.reward.get_reward(texts, answers)

        all_reward_scores = torch.as_tensor(
            reward_scores,
            dtype=torch.float,
            device=torch.device("cpu"),
        ).view(-1, 1)
        return all_reward_scores.flatten()

    # Rewards
    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence

            # Compute rule-based reward
            with self.worker_timer():
                if rollout_result.rewards is None:
                    rollout_result.rewards = self._compute_batch_rewards(
                        batch, rollout_result.answers
                    )

            self.put_result(rollout_result, output_channel)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    # Advantages and returns
    def compute_advantages_and_returns(
        self, input_channel: Channel, output_channel: Channel
    ):
        """Compute the advantages and returns.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence

            with self.worker_timer():
                if rollout_result.advantages is None:
                    mask = batch["attention_mask"][:, -self.response_len :]
                    advantages, returns = calculate_adv_and_returns(
                        adv_type=self.cfg.algorithm.adv_type,
                        reward_scores=batch["rewards"].cuda(),
                        mask=mask.cuda(),
                        num_responses=self.cfg.algorithm.group_size,
                    )
                    rollout_result.advantages = advantages.cpu()

            self.put_result(rollout_result, output_channel)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        world_size = self._world_size
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self._weight_dst_rank_in_rollout = self._rank
        if self._weight_dst_rank_in_rollout >= self._component_placement.get_world_size(
            "rollout"
        ):
            self._weight_dst_rank_in_rollout = None

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.channel = self.connect_channel(cfg.actor.channel.name)
        self.channel.create_queue(
            cfg.actor.channel.queue_name, maxsize=cfg.actor.channel.queue_size
        )

    def init_worker(self):
        self.setup_model_and_optimizer()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.checkpoint_load_path, self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def sync_model_to_rollout(self):
        if next(self.model.parameters()).is_cpu:
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        state_dict = self.get_model_state_dict()
        if self._weight_dst_rank_in_rollout is not None:
            self.send(
                state_dict, self._rollout_group_name, self._weight_dst_rank_in_rollout
            )
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

    async def recv_rollout_batch(self):
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for i in range(split_num):
            recv_list.append(
                await self.channel.get(
                    queue_name=self._replay_buffer_name, async_op=True
                ).async_wait()
            )

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        for key in recv_list[0].keys():
            if "env_info/" not in key:
                self.rollout_batch[key] = torch.cat(
                    [recv_list[i][key] for i in range(split_num)], dim=1
                )
            else:
                self.rollout_batch[key] = torch.cat(
                    [recv_list[i][key] for i in range(split_num)], dim=0
                )

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(self, rollout_batch):
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        for key, value in rollout_batch.items():
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            rollout_batch[key] = new_value

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if self.rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if self.rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & self.rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_logprobs(self):
        self.model.eval()
        self.rollout_batch["logprob"] = self.rollout_batch["prev_logprobs"]

    def compute_advantages_and_returns(self):
        stage_num = self.cfg.rollout.pipeline_stage_num
        env_world_size = self._component_placement.get_world_size("env")
        actor_world_size = self._component_placement.get_world_size("actor")
        num_group_envs_for_train = (
            self.cfg.algorithm.num_group_envs
            * stage_num
            * env_world_size
            // actor_world_size
        )

        kwargs = {
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "normalize_advantages": self.cfg.algorithm.get(
                "normalize_advantages", True
            ),
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "num_group_envs": num_group_envs_for_train,
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "rollout_epoch": self.cfg.algorithm.get("rollout_epoch", 1),
        }
        kwargs = preprocess_advantages_inputs(**kwargs)
        advantages, returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update({"advantages": advantages, "returns": returns})
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def run_training(self):
        if self.cfg.actor.get("enable_offload", False):
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        rollout_size = (
            self.rollout_batch["input_ids"].shape[0]
            * self.rollout_batch["input_ids"].shape[1]
        )
        shuffle_id = torch.randperm(rollout_size)

        for key, value in self.rollout_batch.items():
            self.log_on_first_rank(f"run training, {key}: {value.shape}")

        with torch.no_grad():
            for key, value in self.rollout_batch.items():
                if key in ["dones", "prev_values"]:
                    value = value[:-1]
                if "env_info" in key:
                    continue
                value = value.reshape(rollout_size, *value.shape[2:])
                self.rollout_batch[key] = value[shuffle_id]

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["input_ids"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        rollout_dataloader_iter = get_iterator_k_split(
            self.rollout_batch,
            rollout_size // batch_size_per_rank,
        )

        metrics = {}
        for _, train_global_batch in tqdm(
            enumerate(rollout_dataloader_iter), desc="get loss and metrics"
        ):
            # split batch into micro_batches
            train_global_batch_size = train_global_batch["input_ids"].shape[0]
            assert (
                train_global_batch_size
                == self.cfg.actor.global_batch_size
                // torch.distributed.get_world_size()
            )
            assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
            )
            train_micro_batch = get_iterator_k_split(
                train_global_batch,
                train_global_batch_size // self.cfg.actor.micro_batch_size,
            )

            self.optimizer.zero_grad()
            for data_idx, data in enumerate(train_micro_batch):
                for k, v in data.items():
                    data[k] = v.to(f"cuda:{int(os.environ['LOCAL_RANK'])}")

                data = self.model.preprocess_for_train(data)
                input_ids = data["input_ids"]
                action_tokens = data["action_tokens"]
                attention_mask = data["attention_mask"]
                pixel_values = data["pixel_values"]

                action_token_len = self.model.action_dim * self.model.num_action_chunks

                logits_processor_args = {
                    "action_tokens": action_tokens,
                    "vocab_size": self.model.vocab_size,
                    "n_action_bins": self.model.config.n_action_bins,
                }

                output_dict = custom_forward(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    action_token_len=action_token_len,
                    value_model=True
                    if self.cfg.algorithm.adv_type == "embodied_gae"
                    else False,
                    value_head_mode=self.cfg.actor.model.get("vh_mode", None),
                    temperature=self.cfg.algorithm.sampling_params.temperature_train,
                    top_k=self.cfg.algorithm.sampling_params.top_k,
                    logits_processor_args=logits_processor_args,
                )

                kwargs = {
                    "loss_type": self.cfg.algorithm.loss_type,
                    "logprob_type": self.cfg.algorithm.logprob_type,
                    "entropy_type": self.cfg.algorithm.entropy_type,
                    "single_action_dim": self.model.action_dim,
                    "logprobs": output_dict["logprobs"],
                    "entropy": output_dict["entropy"],
                    "values": output_dict.get("values", None),
                    "old_logprobs": data["prev_logprobs"],
                    "advantages": data["advantages"],
                    "returns": data["returns"],
                    "prev_values": data["prev_values"],
                    "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                    "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                    "value_clip": self.cfg.algorithm.get("value_clip", None),
                    "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                    "entropy_bonus": self.cfg.algorithm.entropy_bonus,
                    "loss_mask": data.get("loss_mask", None),
                    "loss_mask_sum": data.get("loss_mask_sum", None),
                    "max_episode_steps": self.cfg.env.train.max_episode_steps,
                }

                kwargs = preprocess_loss_inputs(**kwargs)

                loss, metrics_data = actor_loss(**kwargs)

                loss /= self.gradient_accumulation
                loss.backward()

                metrics_data["loss"] = loss.detach().item()
                append_to_dict(metrics, metrics_data)

            torch.cuda.empty_cache()

            grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()

            self.optimizer.zero_grad()
            data = {
                "actor/grad_norm": grad_norm.detach().item(),
                "actor/lr": self.optimizer.param_groups[0]["lr"],
            }
            if self.cfg.algorithm.adv_type == "embodied_gae":
                data["critic/lr"] = self.optimizer.param_groups[1]["lr"]
            append_to_dict(metrics, data)

        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

        return mean_metric_dict

    def save_checkpoint(self, save_base_path, step):
        torch.distributed.barrier()
        model_state = self.get_model_state_dict()
        optim_state = self.get_optimizer_state_dict()
        if self._rank == 0:
            os.makedirs(save_base_path, exist_ok=True)
            torch.save(model_state, os.path.join(save_base_path, "model.pt"))
            torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))
        torch.distributed.barrier()
