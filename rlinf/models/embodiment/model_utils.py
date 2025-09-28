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

from typing import Any, Optional

import torch
from transformers.generation import TopKLogitsWarper

from rlinf.utils.utils import compute_entropy_from_logits, compute_logprobs_from_logits


def default_logits_processor(logits, action_tokens, vocab_size, n_action_bins):
    logits = logits.permute(0, 2, 1)  # [B, vocab-size, action-dim]

    logits[:, : vocab_size - n_action_bins] = -torch.inf
    logits[:, vocab_size:] = -torch.inf

    logprobs = compute_logprobs_from_logits(logits=logits, target=action_tokens)

    entropy = compute_entropy_from_logits(logits)

    ret = {"logprobs": logprobs, "entropy": entropy}

    return ret


def custom_forward(
    model,
    input_ids,
    attention_mask,
    pixel_values,
    output_hidden_states=True,
    action_token_len=None,
    value_model=False,
    value_head_mode: str = "a",
    logits_processor=default_logits_processor,
    temperature: int = 1.0,
    top_k: int = -1,
    logits_processor_args: Optional[dict] = None,
):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        output_hidden_states=output_hidden_states,
    )
    logits = output.logits[:, -action_token_len - 1 : -1]  # [B, action_dim, vocab_size]

    processed_logits_tensor = logits / temperature
    top_k = min(top_k, processed_logits_tensor.size(-1))  # Safety check
    if top_k > 0:
        logits_warper = TopKLogitsWarper(
            top_k
        )  # since here is logprob instead of logits, we use 0 instead of -inf
        processed_logits_tensor = logits_warper(None, processed_logits_tensor)

    output_dict = logits_processor(processed_logits_tensor, **logits_processor_args)

    if value_model:
        # NOTE: Here we subtract 1 because the input tokens do not include the EOS token.
        last_hidden_state = output.hidden_states[-1]  # [B, L, hidden_dim]
        if value_head_mode == "a0":
            hidden_features = last_hidden_state[
                :, -action_token_len - 1
            ]  # [batch_size, hidden_dim]
            values = model.value_head(hidden_features)  # [batch_size, 1]
        else:
            raise ValueError(f"Unknown value head mode: {value_head_mode}")
    else:
        values = None

    if values is not None:
        output_dict.update({"values": values})

    return output_dict


def prepare_observations_for_vla(
    simulator_type: str,
    model_name: str,
    raw_obs: dict,
    use_proprio: bool,
    max_length: int,
    processor: Any,
    precision: torch.dtype,
    device: torch.device = torch.device("cuda:0"),
):
    task_descriptions = [
        f"In: What action should the robot take to {t.lower()}?\nOut: "
        for t in raw_obs["task_descriptions"]
    ]

    if simulator_type == "libero":
        image_tensor = torch.stack(
            [
                value.clone().to(device).permute(2, 0, 1)
                for value in raw_obs["images_and_states"]["full_image"]
            ]
        )
    elif simulator_type == "maniskill":
        images = raw_obs["images"]
        image_tensor = images.to(device=device, dtype=precision)
    elif simulator_type == "robotwin":
        images = raw_obs["images"]
        image_tensor = images.to(device=device, dtype=precision)
    else:
        raise NotImplementedError

    proprio_states = None
    if use_proprio:
        proprio_keys = [
            key for key in raw_obs["images_and_states"] if "image" not in key
        ]
        proprio_states = {
            key: torch.stack(
                [val.to(device) for val in raw_obs["images_and_states"][key]]
            )
            for key in proprio_keys
        }

    # Add num_images dimension
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.unsqueeze(1)
    assert image_tensor.ndim == 5

    if model_name == "openvla":
        processed_obs = processor(
            text=task_descriptions,
            images=image_tensor,
            padding="max_length",
            max_length=max_length,
        )
    elif model_name == "openvla_oft":
        images = {"images": image_tensor}
        processed_obs = processor(
            text=task_descriptions,
            images=images,
            proprio_states=proprio_states,
            padding="max_length",
            max_length=max_length,
        )

    processed_obs = processed_obs.to(device=device, dtype=precision)
    for key, value in processed_obs.items():
        processed_obs[key] = value.contiguous()

    return processed_obs


def prepare_observations(
    simulator_type: str,
    model_name: str,
    raw_obs: dict,
    use_proprio: bool,
    max_length: int,
    processor: Any,
    precision: torch.dtype,
    device: torch.device = torch.device("cuda:0"),
):
    if model_name == "openvla" or model_name == "openvla_oft":
        return prepare_observations_for_vla(
            simulator_type=simulator_type,
            model_name=model_name,
            raw_obs=raw_obs,
            use_proprio=use_proprio,
            max_length=max_length,
            processor=processor,
            precision=precision,
            device=device,
        )
    else:
        raise NotImplementedError
