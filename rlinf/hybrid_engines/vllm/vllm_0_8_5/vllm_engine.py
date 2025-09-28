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
from functools import partial
from typing import List, Optional, Union

from omegaconf import DictConfig
from PIL.Image import Image
from vllm.config import VllmConfig
from vllm.inputs.data import TextPrompt, TokensPrompt
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
from vllm.v1.engine.llm_engine import LLMEngine as _LLMEngine

from rlinf.scheduler.manager.worker_manager import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement


class VLLMEngine:
    def __init__(
        self,
        vllm_config: VllmConfig,
        log_stats: bool,
        dp_rank: int,
        rlinf_config: DictConfig,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
        multiprocess_model: bool = False,
    ):
        # vllm_worker_cls = partial(VLLMWorker, rlinf_config=rlinf_config)
        vllm_worker_cls = "rlinf.hybrid_engines.vllm.vllm_0_8_5.worker.VLLMWorker"
        vllm_config.parallel_config.worker_cls = vllm_worker_cls

        from rlinf.hybrid_engines.vllm.vllm_0_8_5.executor import VLLMExecutor

        executor_factory = partial(
            VLLMExecutor,
            rlinf_config=rlinf_config,
            parent_address=parent_address,
            placement=placement,
            dp_rank=dp_rank,
        )

        self._engine = _LLMEngine(
            vllm_config=vllm_config,
            executor_class=executor_factory,
            log_stats=log_stats,
            multiprocess_mode=multiprocess_model,
        )
        self.request_counter = Counter()

    def generate(
        self,
        input_ids: Union[List[List[int]], List[int]],
        sampling_params: Union[SamplingParams, PoolingParams],
        image_data: Optional[List[List[Union[bytes, str]]]] = None,
        prompt_texts: Optional[Union[List[str], str]] = None,
        return_logprobs: bool = False,
    ) -> List[RequestOutput]:
        """
        Use the VLLM engine to generate text based on input token IDs or prompt text.

        Args:
            input_ids: A list of lists of input token IDs, or a single list of input
                token IDs.
            sampling_params: Sampling parameters for generation.
            prompt_text: Optional; A list of prompt strings or a single prompt string,
                if provided, it will be used instead of input_ids.
            return_logprobs: Whether to return log probabilities of the generated tokens.

        Returns:
            A list of RequestOutput objects containing the results of the generation.
        """
        sampling_params.logprobs = 0 if return_logprobs else None
        self._add_requests(
            input_ids=input_ids,
            prompt_texts=prompt_texts,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        results: List[RequestOutput] = self._run_engine()
        return results

    @staticmethod
    def _process_image_data(
        image_data: Optional[List[List[Union[bytes, str]]]],
    ) -> Optional[List[List[Image]]]:
        """
        Process the input image data which can be in various formats including file paths,
        URLs, byte streams, or PIL Image objects. Converts all images to PIL Image format.

        Args:
            image_data: A list of lists where each sublist contains images in various formats.

        Returns:
            A list of lists of PIL Image objects, or None if input is None.
        """
        if image_data is None:
            return None
        batch_image_list = []
        for imgs in image_data:
            if not isinstance(imgs, list):
                imgs = [imgs]
            image_list = []
            for img in imgs:
                if isinstance(img, str):
                    if img.startswith("http://") or img.startswith("https://"):
                        from io import BytesIO

                        import requests
                        from PIL import Image

                        response = requests.get(img)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        image_list.append(image)
                    else:
                        from PIL import Image

                        image = Image.open(img).convert("RGB")
                        image_list.append(image)
                elif isinstance(img, bytes):
                    from io import BytesIO

                    from PIL import Image

                    image = Image.open(BytesIO(img)).convert("RGB")
                    image_list.append(image)
                elif isinstance(img, Image.Image):
                    image_list.append(img)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
            batch_image_list.append(image_list)
        return batch_image_list

    def _add_requests(
        self,
        input_ids: Union[List[List[int]], List[int]],
        sampling_params: Union[SamplingParams, PoolingParams],
        image_data: Optional[List[List[Union[bytes, str]]]] = None,
        prompt_texts: Optional[Union[List[str], str]] = None,
    ) -> None:
        """
        Add generation requests to the engine.

        Args:
            input_ids: A list of lists of input token IDs, or a single list of input token IDs.
            prompt_texts: Optional; A list of prompt strings or a single prompt string, if provided,
                it will be used instead of input_ids.
            sampling_params: Optional; Sampling parameters for generation.
        """
        image_data = self._process_image_data(image_data)

        if prompt_texts is not None:
            # if not None, we use prompt_text rather than input_ids
            if isinstance(prompt_texts, str):
                prompt_texts = [prompt_texts]
            assert isinstance(prompt_texts, list), (
                f"Expected list for prompt_texts, got {type(prompt_texts)}"
            )
            for idx, prompt_text in enumerate(prompt_texts):
                request_id = str(next(self.request_counter))
                text_prompt = TextPrompt(
                    prompt=prompt_text,
                    multi_modal_data=image_data[idx]
                    if image_data is not None
                    else None,
                )
                self._engine.add_request(
                    request_id=request_id,
                    prompt=text_prompt,
                    params=sampling_params,
                )
            return

        assert isinstance(input_ids, list), (
            f"Expected list for input_ids, got {type(input_ids)}"
        )
        if not isinstance(input_ids[0], list):
            input_ids = [input_ids]

        for idx, input_id in enumerate(input_ids):
            request_id = str(next(self.request_counter))
            tokens_prompt = TokensPrompt(
                prompt_token_ids=input_id,
                multi_modal_data={"image": image_data[idx]}
                if image_data is not None
                else None,
            )
            self._engine.add_request(
                request_id=request_id,
                prompt=tokens_prompt,
                params=sampling_params,
            )

    def _run_engine(self) -> List[RequestOutput]:
        """
        Run the engine until all requests are finished.

        Returns:
            A list of RequestOutput objects containing the results of the generation.
        """
        outputs: List[RequestOutput] = []

        while self._engine.has_unfinished_requests():
            step_outputs = self._engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        return sorted(outputs, key=lambda x: int(x.request_id))

    def offload_model_weights(self) -> None:
        """
        Offload most graphic memory vllm used, including model's weights, buffers and kv cache.
        """
        self._engine.collective_rpc("offload_model_weights")

    def sync_hf_weight(self) -> None:
        """
        Sync model weights from actor to the vllm workers.
        """
        self._engine.collective_rpc("sync_hf_weight")
