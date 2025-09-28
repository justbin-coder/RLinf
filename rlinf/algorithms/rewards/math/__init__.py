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

from typing import List

from omegaconf import DictConfig

from toolkits.math_verifier.verify import math_verify_call


class MathReward:
    def __init__(self, config: DictConfig):
        self.scale = config.get("scale", 1.0)

    def get_reward(
        self, response: List[str], reference: List[List[str]]
    ) -> List[float]:
        """
        Calculates reward scores for a list of responses compared to corresponding lists of reference answers.
        For each response, the function checks if it matches any of the provided references using the `process_results` function.
        The reward for each response is computed as the first element of the result (converted to float) multiplied by `self.scale`.
        Args:
            response (List[str]): A list of response strings to be evaluated.
            reference (List[List[str]]): A list where each element is a list of reference strings corresponding to each response.
        Returns:
            List[float]: A list of reward scores, one for each response.
        """

        return math_verify_call(response, reference) * self.scale
