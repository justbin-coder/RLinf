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

from importlib.metadata import PackageNotFoundError, version

from packaging.version import parse


def get_version(pkg):
    try:
        return parse(version(pkg))
    except PackageNotFoundError:
        return None


package_name = "sglang"
package_version = get_version(package_name)

sglang_version = None

if package_version is None:
    raise ValueError(f"vllm version {package_version} not supported")
elif package_version >= parse("0.4.4") and package_version < parse("0.4.6.post2"):
    sglang_version = package_version
    from rlinf.hybrid_engines.sglang.sglang_0_4_4 import io_struct
    from rlinf.hybrid_engines.sglang.sglang_0_4_4.sgl_engine import (
        Engine,
    )
elif package_version >= parse("0.4.6.post2") and package_version < parse("0.4.8"):
    sglang_version = package_version
    from rlinf.hybrid_engines.sglang.sglang_0_4_6 import io_struct
    from rlinf.hybrid_engines.sglang.sglang_0_4_6.sgl_engine import (
        Engine,
    )
elif package_version >= parse("0.4.8") and package_version <= parse("0.4.9"):
    sglang_version = package_version
    from rlinf.hybrid_engines.sglang.sglang_0_4_9 import io_struct
    from rlinf.hybrid_engines.sglang.sglang_0_4_9.sgl_engine import (
        Engine,
    )
elif package_version >= parse("0.5.0") and package_version < parse("0.5.3"):
    sglang_version = package_version
    from rlinf.hybrid_engines.sglang.sglang_0_5_2 import io_struct
    from rlinf.hybrid_engines.sglang.sglang_0_5_2.sgl_engine import (
        Engine,
    )
else:
    raise ValueError(f"sglang version {package_version} not supported")

__all__ = ["Engine", "io_struct"]
