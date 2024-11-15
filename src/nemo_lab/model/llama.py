# Copyright Justin R. Goheen.
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

import os
from dataclasses import dataclass

from nemo.collections import llm
from nemo.collections.llm.gpt.model.llama import Llama3Config

ckpts = os.path.join(".models/pretained")


@dataclass
class Llama32Config1B(Llama3Config): ...


@dataclass
class Llama32Config3B(Llama3Config): ...


class Llama32Model1B(llm.LlamaModel): ...


class Llama32Model3B(llm.LlamaModel): ...
