# Copyright Justin R. Goheen.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from nemo.collections import llm
from nemo.collections.llm.gpt.model.llama import Llama3Config


@dataclass
class Llama32Config(Llama3Config):
    init_method_std = 0.02
    ffn_hidden_size = 8192
    layernorm_epsilon = 1e-05
    num_query_groups = 8
    rotary_base = 500000.0
    scale_factor: int = 32
    low_freq_factor: int = 1
    high_freq_factor: int = 4
    make_vocab_size_divisible_by = 128

    # #### conversion mapping from HFLlamaImporter #### #
    # num_layers=source.num_hidden_layers
    # hidden_size=source.hidden_size
    # ffn_hidden_size=source.intermediate_size
    # num_attention_heads=source.num_attention_heads
    # init_method_std=source.initializer_range
    # layernorm_epsilon=source.rms_norm_eps
    # num_query_groups=source.num_key_value_heads
    # rotary_base=source.rope_theta


@dataclass
class Llama32Config1B(Llama32Config):
    hidden_size = 2048
    num_attention_heads = 32
    num_layers = 16


@dataclass
class Llama32Config3B(Llama32Config):
    hidden_size = 3072
    num_attention_heads = 24
    num_layers = 28


class Llama32Model1B(llm.LlamaModel):
    def __init__(self, config=Llama32Config1B, optim=None, tokenizer=None, model_transform=None):
        super().__init__(config, optim, tokenizer, model_transform)


class Llama32Model3B(llm.LlamaModel):
    def __init__(self, config=Llama32Config3B, optim=None, tokenizer=None, model_transform=None):
        super().__init__(config, optim, tokenizer, model_transform)
