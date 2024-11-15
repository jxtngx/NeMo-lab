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
class Llama32Config1B(Llama3Config):
    attention_bias = False
    attention_dropout = 0.0
    bos_token_id = 128000
    eos_token_id = 128001
    head_dim = 64
    hidden_act = "silu"
    hidden_size = 2048
    initializer_range = 0.02
    intermediate_size = 8192
    max_position_embeddings = 131072
    mlp_bias = False
    model_type = "llama"
    num_attention_heads = 32
    num_hidden_layers = 16
    num_key_value_heads = 8
    pretraining_tp = 1
    rms_norm_eps = 1e-05
    rope_scaling = {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    }
    rope_theta = 500000.0
    tie_word_embeddings = True
    torch_dtype = "bfloat16"
    transformers_version = "4.45.0.dev0"
    use_cache = True
    vocab_size = 128256


@dataclass
class Llama32Config3B(Llama3Config):
    attention_bias = False
    attention_dropout = 0.0
    bos_token_id = 128000
    eos_token_id = 128001
    head_dim = 128
    hidden_act = "silu"
    hidden_size = 3072
    initializer_range = 0.02
    intermediate_size = 8192
    max_position_embeddings = 131072
    mlp_bias = False
    model_type = "llama"
    num_attention_heads = 24
    num_hidden_layers = 28
    num_key_value_heads = 8
    pretraining_tp = 1
    rms_norm_eps = 1e-05
    rope_scaling = {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    }
    rope_theta = 500000.0
    tie_word_embeddings = True
    torch_dtype = "bfloat16"
    transformers_version = "4.45.0.dev0"
    use_cache = True
    vocab_size = 128256


class Llama32Model1B(llm.LlamaModel): ...


class Llama32Model3B(llm.LlamaModel): ...
