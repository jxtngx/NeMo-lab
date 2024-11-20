# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# code derived from https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/gpt/model/llama.py
# updated for Llama 3.2

from pathlib import Path
from dataclasses import dataclass


import torch
from transformers import LlamaForCausalLM
from nemo.collections import llm
from nemo.collections.llm.gpt.model.llama import LlamaModel, LlamaConfig, Llama3Config
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning import io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf


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


@io.model_importer(LlamaModel, "hf")
class HFLlamaImporter(io.ModelConnector["LlamaForCausalLM", LlamaModel]):
    def init(self) -> LlamaModel:
        return LlamaModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:

        source = LlamaForCausalLM.from_pretrained(str(self), torch_dtype="auto")
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Llama model to Nemo, model saved to {output_path} in {source.dtype}.")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv, _import_linear_fc1])

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> LlamaConfig:
        from transformers import LlamaConfig as HFLlamaConfig

        source = HFLlamaConfig.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        if getattr(source, "rope_scaling", None) is not None and source.rope_scaling.get("rope_type") == "llama3":
            # Apply Llama3.1 customize rope scaling
            cls = Llama32Config
        else:
            cls = LlamaConfig
        output = cls(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=False,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


def _import_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


@io.state_transform(
    source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
    target_key="decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0)
