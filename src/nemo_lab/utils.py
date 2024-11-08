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
import tomllib
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_evals_from_config(config) -> dict:
    return {k: config[k] for k in config.evaluations if config.evaluations[k]}


def load_system_prompt(prompt_path: Union[str, Path]) -> str:
    with open(prompt_path, "rb") as f:
        system_prompt = tomllib.load(f)
    return system_prompt["system"]["prompt"]


def _join_persona(persona: dict) -> str:
    description = persona["base"]["description"].replace("\n", " ")
    rules = ["".join(["[RULE]", "\n", persona["rules"][k]["rule"].replace("\n", "")]) for k in persona["rules"].keys()]
    body = persona["body"]["text"].replace("\n", " ")
    return "\n\n".join([description, *rules, body])


def load_persona(persona_path: Union[str, Path]) -> str:
    with open(persona_path, "rb") as f:
        persona = tomllib.load(f)
    return _join_persona(persona)


def load_config(filepath: Union[Path, str]) -> DictConfig | ListConfig:
    cfgpath = os.path.join(filepath)
    return OmegaConf.load(cfgpath)
