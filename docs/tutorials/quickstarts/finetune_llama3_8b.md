# NeMo 2.0 Llama 3 Finetuning Quickstart

The steps shown below have been collected from the NeMo 2.0 documentation and tutorials, and help to familiarize with finetuning recipes made available in NeMo 2.0.

> [!IMPORTANT]
> A CUDA compatible operating system and device is required for this quickstart

> [!IMPORTANT]
> attempting to run the quickstart in a notebook may affect the Lightning Trainer and NeMo training strategy

# The Goal

We will create a Llama 3 variant finetuned on SQuAD (Stanford Q&A) via a finetuning recipe found in NeMo; and we will manage the experiment with NeMo Run.

# Intro

While there are only a few lines of code required to run the quickstart, we should acknowledge the fact that this means the NeMo framework engineers have done a lot of hard work for us, and have successfully abstracted many processes behind these high level interfaces. Namely data processing pipelines, model downloading and instantiation as modules, trainer instantiation, and recipe configs. 

As such, we should be prepared to troubleshoot errors that may lead us into the source code of NeMo, Megatron Core, PyTorch, PyTorch Lightning, and NVIDIA Apex. Additionally, this will mean sharing issues with the maintainers on GitHub, and helping to guide other community members by sharing common resolutions in community forums. 

# An Even Faster Quickstart

If you wish to run the quickstart, and then read the accompanying commentary that is provided below, we can run the following in terminal:

```bash
bash install_requirements.sh
python scripts/tutorials/nemo/llama3-finetuning-quickstart.py
```

> [!WARNING]
> installing the requirements takes several minutes <br>
> DO NOT INTERRUPT THE PROCESS!

> [!IMPORTANT]
> attempting to run the quickstart in a notebook may affect the Lightning Trainer and NeMo training strategy

# The Steps

## Install requirements

The following installation commands are provided in [`install_requirements.sh`](../../install_requirements.sh), and can be ran from the terminal with `bash install_requirements.sh`. Each command is shown here so that we might state why each is included as a requirement.

```bash
# torch
pip install "torch==2.2.1" --index-url https://download.pytorch.org/whl/cu121
# nvidia apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings \
"--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
git+https://github.com/NVIDIA/apex.git@24.04.01
# megatron-core
pip install "megatron-core==0.9.0"
# nemo-run
pip install git+https://github.com/NVIDIA/NeMo-Run.git@6b00501e68d56070bb4fb9808ab11bb8d0f03b51
# flash-attn
# see https://github.com/NVIDIA/TransformerEngine/issues/1236
pip install "flash-attn<2.5.7"
# nvidia transformer-engine
pip install "transformer-engine[pytorch]==1.11.0"
```

1. NVIDIA Apex extends PyTorch
2. Megatron Core is required by NeMo for MegatronStrategies
3. NeMo Run enables running NeMo recipes with convienent executors
4. Flash Attention and TransformerEngine support efficient execution primitives

> [!WARNING]
> the NVIDIA Apex build may take several minutes to complete the CUDA and C++ extension installations <br>
> DO NOT INTERRUPT THE PROCESS!

> [!TIP]
> run `bash install_requirements.sh` to run the above installation steps

## Imports

```python
from pathlib import Path

from nemo.collections import llm
import nemo_run as run
```

## Set the Config and Model

```python
config = llm.Llama31Config8B()
model = llm.LlamaModel(config=config)
```

## Import the Checkpoint

```python
llm.import_ckpt(
    model=model,
    source="hf://meta-llama/Llama-3.1-8B",
)
```

## Create the recipe

```python
recipe = llm.llama31_8b.finetune_recipe(
    name="llama31_8b_finetuning",
    dir="finetune-logs",
    num_nodes=1,
    num_gpus_per_node=1,
    peft_scheme="lora",  # 'lora', 'none'
    packed_sequence=False,
)
```

```python
recipe.trainer.strategy = "auto"  # let PTL do the work of choosing the training strategy
```

## Run the recipe

```python
run.run(recipe, executor=run.LocalExecutor())
```

# Conclusion

We just completed the basic steps of finetuning a model with NeMo recipes and NeMo Run. Future work might include evaluating any changes in quantitative or qualitative performance, and then deploying the model into a production pipeline with TensorRT-LLM and NVIDIA Inference Microservice (NIMs).