# NeMo 2.0 Finetuning Quickstart

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
python scripts/quickstart/nemo2_quickstart.py
```

> [!WARNING]
> installing the requirements takes several minutes
> DO NOT INTERRUPT THE PROCESS

> [!IMPORTANT]
> attempting to run the quickstart in a notebook may affect the Lightning Trainer and NeMo training strategy

# The Steps

## Install requirements

The following installation commands are provided in [`install_requirements.sh`](../../install_requirements.sh), and can be ran from the terminal with `bash install_requirements.sh`. Each command is shown here so that we might state why each is included as a requirement.

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install "nemo_toolkit[all]"
pip install git+https://github.com/NVIDIA/NeMo-Run.git
pip install git+https://github.com/NVIDIA/Megatron-LM.git
pip install -v \
    --disable-pip-version-check \
    --no-cache-dir --no-build-isolation --config-settings \
    "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git
```

1. `libsndfile1` and `ffmpeg` are needed for audio and vision files
2. `nemo_tool[all]` is the NeMo framework
3. `git+https://github.com/NVIDIA/NeMo-Run.git` installs NeMo Run
4. `git+https://github.com/NVIDIA/Megatron-LM.git` installs the nightly version of `megatron_core`
5. `git+https://github.com/NVIDIA/apex.git` is NVIDIA Apex and is needed for RoPE scaling and other methods

> [!WARNING]
> the NVIDIA Apex build may take several minutes to complete the CUDA and C++ extension installations
> DO NOT INTERRUPT THE PROCESS

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