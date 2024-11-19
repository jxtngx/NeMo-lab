from pytorch_lightning.utilities.exceptions import MisconfigurationException

from nemo.collections import llm
import nemo_run as run

# configure the model
config = llm.Llama31Config8B()
model = llm.LlamaModel(
    config=config,
)

try:
    llm.import_ckpt(
        model=model,
        source="hf://meta-llama/Llama-3.1-8B",
    )
except MisconfigurationException as e:
    print(e)


recipe = llm.llama31_8b.finetune_recipe(
    name="llama31_8b_finetuning",
    dir="finetune-logs",
    num_nodes=1,
    num_gpus_per_node=1,
    peft_scheme="lora",  # 'lora', 'none'
    packed_sequence=False,
)

recipe.trainer.strategy = "auto"  # let PTL do the work

run.run(recipe, executor=run.LocalExecutor())
