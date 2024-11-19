import logging
from rich.logging import RichHandler

from pytorch_lightning.utilities.exceptions import MisconfigurationException

from nemo.collections import llm
import nemo_run as run

# create a logger
FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")

# configure the model
log.info("[green][FineTune] Configuring the model[/]")
config = llm.Llama31Config8B()
model = llm.LlamaModel(
    config=config,
)

# load the model
log.info("[green][FineTune] Loading the model[/]")
try:
    llm.import_ckpt(
        model=model,
        source="hf://meta-llama/Llama-3.1-8B",
    )
except MisconfigurationException as e:
    print(e)

# create the recipe
log.info("[green][FineTune] Instantiating recipe[/]")
recipe = llm.llama31_8b.finetune_recipe(
    name="llama31_8b_finetuning",
    dir="finetune-logs",
    num_nodes=1,
    num_gpus_per_node=1,
    peft_scheme="lora",  # 'lora', 'none'
    packed_sequence=False,
)
recipe.trainer.strategy = "auto"  # let PTL do the work

# run the recipe
log.info("[green][FineTune] Running recipe[/]")
run.run(recipe, executor=run.LocalExecutor())
