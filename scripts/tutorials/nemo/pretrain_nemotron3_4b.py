# code adapted from NeMo 2.0 documentation
# see https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html

import nemo_run as run

from nemo.collections import llm


def configure_recipe(
    nodes: int = 1,
    gpus_per_node: int = 1,
    max_steps: int = 100,
    val_check_interval: int = 10,
):
    recipe = llm.nemotron3_4b.pretrain_recipe(
        dir="./checkpoints/nemotron",  # Path to store checkpoints
        name="nemotron_pretraining",
        tensor_parallelism=1,
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        max_steps=max_steps,  # Setting a small value for the quickstart
    )
    # see https://pytorch-lightning.readthedocs.io/en/2.4.0/pytorch/common/trainer.html#
    recipe.trainer.val_check_interval = val_check_interval
    # recipe.trainer.strategy = "auto"  # try to override megatron strategy
    recipe.model.config.num_layers = 8
    return recipe


def local_executor_torchrun(nodes: int = 1, devices: int = 1) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "0",
        "CUDA_VISIBLE_DEVICES": "0",
    }

    executor = run.LocalExecutor(
        ntasks_per_node=devices,
        env_vars=env_vars,
        launcher="torchrun",
    )

    return executor


def run_pretraining():
    recipe = configure_recipe()
    executor = local_executor_torchrun(
        nodes=recipe.trainer.num_nodes,
        devices=recipe.trainer.devices,
    )

    run.run(recipe, executor=executor)


# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()
