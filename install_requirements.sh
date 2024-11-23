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