# apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install nemo_toolkit['all']
pip install transformer-engine['pytorch']
pip install git+https://github.com/NVIDIA/NeMo-Run.git
pip install git+https://github.com/NVIDIA/Megatron-LM.git
# apex python
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir git+https://github.com/NVIDIA/apex.git
# apex c++ extensions
# pip install -v \
#     --disable-pip-version-check \
#     --no-cache-dir --no-build-isolation --config-settings \
#     "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
#     git+https://github.com/NVIDIA/apex.git