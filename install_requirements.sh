# apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install nemo_toolkit['all']
pip install git+https://github.com/NVIDIA/NeMo-Run.git
pip install git+https://github.com/NVIDIA/Megatron-LM.git
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir git+https://github.com/NVIDIA/apex.git