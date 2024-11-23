if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    apt-get install -y libsndfile1 ffmpeg
fi
pip install Cython packaging
pip install "nemo_toolkit[all]"
pip install git+https://github.com/NVIDIA/NeMo-Run.git 
# apex python
# pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir git+https://github.com/NVIDIA/apex.git
# apex c++ extensions
pip install -v \
    --disable-pip-version-check \
    --no-cache-dir --no-build-isolation --config-settings \
    "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git
pip install -U numpy
# transformer engine
# see https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/intro.html#installation
git clone https://github.com/NVIDIA/TransformerEngine.git && \
cd TransformerEngine && \
git fetch origin 8c9abbb80dba196f086b8b602a7cf1bce0040a6a && \
git checkout FETCH_HEAD && \
git submodule init && git submodule update && \
NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .
cd ..
# megatron core
git clone https://github.com/NVIDIA/Megatron-LM.git && \
cd Megatron-LM && \
git checkout a5415fcfacef2a37416259bd38b7c4b673583675 && \
pip install .
cd ..
# tensorRT model optimizer
pip install nvidia-modelopt[torch]~=0.19.0 --extra-index-url https://pypi.nvidia.com