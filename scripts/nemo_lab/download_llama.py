import os
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

# simple resolve to set cache dir
filepath = Path(__file__)
rootpath = filepath.parents[2]
cache_dir = os.path.join(rootpath, ".lab-models", "hugging-face")

# before download, .from_pretrained will check the cache dir
# for existing downloads and avoid overwriting an existing model

# #### Llama 3.2 1B #### #
# download tokenizer
AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)
# download model
AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)

# #### Llama 3.2 3B #### #
# download tokenizer
AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", cache_dir=cache_dir)
# download model
AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", cache_dir=cache_dir)
