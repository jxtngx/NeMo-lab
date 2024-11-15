# Load model directly
import os

from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = os.path.join(".models/pretrained")

# download tokenizer
AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)
# download model
AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir=cache_dir)
