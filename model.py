from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

def load_model(model):
    model = AutoModelForCausalLM.from_pretrained(model,dtype=torch.float16)
    model.gradient_checkpointing_enable()
    model.use_cache = False
    return model

def load_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer
