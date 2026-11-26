# ============================================================
# BASE MODEL VS LORA MODEL GENERATION TEST
# ============================================================
from config import OUTPUT_DIR,OUTPUT_DIR_2,OUTPUT_DIR_3
import gc

import torch

from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"

ADAPTER_PATH = OUTPUT_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)


TEST_PROMPTS = [
    "Return exactly three Python learning tips. Use JSON with a key named tips.",

    "Explain gradient descent in exactly two sentences.",

    "Classify this review as positive or negative. Return only the label: "
    "\"The product broke after one day.\"",

    "Rewrite this sentence formally: "
    "\"hey can u send me that file asap\"",

    "List four benefits of regular exercise. "
    "Each item must contain exactly five words.",

    "Write a Python function named is_even with type hints. "
    "Return only the code.",

    "Summarize this in fewer than 20 words: "
    "\"Machine learning systems identify patterns in data "
    "and use those patterns to make predictions.\"",
]


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ============================================================
# BASE MODEL
# ============================================================

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=DTYPE,
    ).to(DEVICE)

    model.config.use_cache = True
    model.eval()

    return model


# ============================================================
# BASE MODEL WITH LORA ADAPTER
# ============================================================

def load_lora_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=DTYPE,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    ).to(DEVICE)

    model.config.use_cache = True
    model.eval()

    return model


# ============================================================
# GENERATION
# ============================================================

def generate_response(
    model,
    tokenizer,
    prompt,
):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)

    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        IM_END = tokenizer.convert_tokens_to_ids("<|im_end|>")

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            eos_token_id=[IM_END, tokenizer.eos_token_id],
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[
        0,
        prompt_length:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


# ============================================================
# MEMORY CLEANUP
# ============================================================

def release_model(model):
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print(f"Evaluation device: {DEVICE}")

    tokenizer = load_tokenizer()

    results = {
        prompt: {}
        for prompt in TEST_PROMPTS
    }

    # --------------------------------------------------------
    # Generate with original base model
    # --------------------------------------------------------

    print("\nLoading original base model...")

    base_model = load_base_model()

    for prompt in TEST_PROMPTS:
        results[prompt]["base"] = generate_response(
            model=base_model,
            tokenizer=tokenizer,
            prompt=prompt,
        )

    release_model(base_model)

    # --------------------------------------------------------
    # Generate with LoRA-adapted model
    # --------------------------------------------------------

    print("Loading base model with LoRA adapter...")

    lora_model = load_lora_model()

    for prompt in TEST_PROMPTS:
        results[prompt]["lora"] = generate_response(
            model=lora_model,
            tokenizer=tokenizer,
            prompt=prompt,
        )

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    for prompt, responses in results.items():
        print("=" * 90)

        print(f"PROMPT:\n{prompt}")

        print("\nBASE MODEL:\n")
        print(responses["base"])

        print("\nLORA-ADAPTED MODEL:\n")
        print(responses["lora"])

        print()

    release_model(lora_model)


if __name__ == "__main__":
    main()