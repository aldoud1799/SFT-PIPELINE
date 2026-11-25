from peft import LoraConfig,TaskType

def create_lora_config(
        r,
        lora_alpha,
        lora_dropout,
        target_modules,
        bias,
        task_type
):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type
    )
    return lora_config