MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"

OUTPUT_DIR = (
    "/content/drive/MyDrive/llms/checkpoints/general-instruction-lora/"
)



MAX_LENGTH = 512
MAX_STEPS = 1000
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0005
DROPOUT = 2e-4
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
ACCUM_STEPS = 8
EVAL_STRATEGY = "steps"
EVAL_STEPS = 250
LOGGING_STEPS = 10
ASSISTANT_ONLY_LOSS = True
R=8
LORA_ALPHA = 16
LORA_DROPOUT = DROPOUT
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]
BIAS = "none"
TASKTYPE="CAUSAL_LM"
