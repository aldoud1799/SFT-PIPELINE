from config import EVAL_STEPS,EVAL_STRATEGY,WEIGHT_DECAY,EVAL_BATCH_SIZE,MAX_LENGTH,MAX_STEPS,OUTPUT_DIR,DATASET_NAME,MODEL_NAME,DROPOUT,TRAIN_BATCH_SIZE,LEARNING_RATE,ACCUM_STEPS,BIAS,LORA_ALPHA,LORA_DROPOUT,R,TARGET_MODULES,TASKTYPE
from dataset import load_train_dataset,load_test_dataset,load_val_dataset,prepare_dataset
from model import load_tokenizer,load_model
from peft_config import create_lora_config
from trainer import create_trainer,create_training_args
import os
os.environ["WANDB_PROJECT"] = "phase-3-post-training"

def main():
    #===============================================================================
    #LOAD THE TOKENIZER AND THE MODEL
    #===============================================================================
    model = load_model(MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    #===============================================================================
    #LOAD THE DATASET
    #===============================================================================
    train_dataset = load_train_dataset(DATASET_NAME)
    test_dataset = load_test_dataset(DATASET_NAME)
    #val_dataset = load_val_dataset(DATASET_NAME)
    small_train_dataset = train_dataset.select(range(5000))
    small_test_dataset = test_dataset.select(range(500))
    #small_val_dataset = val_dataset.select(range(100))
    small_train_dataset = prepare_dataset(small_train_dataset)
    small_test_dataset = prepare_dataset(small_test_dataset)
    #===============================================================================
    #CREATE SFT CONFIG AND SFT TRAINER
    #===============================================================================
    training_args = create_training_args(
        output_dir=OUTPUT_DIR,
        max_length=MAX_LENGTH,
        max_steps=MAX_STEPS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=ACCUM_STEPS,
        weight_decay=WEIGHT_DECAY,
        eval_steps=EVAL_STEPS,
        eval_strategy=EVAL_STRATEGY,
        report_to="wandb",
        logging_steps=1
    )

    lora_config = create_lora_config(
        r=R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias=BIAS,
        task_type=TASKTYPE
    )

    trainer = create_trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_test_dataset,
        processing_class=tokenizer,
        peft_config=lora_config
    )
    #===============================================================================
    #START TRAINING
    #===============================================================================
    training_result = trainer.train()
    print("TRAINING IS COMPLETE")
    print(training_result.metrics)
    #===============================================================================
    #SAVE THE MODEL AND THE TOKENIZER
    #===============================================================================
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"THE FINE TUNED MODEL AND THE TOKENIZER ARE SAVED TO {OUTPUT_DIR}")

if __name__ == "__main__":
    main()