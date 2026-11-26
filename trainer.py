from trl import SFTTrainer,SFTConfig

def create_training_args(
        output_dir,
        max_length,
        max_steps,
        train_batch_size,
        eval_batch_size,
        learning_rate,
        gradient_accumulation_steps,
        weight_decay,
        eval_steps,
        eval_strategy,
        report_to,
        logging_steps
):
    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_length=max_length,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        report_to=report_to,
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_cpu=False,
        bf16=False,
        fp16=True
    )
    return training_args

def create_trainer(
        model,
        args,
        train_dataset,
        eval_dataset,
        processing_class,
        peft_config
):
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=peft_config
    )
    return trainer