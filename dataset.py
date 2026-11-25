from datasets import load_dataset

def load_train_dataset(dataset,split="train_sft"):
    dataset = load_dataset(dataset,split=split)
    return dataset

def load_test_dataset(dataset,split="test_sft"):
    dataset = load_dataset(dataset,split=split)
    return dataset

def load_val_dataset(dataset,split="val_sft"):
    dataset = load_dataset(dataset,split=split)
    return dataset

def prepare_dataset(dataset):
    dataset = dataset.select_columns(["messages"])
    return dataset