import torch
from config import MODEL_NAME
from model import load_model

model = load_model(MODEL_NAME)

linear_names = sorted({
    name.split(".")[-1]
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
})

print(linear_names)

for name in linear_names:
    print(name)