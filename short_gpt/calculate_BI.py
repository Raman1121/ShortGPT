from tqdm.notebook import tqdm

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from transformers import default_data_collator, Trainer, TrainingArguments

from short_hf import ShortHFModel
from transformers import AutoModelForCausalLM


data = load_dataset("arcee-ai/sec-data-mini", split="train")  # authors sample 10,000 texts to compute block influences

dataloader = DataLoader(
    data,
    batch_size=1,
    shuffle=True,
)

MAX_SEQ_LEN = 1024
short_model = ShortHFModel(
    model_name="/nfs/ukrc_roma_ait/models/huggingface/meta-llama/Meta-Llama-3-8B",
    layers_path="model.layers",
    n_prune_layers=9
)

for i, batch in enumerate(tqdm(dataloader)):
    prompts = batch['text']

    short_model.eval_importance(
        prompts=prompts,
        max_seq_len=MAX_SEQ_LEN,
        stride=256,
        max_gen_len=0
    )

print("############ IMPORTANCES")
print(short_model.importances)
print("############ REMOVE LAYERS")
print(short_model.remove_layers())
