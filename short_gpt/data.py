from tqdm import tqdm
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
import argparse
import shutil
import json
import os
import pandas as pd
import re

def get_preprocessed_samsum(tokenizer):
    dataset = load_dataset("samsum", split="train", trust_remote_code=True)

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)
        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

def create_peft_config(model, args):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, peft_config

def get_alpaca_small(tokenizer):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    return dataset


def get_medical_dataset(tokenizer):
    dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
    train_dataset = dataset.train_test_split(test_size=0.1)["train"] 
    eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

    def preprocess(train_dataset):
        df = pd.DataFrame(train_dataset[::])
        df = df[["Description", "Doctor"]].rename(columns={"Description": "question", "Doctor": "answer"})
        # Clean the question and answer columns
        df['question'] = df['question'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
        df['answer'] = df['answer'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
        # Assuming your DataFrame is named 'df' and the column is named 'df' and the column is named 'question'
        df['question'] = df['question'].str.lstrip('Q. ')
        df['answer'] = df['answer'].str.replace('-->', '')
        # build training dataset with the right format
        df['text'] = '[INST]@Enlighten. ' + df['question'] +'[/INST]'+ df['answer'] + ''
        # remove columns
        df=df.drop(['question','answer'],axis=1)
        dataset = ds.dataset(pa.Table.from_pandas(df).to_batches())
        dataset = Dataset(pa.Table.from_pandas(df))
        return dataset

    train_dataset = preprocess(train_dataset)
    eval_dataset = preprocess(eval_dataset)

    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )

        # "self-supervised learning" means the labels are also the inputs:
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt =f"""You are a AI Medical Assistant model. 
        Your job is to answer questions about a health. 
        You are given a question and context regarding the problem.

    You must answer the question.

    ### Input:
    {data_point["question"]}

    ### Context:
    {data_point["context"]}

    ### Response:
    {data_point["answer"]}
    """
        return tokenize(full_prompt)

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    return tokenized_train_dataset, tokenized_val_dataset
