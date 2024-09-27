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

def main(args):

    MODEL_PATH_DICT = {
        "llama3": "/nfs/ukrc_roma_ait/models/huggingface/meta-llama/Meta-Llama-3-8B",
        "llama3_instruct": "/nfs/ukrc_roma_ait/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
        "gemma2b": "google/gemma-2-2b-it",
        "llama3.2_1b": "/home/rdutt/Llama-3.2-1B/",
        "llama3.2_3b": "/home/rdutt/Llama-3.2-3B/",
    }

    data = load_dataset(args.dataset, split="train")  # authors sample 10,000 texts to compute block influences

    dataloader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
    )

    MAX_SEQ_LEN = 1024
    short_model = ShortHFModel(
        model_name=MODEL_PATH_DICT[args.model],
        # model_name="google/gemma-2-2b-it",
        layers_path="model.layers",
        n_prune_layers=args.num_layers_to_prune,
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

    print("Saving Pruned model!!!")
    
    ROOT_SAVEPATH = "/nfs/ukrc_roma_ait/models/"
    model_name = "Pruned_"+args.model+"_dataset_"+args.dataset.split("/")[-1]+"_layers_"+str(args.num_layers_to_prune)
    MODEL_SAVEPATH = os.path.join(ROOT_SAVEPATH, "Pruned_Models", model_name)
    short_model.model.save_pretrained(MODEL_SAVEPATH)

    # Copying tokenizer and other configs from the original Model for Completion
    print("!!! Copying tokenizer and other configs from the original Model for Completion")
    original_model_path = MODEL_PATH_DICT[args.model]
    shutil.copyfile(original_model_path+"/tokenizer.json", os.path.join(MODEL_SAVEPATH, "tokenizer.json"))
    shutil.copyfile(original_model_path+"/tokenizer_config.json", os.path.join(MODEL_SAVEPATH, "tokenizer_config.json"))
    shutil.copyfile(original_model_path+"/special_tokens_map.json", os.path.join(MODEL_SAVEPATH, "special_tokens_map.json"))
    
    print("!!! Editing the new config file to reflect the number of layers")
    # Read json file
    data = json.load(open(MODEL_PATH_DICT[args.model] + "/config.json", "r"))
    data['num_hidden_layers'] = data['num_hidden_layers'] - args.num_layers_to_prune

    print("!!! New Number of Layers: ", data['num_hidden_layers'])

    # Save the new config file
    print("!!! Saving the new config file")
    json.dump(data, open(os.path.join(MODEL_SAVEPATH, "config.json"), "w"))
    print("!!! Finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="arcee-ai/sec-data-mini")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--num_layers_to_prune", type=int, default=5)
    args = parser.parse_args()

    main(args)



