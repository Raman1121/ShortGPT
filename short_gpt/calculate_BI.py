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
import torch.nn as nn
from short_hf import ShortHFModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import shutil
import json
import os
from data import get_preprocessed_samsum, create_peft_config, get_alpaca_small
from trl import SFTTrainer
from peft import PeftConfig, get_peft_model, prepare_model_for_kbit_training


################################################################################

def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def main(args):

    PARAMETER_BUDGET = 0

    MODEL_PATH_DICT = {
        "llama3": "/nfs/ukrc_roma_ait/models/huggingface/meta-llama/Meta-Llama-3-8B",
        "llama3_instruct": "/nfs/ukrc_roma_ait/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
        "gemma2b": "google/gemma-2-2b-it",
        "llama3.2_1b": "/home/rdutt/Llama-3.2-1B/",
        "llama3.2_3b": "/home/rdutt/Llama-3.2-3B/",
    }

    ### CONSTANTS
    ROOT_SAVEPATH = "/nfs/ukrc_roma_ait/models/"
    model_name = "Pruned_"+args.model+"_dataset_"+args.dataset.split("/")[-1]+"_layers_"+str(args.num_layers_to_prune)
    MODEL_SAVEPATH = os.path.join(ROOT_SAVEPATH, "Pruned_Models", model_name)
    original_model_path = MODEL_PATH_DICT[args.model]

    data = load_dataset(args.dataset, split="train")  # authors sample 10,000 texts to compute block influences

    dataloader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
    )

    MAX_SEQ_LEN = 1024
    short_model = ShortHFModel(
        model_name=MODEL_PATH_DICT[args.model],
        layers_path="model.layers",
        n_prune_layers=args.num_layers_to_prune,
    )

    # MAX_SEQ_LEN = short_model.model.config.max_position_embeddings

    original_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_DICT[args.model], device_map='cpu')    # Original model is on CPU because we need it only to calculate the numnber of pruned parameters
    tokenizer = short_model.tokenizer

    for i, batch in enumerate(tqdm(dataloader)):
        prompts = batch['text']

        short_model.eval_importance(
            prompts=prompts,
            max_seq_len=MAX_SEQ_LEN,
            stride=256,
            max_gen_len=0,
            angular=args.angular,
        )

    print("############ IMPORTANCES")
    print(short_model.importances)
    print("############ REMOVE LAYERS")
    layers_to_remove = short_model.remove_layers()
    print(layers_to_remove)

    len(original_model.model.layers), len(short_model.model.model.layers)

    # CALCULATING THE NUMBER OF PARAMETERS
    for _layer_idx in layers_to_remove:
        PARAMETER_BUDGET += count_parameters(original_model.model.layers[_layer_idx])

    print("##### PARAMETER BUDGET: ", PARAMETER_BUDGET)

    """
    Saving the pruned model
    """
    short_model.model.save_pretrained(MODEL_SAVEPATH)

    # Copying tokenizer and other configs from the original Model for Completion
    print("!!! Copying tokenizer and other configs from the original Model for Completion")
    
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


    if(args.do_model_healing):
        print("Performing model healing by finetuning on Alpaca dataset")
        print("Loading model {}".format(MODEL_SAVEPATH))
        
        short_model = ShortHFModel(
            model_name=MODEL_SAVEPATH,
            layers_path="model.layers",
            n_prune_layers=args.num_layers_to_prune,
        )

        tokenizer = short_model.tokenizer
        peft_model, lora_config = create_peft_config(short_model.model, args)

        output_dir = os.path.join(MODEL_SAVEPATH, "logs")

        config = {
            'lora_config': lora_config,
            'learning_rate': args.lr,
            'num_train_epochs': args.epochs,
            'gradient_checkpointing': False,
        }

        dataset = get_alpaca_small(tokenizer) # Alpaca Dataset

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            # logging strategies
            logging_strategy="steps",
            per_device_train_batch_size=args.healing_batch_size,
            gradient_accumulation_steps=args.grad_acc_steps,
            logging_steps=10,
            save_strategy="no",
            optim="adamw_torch",
            **{k:v for k,v in config.items() if k != 'lora_config'}
        )

        

        # Create Trainer instance
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset = dataset,
            dataset_text_field = "text",
            callbacks=[],
        )

        # Start training
        trainer.train()

        print("Saving the Healed Model")
        model_name = model_name + "_Healed"
        MODEL_SAVEPATH = os.path.join(ROOT_SAVEPATH, "Pruned_Models", model_name)
        peft_model.save_pretrained(MODEL_SAVEPATH)

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--healing_batch_size", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="arcee-ai/sec-data-mini")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--num_layers_to_prune", type=int, default=5)

    parser.add_argument("--do_model_healing", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--grad_acc_steps", type=int, default=8)
    parser.add_argument("--angular", action="store_true")

    args = parser.parse_args()

    main(args)



