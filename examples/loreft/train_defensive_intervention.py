import argparse
from typing import List
import torch
import transformers
import pandas as pd
import pyreft

PROMPT_TEMPLATE = """<s>[INST] %s [/INST]"""
DATA_DIR = 'dataset/advbench/harmful_behaviors.csv'

def main(
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
    layers: str = "18;28",
    low_rank_dimension: int = 2,
    n_train_examples: int = 10,
    batch_size: int = 10,
    learning_rate: float = 4e-3,
    num_train_epochs: float = 5.0,
    output_dir: str = "defense_results",
    logging_steps: int = 1,
    positions: str = "f1+l1",
    share_weights: bool = True,
    nonstop: bool = True
):
    print(
        f"model: {model_name_or_path}, "
        f"layers: {layers}, rank: {low_rank_dimension}, "
        f"position: {positions}, epoch: {num_train_epochs}, "
        f"num examples: {n_train_examples}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token

    # which layers to intervene on
    if layers != "all":
        layers = [int(l) for l in layers.split(";")]
    else:
        layers = [l for l in range(model.config.num_hidden_layers)]

    # get reft model
    reft_config = pyreft.ReftConfig(representations=[
        {
            "layer": layer,
            "component": "block_output",
            "low_rank_dimension": low_rank_dimension,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=low_rank_dimension
            )
        } for layer in layers
    ])
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(device)

    print('Number of interventions:', len(reft_config.representations))
    reft_model.print_trainable_parameters()

    train_df = pd.read_csv(DATA_DIR).iloc[:n_train_examples]
    prompts = [PROMPT_TEMPLATE % p for p in train_df["goal"].tolist()]
    completions = train_df["target"].tolist()

    num_interventions = len(reft_config.representations)

    data_module = pyreft.make_multiple_position_supervised_data_module(
        tokenizer, 
        model, 
        prompts, 
        completions,
        positions=positions, 
        share_weights=share_weights, 
        num_interventions=num_interventions, 
        nonstop=nonstop
    )

    training_args = transformers.TrainingArguments(
        num_train_epochs=num_train_epochs,
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        report_to="none"
    )

    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_model(f'{output_dir}/weights')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a defensive intervention on AdvBench.")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layers", type=str, default="18;28")
    parser.add_argument("--low_rank_dimension", type=int, default=2)
    parser.add_argument("--n_train_examples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=4e-3)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--output_dir", type=str, default="defense_results")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--positions", type=str, default="f1+l1")
    parser.add_argument("--share_weights", action="store_true")
    parser.add_argument("--nonstop", action="store_true")
    args = parser.parse_args()
    
    main(**vars(args))