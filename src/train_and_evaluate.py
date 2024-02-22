from transformer_code  import config
from transformer_code import dataset
from transformer_code import engine
import torch
import pandas as pd
import numpy as np
import json
from get_data import read_params
from transformer_code.model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
import argparse
from transformers import get_linear_schedule_with_warmup


def run(config_path):
    config = read_params(config_path)
    dfx = pd.read_csv(config["ModelTraining"]["BERT"]["params"]["TRAINING_FILE"],nrows=50).fillna("none")
    dfx.sentiment = dfx.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=config["split_data"]["test_size"], 
        random_state=config["split_data"]["random_state"], stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        review=df_train.review.values, target=df_train.sentiment.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["ModelTraining"]["BERT"]["params"]["TRAIN_BATCH_SIZE"], num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values, target=df_valid.sentiment.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config["ModelTraining"]["BERT"]["params"]["VALID_BATCH_SIZE"], num_workers=1
    )

    device = torch.device(config["ModelTraining"]["BERT"]["params"]["DEVICE"])
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config["ModelTraining"]["BERT"]["params"]["TRAIN_BATCH_SIZE"] * config["ModelTraining"]["BERT"]["params"]["EPOCHS"])
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config["ModelTraining"]["BERT"]["params"]["EPOCHS"]):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        
        scores_file = config["reports"]["scores"]
        params_file = config["reports"]["params"]
        with open(scores_file, "w") as f:
            scores = {
                "Accuracy": accuracy,
                }
            json.dump(scores, f, indent=4)
        
        with open(params_file, "w") as f:
            params = {
                "TRAIN_BATCH_SIZE": config["ModelTraining"]["BERT"]["params"]["TRAIN_BATCH_SIZE"],
                "VALID_BATCH_SIZE": config["ModelTraining"]["BERT"]["params"]["TRAIN_BATCH_SIZE"],
                "EPOCHS":config["ModelTraining"]["BERT"]["params"]["EPOCHS"]
                }
            json.dump(params, f, indent=4)

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config["ModelTraining"]["BERT"]["params"]["MODEL_PATH"])
            best_accuracy = accuracy


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    run(config_path=parsed_args.config)