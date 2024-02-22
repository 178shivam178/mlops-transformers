import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformer_code import dataset
from transformer_code.model import BERTBaseUncased
import numpy as np

def predict_sentiments(model_path, data_file):
    # Load data
    df = pd.read_csv(data_file).fillna("none")

    # Prepare dataset and data loader
    dataset = dataset.BERTDataset(review=df.review.values, target=None, is_test=True)
    data_loader = DataLoader(dataset, batch_size=8, num_workers=1)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTBaseUncased()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Perform inference
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data["review"]
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(outputs)

    # Convert predictions to sentiment labels
    sentiment_predictions = ['positive' if pred >= 0.5 else 'negative' for pred in predictions]

    return sentiment_predictions

# Example usage:
# model_path = "path_to_trained_model.pth"
# data_file = "path_to_data_file.csv"
# predictions = predict_sentiments(model_path, data_file)
# print(predictions)
