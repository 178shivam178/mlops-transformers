import src.transformer_code.config
import torch
from src.transformer_code.model import BERTBaseUncased

DEVICE = "cpu"
MODEL_PATH = "saved_models/model.bin"
MODEL = BERTBaseUncased()
MODEL.load_state_dict(torch.load(src.transformer_code.config.MODEL_PATH,map_location=torch.device('cpu')))
MODEL.to(DEVICE)
MODEL.eval()

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)

class lessInfo(Exception):
    def __init__(self, message="Not having proper input"):
        self.message = message
        super().__init__(self.message)

def sentence_prediction(sentence):
    if len(str(sentence).split()) > 150:
        raise NotInRange("Input sentence should not exceed 150 words")
    
    if len(str(sentence).split()) < 2:
        raise lessInfo("Input sentence should be greater than 2 words")

    tokenizer = src.transformer_code.config.TOKENIZER
    max_len = src.transformer_code.config.MAX_LEN
    text = str(sentence)
    text = " ".join(text.split())

    inputs = tokenizer.encode_plus(
        text, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    sentiment_predictions = ['positive' if pred >= 0.5 else 'negative' for pred in outputs]
    return sentiment_predictions[0]
