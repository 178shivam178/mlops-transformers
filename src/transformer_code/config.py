import transformers

MAX_LEN = 64
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "saved_models/model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)