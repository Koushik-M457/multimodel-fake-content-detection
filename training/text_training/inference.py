import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("checkpoints/text")
model.eval()

def caption_fake_probability(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)

    return probs[:, 1].item()

