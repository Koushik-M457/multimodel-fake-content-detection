import torch
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from dataset import CaptionDataset

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

train_data = CaptionDataset("data/text/train.csv")
val_data = CaptionDataset("data/text/val.csv")

training_args = TrainingArguments(
    output_dir="checkpoints/text",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="logs/text",
    save_total_limit=2,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data
)

trainer.train()

