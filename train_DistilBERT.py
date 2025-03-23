import torch
import os
import numpy as np
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
import evaluate  #  Correct way to load metrics

#  Use DistilBERT for faster training
MODEL_PATH = "fake_news_model.pt"
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load & Preprocess Dataset (More Optimized)
def load_data():
    dataset = load_dataset("liar", trust_remote_code=True)

    # Convert labels (Fake = 1, Real = 0)
    def map_labels(example):
        example["labels"] = 1 if example["label"] > 2 else 0
        return example

    dataset = dataset.map(map_labels)

    #  Balance dataset (Equal Fake & Real News)
    min_size = min(sum(1 for _ in dataset["train"] if _["labels"] == 1),
                   sum(1 for _ in dataset["train"] if _["labels"] == 0))
    train_dataset = dataset["train"].select(range(min_size * 2))
    test_dataset = dataset["test"].select(range(min(len(dataset["test"]), 500)))  # ✅ Limit test size

    #  Tokenize with reduced sequence length (96 instead of 128)
    def preprocess(example):
        return TOKENIZER(example["statement"], padding="max_length", truncation=True, max_length=96)

    train_dataset = train_dataset.map(preprocess, batched=True)
    test_dataset = test_dataset.map(preprocess, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, test_dataset

#  Function to compute accuracy (Fixed)
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    return accuracy_metric.compute(predictions=predictions, references=labels)

#  Train Model (Faster & More Efficient)
def train_model():
    train_dataset, test_dataset = load_data()
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",  #  Save best model
        learning_rate=5e-5,  #  Increased for faster convergence
        per_device_train_batch_size=32,  # Increased batch size
        per_device_eval_batch_size=32,  
        num_train_epochs=1,  # Reduced epochs to speed up training
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True if torch.cuda.is_available() else False,  #  Mixed precision for speed
        load_best_model_at_end=True,  
        metric_for_best_model="accuracy",  
        gradient_accumulation_steps=2,  
        disable_tqdm=True  #  Reduce logging overhead
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,  
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  
    )

    trainer.train()

    torch.save(model.state_dict(), MODEL_PATH)
    print(" Model training complete! Saved as `fake_news_model.pt`")
    
    return model

print(" Training a faster and more accurate model...")
model = train_model()
