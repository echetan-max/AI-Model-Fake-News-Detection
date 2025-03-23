import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

MODEL_PATH = "fake_news_model.pt"
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ Delete Old Model If It Exists (For Fresh Training)
if os.path.exists(MODEL_PATH):
    print("🗑️ Deleting old model to train a new one from scratch...")
    os.remove(MODEL_PATH)

# ✅ Load Dataset (LIAR Dataset)
def load_data():
    dataset = load_dataset("liar", trust_remote_code=True)

    # ✅ Fix label mapping (Fake News = 1, Real News = 0)
    def map_labels(example):
        example["labels"] = 1 if example["label"] > 2 else 0
        return example

    dataset = dataset.map(map_labels)

    # ✅ Balance dataset (Equal fake & real news samples)
    fake_news = [d for d in dataset["train"] if d["labels"] == 1]
    real_news = [d for d in dataset["train"] if d["labels"] == 0]
    min_size = min(len(fake_news), len(real_news))
    train_dataset = dataset["train"].select(range(min_size * 2))
    test_dataset = dataset["test"]

    # ✅ Tokenize data (Improved Preprocessing)
    def preprocess(example):
        return TOKENIZER(
            example["statement"],  
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    train_dataset = train_dataset.map(preprocess, batched=True)
    test_dataset = test_dataset.map(preprocess, batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, test_dataset

# ✅ Train Model (Always New Training)
def train_model():
    train_dataset, test_dataset = load_data()

    # ✅ Load BERT Model (Better Accuracy)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # ✅ Training Arguments (Fixed `eval_stratergy` → `evaluation_strategy`)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # ✅ Corrected spelling
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs"
    )

    # ✅ Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # ✅ Train Model
    trainer.train()

    # ✅ Save Trained Model
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model training complete! Saved as `fake_news_model.pt`")
    
    return model

# ✅ Always Train a New Model
print("🔄 Training a new model from scratch...")
model = train_model()
