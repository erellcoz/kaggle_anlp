import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(filepath):
    df = pd.read_csv(filepath)
    texts = df['Text'].tolist()
    labels = df['Label'].tolist()
    unique_labels = list(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_to_id[label] for label in labels]
    return texts, labels, label_to_id

def main():
    # Load data
    filepath = "data/train_submission.csv"
    texts, labels, label_to_id = load_data(filepath)

    # Split data into training and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Initialize tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=len(label_to_id))

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len=128)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,  # Enable mixed precision training
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train model
    trainer.train()

if __name__ == "__main__":
    main()