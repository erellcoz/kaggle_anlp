import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import json

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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
        item = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_data(filepath):
    df = pd.read_csv(filepath)
    texts = df['Text'].tolist()
    labels = df['Label'].tolist() if 'Label' in df.columns else None
    unique_labels = list(set(labels)) if labels is not None else []
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)} if labels is not None else {}
    if labels is not None:
        labels = [label_to_id[label] for label in labels]
    return texts, labels, label_to_id

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1_weighted, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted', zero_division=0)
    precision_unweighted, recall_unweighted , f1_unweighted, _ = precision_recall_fscore_support(p.label_ids, preds, average='macro', zero_division=0)
    accuracy = accuracy_score(p.label_ids, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_weighted": f1_weighted,
        "precision_unweighted": precision_unweighted,
        "recall_unweighted": recall_unweighted,
        "f1_unweighted": f1_unweighted,
    }

def train_model():
    # Load data
    filepath = "data/train_submission.csv"
    texts, labels, label_to_id = load_data(filepath)

    # Save label_to_id mapping
    with open('results/label_to_id.json', 'w') as f:
        json.dump(label_to_id, f)

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
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        learning_rate=5e-5,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Save the best model
    model_path = "results/best_model.pth"
    torch.save(model.state_dict(), model_path)

def load_model(model_path, num_labels):
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=num_labels)
    model.load_state_dict(torch.load(model_path))
    return tokenizer, model

def predict(model, tokenizer, texts, max_len=128, batch_size=16, device='cpu'):
    dataset = TextDataset(texts, tokenizer=tokenizer, max_len=max_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions

def inference():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_filepath = "data/train_submission.csv"
    test_df = pd.read_csv(test_filepath)
    test_texts = test_df['Text'].tolist()
    
    # Load data
    filepath = "data/train_submission.csv"
    df = pd.read_csv(filepath)
    texts = df['Text'].tolist()
    labels = df['Label'].tolist()
    ids = df['ID'].tolist()
    # Split data into training and test sets
    train_texts, test_texts, train_labels, test_labels,train_ids,test_ids = train_test_split(texts, labels,ids, test_size=0.2, random_state=42)

    # Load model and tokenizer
    model_path = './results/best_model.pth'
    with open('results/label_to_id.json', 'r') as f:
        label_to_id = json.load(f)
    id_to_label = {v: k for k, v in label_to_id.items()}
    tokenizer, model = load_model(model_path, num_labels=len(label_to_id))
    model.to(device)  # Move model to GPU if available

    predictions_test = predict(model, tokenizer, test_texts, device=device)
    predictions_train = predict(model, tokenizer, train_texts, device=device)

    predicted_labels_test = [id_to_label[pred] for pred in predictions_test]
    predicted_labels_train = [id_to_label[pred] for pred in predictions_train]
    
    usage_test = ["Public" for i in range(len(test_texts))]
    usage_train = ["Public" for i in range(len(train_texts))]
    
    type_test = ["Test" for i in range(len(test_texts))]
    type_train = ["Train" for i in range(len(train_texts))]

    # Add predictions to DataFrame
    df_test = pd.DataFrame({'ID': test_ids, 'Usage': usage_test, 'Text': test_texts, 'Label': test_labels, 'PredictedLabel': predicted_labels_test, 'Type': type_test})
    df_train = pd.DataFrame({'ID': train_ids, 'Usage': usage_train, 'Text': train_texts, 'Label': train_labels, 'PredictedLabel': predicted_labels_train, 'Type': type_train})
    final_df = pd.concat([df_test, df_train], ignore_index=True)
    final_df.to_csv('data/train_predictions.csv', index=False)

def main():
    # os.makedirs('results', exist_ok=True)
    # train_model()
    inference()

if __name__ == "__main__":
    main()