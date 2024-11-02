import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import Dict, List, Tuple

from task1 import *


class MultiTaskDataset(Dataset):
    def __init__(self, sentences: List[str], 
                 topic_labels: List[int], 
                 sentiment_labels: List[int], 
                 max_length: int = 128):
        self.sentences = sentences
        self.topic_labels = topic_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

        # Define the label mappings
        self.topic_classes = {
            0: "Technology",
            1: "Science",
            2: "Politics",
            3: "Entertainment",
            4: "Sports"
        }

        self.sentiment_classes = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'topic_label': torch.tensor(self.topic_labels[idx], dtype=torch.long),
            'sentiment_label': torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        }


class MultiTaskTransformer(nn.Module):
    def __init__(self, 
                 embedding_dim: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 dropout: float = 0.1,
                 num_topic_classes: int = 5,
                 num_sentiment_classes: int = 3):
        super(MultiTaskTransformer, self).__init__()

        # Shared layers (from previous implementation)
        self.embedding = nn.Embedding(30522, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = MeanPooling()

        # Task-specific heads
        self.topic_classifier = TaskHead(
            input_dim=embedding_dim,
            hidden_dim=256,
            num_classes=num_topic_classes
        )

        self.sentiment_classifier = TaskHead(
            input_dim=embedding_dim,
            hidden_dim=256,
            num_classes=num_sentiment_classes
        )

        # Task-specific layer normalization
        self.topic_norm = nn.LayerNorm(embedding_dim)
        self.sentiment_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids, attention_mask):
        # Shared encoding
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        pooled = self.pooling(x, attention_mask)

        # Task-specific processing
        topic_features = self.topic_norm(pooled)
        sentiment_features = self.sentiment_norm(pooled)

        topic_output = self.topic_classifier(topic_features)
        sentiment_output = self.sentiment_classifier(sentiment_features)

        return {
            'topic': topic_output,
            'sentiment': sentiment_output,
            'shared_embedding': pooled
        }


class TaskHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super(TaskHead, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        # Task-specific attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=1,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, x):
        # Apply task-specific attention
        x_attended, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x_attended = x_attended.squeeze(1)

        # Feed-forward classification
        return self.ffn(x_attended)


class MultiTaskTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.topic_criterion = nn.CrossEntropyLoss()
        self.sentiment_criterion = nn.CrossEntropyLoss()

        # Task balancing parameters
        self.task_weights = {
            'topic': 0.5,
            'sentiment': 0.5
        }

    def train_step(self, batch, optimizer):
        self.model.train()
        optimizer.zero_grad()

        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        topic_labels = batch['topic_label'].to(self.device)
        sentiment_labels = batch['sentiment_label'].to(self.device)

        # Forward pass
        outputs = self.model(input_ids, attention_mask)

        # Calculate losses
        topic_loss = self.topic_criterion(outputs['topic'], topic_labels)
        sentiment_loss = self.sentiment_criterion(outputs['sentiment'], sentiment_labels)

        # Combined loss with task weights
        total_loss = (
            self.task_weights['topic'] * topic_loss + 
            self.task_weights['sentiment'] * sentiment_loss
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'topic_loss': topic_loss.item(),
            'sentiment_loss': sentiment_loss.item()
        }


def test_multitask_model():
    # Sample data
    sentences = [
        "The new AI technology is revolutionary!",
        "I didn't enjoy the movie at all.",
        "The sports team played amazingly well."
    ]
    topic_labels = [0, 3, 4]  # Technology, Entertainment, Sports
    sentiment_labels = [2, 0, 2]  # Positive, Negative, Positive

    # Create dataset and dataloader
    dataset = MultiTaskDataset(sentences, topic_labels, sentiment_labels)
    dataloader = DataLoader(dataset, batch_size=2)

    # Initialize model and trainer
    model = MultiTaskTransformer()
    trainer = MultiTaskTrainer(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    for batch in dataloader:
        losses = trainer.train_step(batch, optimizer)
        print("Training losses:", losses)

        # Get predictions
        with torch.no_grad():
            outputs = model(batch['input_ids'], batch['attention_mask'])
            topic_preds = torch.argmax(outputs['topic'], dim=1)
            sentiment_preds = torch.argmax(outputs['sentiment'], dim=1)
            print(f"Topic predictions: {topic_preds}")
            print(f"Sentiment predictions: {sentiment_preds}")


if __name__ == "__main__":
    test_multitask_model()
