import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class SentenceTransformerDataset(Dataset):
    def __init__(self, sentences, max_length=128):
        self.sentences = sentences
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

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
            'attention_mask': encoding['attention_mask'].squeeze()
        }


class SentenceTransformer(nn.Module):
    def __init__(self, embedding_dim=768, num_layers=6, num_heads=8, ff_dim=2048, dropout=0.1):
        super(SentenceTransformer, self).__init__()

        # Embedding layer (vocabulary size from BERT tokenizer)
        self.embedding = nn.Embedding(30522, embedding_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output pooling layer
        self.pooling = MeanPooling()

    def forward(self, input_ids, attention_mask):
        # Get embeddings
        x = self.embedding(input_ids)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create attention mask for transformer
        # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Pass through transformer
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())

        # Pool the output
        sentence_embedding = self.pooling(x, attention_mask)

        # Normalize the embedding
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        return sentence_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class MeanPooling(nn.Module):
    def forward(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


# Test the implementation
def test_model():
    # Sample sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "I love programming in Python."
    ]

    # Create dataset
    dataset = SentenceTransformerDataset(sentences)
    dataloader = DataLoader(dataset, batch_size=2)

    # Initialize model
    model = SentenceTransformer()
    model.eval()

    # Get embeddings
    with torch.no_grad():
        for batch in dataloader:
            embeddings = model(batch['input_ids'], batch['attention_mask'])
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Sample embedding norm: {torch.norm(embeddings[0])}")
            print(f"First 5 values of first embedding: {embeddings[0][:5]}")


if __name__ == "__main__":
    test_model()
