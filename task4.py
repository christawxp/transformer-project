import math
import torch
import torch.nn as nn

from torch.optim import AdamW
from transformers import AutoModel
from typing import Dict, List, Optional


class TaskHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class LayerwiseLearningRateTransformer(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = 'bert-base-uncased',
        num_topic_classes: int = 5,
        num_sentiment_classes: int = 3
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.backbone.config.hidden_size

        self.topic_head = TaskHead(hidden_size, num_topic_classes)
        self.sentiment_head = TaskHead(hidden_size, num_sentiment_classes)
        self.layer_groups = self._create_layer_groups()

    def forward(self, input_ids, attention_mask=None):
        # Get backbone outputs
        backbone_outputs = self.backbone(input_ids, attention_mask=attention_mask)
        pooled_output = backbone_outputs.last_hidden_state[:, 0, :]  # Using [CLS] token

        # Task-specific predictions
        topic_logits = self.topic_head(pooled_output)
        sentiment_logits = self.sentiment_head(pooled_output)

        return {
            'topic': topic_logits,
            'sentiment': sentiment_logits
        }

    def _create_layer_groups(self) -> Dict[str, List[str]]:
        layer_groups = {
            'embeddings': [],
            'encoder_layers': [],
            'pooler': [],
            'topic_head': [],
            'sentiment_head': []
        }

        for name, _ in self.named_parameters():
            if 'embeddings' in name:
                layer_groups['embeddings'].append(name)
            elif 'encoder.layer.' in name:
                layer_groups['encoder_layers'].append(name)
            elif 'pooler' in name:
                layer_groups['pooler'].append(name)
            elif 'topic_head' in name:
                layer_groups['topic_head'].append(name)
            elif 'sentiment_head' in name:
                layer_groups['sentiment_head'].append(name)

        return layer_groups


class LayerwiseLearningRateOptimizer:
    def __init__(
        self,
        model: LayerwiseLearningRateTransformer,
        base_lr: float = 1e-4,
        layer_decay: float = 0.9,
        task_multipliers: Dict[str, float] = None,
        weight_decay: float = 0.01
    ):
        self.model = model
        self.base_lr = base_lr
        self.layer_decay = layer_decay
        self.task_multipliers = task_multipliers or {
            'topic': 1.0,
            'sentiment': 1.0
        }

        # Create optimizer with layer-wise learning rates
        self.optimizer = self._create_optimizer(weight_decay)

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

    def _get_layer_lr(self, layer_name: str, num_layers: int = 12) -> float:
        """Calculate learning rate for a specific layer"""
        if 'embeddings' in layer_name:
            # Embeddings get the lowest learning rate
            return self.base_lr * (self.layer_decay ** num_layers)

        elif 'encoder.layer.' in layer_name:
            # Extract layer number and calculate decay
            layer_num = int(layer_name.split('encoder.layer.')[1].split('.')[0])
            return self.base_lr * (self.layer_decay ** (num_layers - layer_num))

        elif 'pooler' in layer_name:
            # Pooler gets a higher learning rate
            return self.base_lr * (self.layer_decay ** 2)

        elif 'topic_head' in layer_name:
            # Task-specific learning rate for topic head
            return self.base_lr * self.task_multipliers['topic']

        elif 'sentiment_head' in layer_name:
            # Task-specific learning rate for sentiment head
            return self.base_lr * self.task_multipliers['sentiment']

        return self.base_lr

    def _create_optimizer(self, weight_decay: float) -> AdamW:
        """Create optimizer with layer-wise learning rates"""
        # Group parameters by layer and assign learning rates
        param_groups = []
        no_decay = ['bias', 'LayerNorm.weight']

        for name, param in self.model.named_parameters():
            # Calculate learning rate for this layer
            lr = self._get_layer_lr(name)

            # Determine if weight decay should be applied
            should_decay = not any(nd in name for nd in no_decay)

            param_groups.append({
                'params': [param],
                'lr': lr,
                'weight_decay': weight_decay if should_decay else 0.0,
                'layer_name': name  # Store for logging/debugging
            })

        return AdamW(param_groups)

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        class LayerwiseScheduler:
            def __init__(self, optimizer, warmup_steps: int = 1000):
                self.optimizer = optimizer
                self.warmup_steps = warmup_steps
                self.current_step = 0

            def step(self):
                self.current_step += 1
                if self.current_step < self.warmup_steps:
                    # Linear warmup
                    warmup_factor = float(self.current_step) / float(max(1, self.warmup_steps))
                else:
                    # Cosine decay
                    warmup_factor = 0.5 * (1.0 + math.cos(
                        math.pi * (self.current_step - self.warmup_steps) / 
                        (self.current_step - self.warmup_steps + 1)
                    ))

                # Update learning rates
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * warmup_factor

        return LayerwiseScheduler(self.optimizer)


class LayerwiseTrainer:
    def __init__(
        self,
        model: LayerwiseLearningRateTransformer,
        optimizer: LayerwiseLearningRateOptimizer,
        task_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.task_weights = task_weights or {'topic': 0.5, 'sentiment': 0.5}

        # Task-specific losses
        self.topic_criterion = nn.CrossEntropyLoss()
        self.sentiment_criterion = nn.CrossEntropyLoss()

        # Gradient clipping value
        self.clip_value = 1.0

    def train_step(self, batch):
        self.model.train()
        self.optimizer.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(
            batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        # Calculate losses
        topic_loss = self.topic_criterion(
            outputs['topic'],
            batch['topic_labels']
        )
        sentiment_loss = self.sentiment_criterion(
            outputs['sentiment'],
            batch['sentiment_labels']
        )

        # Weighted loss
        total_loss = (
            self.task_weights['topic'] * topic_loss +
            self.task_weights['sentiment'] * sentiment_loss
        )

        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.clip_value
        )

        # Optimizer and scheduler steps
        self.optimizer.optimizer.step()
        self.optimizer.scheduler.step()

        return {
            'total_loss': total_loss.item(),
            'topic_loss': topic_loss.item(),
            'sentiment_loss': sentiment_loss.item()
        }


# Example usage
def demonstrate_layerwise_learning():
    # Create model and optimizer
    model = LayerwiseLearningRateTransformer()
    optimizer = LayerwiseLearningRateOptimizer(
        model,
        base_lr=1e-4,
        layer_decay=0.9,
        task_multipliers={
            'topic': 1.2,  # Higher learning rate for topic classification
            'sentiment': 0.8  # Lower learning rate for sentiment analysis
        }
    )

    # Create trainer
    trainer = LayerwiseTrainer(
        model,
        optimizer,
        task_weights={
            'topic': 0.6,  # More emphasis on topic classification
            'sentiment': 0.4  # Less emphasis on sentiment analysis
        }
    )

    return model, optimizer, trainer


def test_layerwise_learning():
    # Create model
    model = LayerwiseLearningRateTransformer()

    # Create sample batch
    batch = {
        'input_ids': torch.randint(0, 30000, (4, 128)),  # batch_size=4, seq_length=128
        'attention_mask': torch.ones(4, 128),
        'topic_labels': torch.randint(0, 5, (4,)),  # 5 topic classes
        'sentiment_labels': torch.randint(0, 3, (4,))  # 3 sentiment classes
    }

    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Create optimizer
    optimizer = LayerwiseLearningRateOptimizer(
        model,
        base_lr=1e-4,
        layer_decay=0.9,
        task_multipliers={
            'topic': 1.2,
            'sentiment': 0.8
        }
    )

    # Create trainer
    trainer = LayerwiseTrainer(
        model,
        optimizer,
        task_weights={
            'topic': 0.6,
            'sentiment': 0.4
        }
    )

    # Test forward pass
    outputs = model(batch['input_ids'], batch['attention_mask'])
    print("\nModel outputs shape:")
    print(f"Topic logits shape: {outputs['topic'].shape}")
    print(f"Sentiment logits shape: {outputs['sentiment'].shape}")

    # Test training step
    losses = trainer.train_step(batch)
    print("\nTraining losses:")
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value:.4f}")

    # Print learning rates
    print("\nLearning rates by layer:")
    for param_group in optimizer.optimizer.param_groups:
        if 'layer_name' in param_group:
            print(f"Layer: {param_group['layer_name']:<40} LR: {param_group['lr']:.2e}")


if __name__ == '__main__':
    test_layerwise_learning()
