# Transformer Project
### Environment setup
Python version: 3.12.7
If you are using MacOS with the latest Anaconda environment, run the following commands:
```
conda create -n fetch
conda activate fetch
conda install numpy
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install conda-forge::transformers
```
Otherwise, please refer to `requirements.txt`.

### Sentence Transformer Implementation
##### How to run
Simply run command `python task1.py`.

##### Choices made
1. Architecture Choices:
    - Used BERT's tokenizer for consistency with existing models.
    - Implemented a 6-layer transformer with 8 attention heads.
    - Set embedding dimension to 768 (standard for BERT-base).
    - Added dropout (0.1) for regularization.
    - Included mean pooling for sentence representation.

2. Key Components:
    - Custom Dataset class for handling text input.
    - Positional encoding layer for sequence information.
    - Transformer encoder layers with multi-head attention.
    - Mean pooling layer for creating fixed-length sentence embeddings.
    - L2 normalization of final embeddings.

3. Notable Features:
    - Handles variable length inputs through padding.
    - Attention masking for padded sequences.
    - Normalized output embeddings.
    - Batch processing support.


### Multi-Task Learning Expansion
##### How to run
Simply run command `python task2.py`.

##### Multi-task
- Task A: sentence classification
- Task B: sentiment analysis

##### Changes made
1. Architecture Changes:
    - Added task-specific heads for topic classification and sentiment analysis.
    - Implemented separate layer normalization for each task.
    - Added task-specific attention mechanisms in each head.
    - Maintained shared transformer encoder layers.

2. New Components:
    - MultiTaskDataset: Handles both topic and sentiment labels.
    - TaskHead: Specialized classification heads with attention.
    - MultiTaskTrainer: Manages training with task balancing.

3. Key Features:
    - Task-specific attention mechanisms to focus on relevant features.
    - Balanced loss function with task weights.
    - Shared embeddings with task-specific normalization.
    - Sample class definitions for both tasks:.
        - Topics: Technology, Science, Politics, Entertainment, Sports.
        - Sentiments: Negative, Neutral, Positive.

4. Training Improvements:
    - Task weight balancing (currently 0.5/0.5).
    - Separate loss tracking for each task.
    - Combined optimizer for end-to-end training.


### Traning Considerations
##### Discuss the implications and advantages of each scenario and explain your rationale as to how the model should be trained given the following:
1. If the entire network should be frozen.
    - Pros:
        - Fastest training time.
        - Prevents catastrophic forgetting.
        - Guarantees preservation of pretrained knowledge.
        - Very low memory requirements during training.
    - Cons:
        - Limited ability to adapt to new domains.
        - Fixed feature representations.
        - May underperform if target domain differs significantly from pretraining.
    - Use case:
        - Target domain is very similar to pretraining domain
        - Very limited training data available.
        - Need for quick inference setup.
        - Computing resources are constrained.

2. If only the transformer backbone should be frozen.
    - Pros:
        - Preserves learned language understanding.
        - Allows task-specific adaptation.
        - Reduced risk of overfitting.
        - Faster training than full fine-tuning.
    - Cons:
        - Less flexible than full fine-tuning.
        - May miss domain-specific languange patterns.
        - Limited ability to adapt to new domains.
    - Use case:
        - Moderate amount of training data available.
        - Target tasks are similar to pretraining objectives.
        - Computing resources are moderate.
        - Need balance between adaptation and stability.

3. If only one of the task-specific heads (either for Task A or Task B) should be frozen.
    - Pros:
        - Allows selective transfer learning.
        - Useful for sequential task adaptation.
        - Can preserve performance on critical tasks.
        - Enables controlled multi-task learning.
    - Cons:
        - May lead to task interference.
        - Complex optimization dynamics.
        - Potential for reduced performance on frozen task.
    - Use case:
        - One task is already well-optimized.
        - Adding new tasks incrementally.
        - Different data availability per task.
        - Need to preserve specific task performance.

##### Consider a scenario where transfer learning can be beneficial. Explain how you would approach the transfer learning process, including:
1. The choice of a pre-trained model.
    - RoBERTa or DeBERTa

2. The layers you would freeze/unfreeze.
    - Graduated unfreezing approach.

3. The rationale behind these choices.
    - Choice of pre-trained model:
        - String performance on diverse NLP tasks.
        - Robust pretrained representations.
        - Efficient fine-tuning characteristics.
    - Choice of layer freezing:
        - Graduated freezing prevents catastrophic forgetting.
        - Different learning rates for different components optimize adaptation.
        - Initial head training establishes task-specific patterns.
        - Progressive unfreezing allows fine-grained control.
        - Layer-wise learning rates respect parameter sensitivity.

### Layer-wise Learning Rate
##### How to run
Simply run command `python task4.py`.

##### Rationale for the specific learning rates for each layer:
1. Learning rate distribution
    - Embeddings: base_lr * 0.9^12
    - Early Encoder Layers: base_lr * 0.9^(12 - n)
    - Middle Encoder Layers: base_lr * 0.9^(6 - n)
    - Later Encoder Layers: base_lr * 0.9^n
    - Task Heads:
        - Topic: base_lr * 1.2^n
        - Sentiment: base_lr * 0.8^n

2. Rationale for each layer:
    1) Embeddings:
        - Contains fundamental vocabulary understanding.
        - Most general features that transfer well.
        - Should change very slowly to preserve learned representations.
        - Highest risk of catastrophic forgetting if changed too quickly.
    2) Early Encoder Layers:
        - Process basic linguistic patterns.
        - Handle syntax and basic semantics.
        - Need stability but slightly more flexibility than embeddings.
        - Gradual increase in learning rate as we go up the layers.
    3) Middle Encoder Layers:
        - Balance between general and specific features.
        - Can adapt more freely to new domains.
        - Handle more complex language understanding.
        - Moderate learning rates allow balanced adaptation.
    4) Layer Encoder Layers:
        - More task-specific features.
        - Can change more rapidly without harming base knowledge.
        - Need flexibility to adapt to specific tasks.
        - Higher learning rates enable faster task adaptation.
    5) Task specific Heads
        - Topic classification:
            - More complex task requiring finer distinctions.
            - Needs more parameter adaptation.
            - Benefits from faster learning.
        - Sentiment analysis:
            - More general task.
            - Can leverage basic language understanding.
            - Benefits from more stable learning.

##### Potential benefits of using layer-wise learning rates for traning DNN
- Prevents vanishing or exploding gradients.
- Maintains stable traning across deep layers.
- Allows better backpropagation through the network.
- Lower learning rates in early layers preserve fundamental knowledge.
- Higher rates in later layers allow task-specific adaptation.
- Balances stability and plasticity.
- Different learning dynamics at different depths.
- Better convergence properties.
- More efficient training process.

##### The multi-task setting benefits from layer-wise learning rates because
- Different tasks may require different adaptation speeds.
- Shared layers need careful tuning to benefit all tasks.
- Task-specific components can learn at appropriate rates.
- Better balance between task performance and knowledge transfer.
- More controlled learning process for complex multi-task scenarios.
