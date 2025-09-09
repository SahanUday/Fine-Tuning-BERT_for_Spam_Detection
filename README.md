# Fine-Tuning BERT for Spam Detection

This comprehensive t## 9. Create DataLoaders
## 11. Define Model Arch## 13. Find Class Weights
This code block computes class weights to address the imbalance between non-spam (86.6%) and spam (13.4%) classes in the training dataset:

### Why Class Weights Matter:
- **Imbalanced Dataset**: Spam messages are much fewer than non-spam messages
- **Balanced Learning**: Assigns higher weight to the minority spam class
- **Improved Performance**: Ensures the model's loss function prioritizes learning to identify spam
- **Better Results**: Leads to higher accuracy and balanced F1-scores for both classes

## 14. Define the Loss Function - Negative Log-Likelihood Loss
Configure the loss function for training:

### How Negative Log-Likelihood (NLL) Works in Classification:
The goal is for the model to output a probability distribution over classes and accurately predict the true class.

1. **Model Outputs Probabilities**: The neural network ends with a LogSoftmax layer that converts raw scores into log probabilities
2. **Calculating Loss with NLL**: For each training example, the model predicts a probability for the correct class
   - **NLL Loss Formula**: `NLL = -log(p_true)`
   - **Penalty System**: Penalizes the model more when it assigns low probability to the correct class

The loss function uses the computed class weights to handle the imbalanced dataset effectively.ure
Create a custom neural network class (`BERT_Arch`) that combines BERT with classification layers:

### Architecture Components:
- **BERT Base**: Pre-trained BERT model for feature extraction
- **Dropout Layer**: Prevents overfitting (0.1 dropout rate)
- **ReLU Activation**: Non-linear activation function
- **Dense Layer 1**: Transforms 768-dimensional BERT output to 512 dimensions
- **Dense Layer 2**: Final classification layer (512 → 2 classes)
- **LogSoftmax**: Outputs log probabilities for classification

The forward pass extracts the [CLS] token embedding from BERT's output, which represents the entire sequence.

## 12. Set Up the Optimizer - Adam (Adaptive Moment Estimation)
Configure the AdamW optimizer for training:
- **Smart Learning**: Adjusts learning rates for each parameter individually using past gradient information
- **Mini-batch Gradient Descent**: Updates weights using small batches of data
- **Learning Rate**: Set to 1e-3 (0.001) for stable convergence
- **Default Choice**: Often the first choice for deep learning models unless there's a specific reason to use alternativesblock ensures that the tokenized training and validation data are organized into batches for efficient model training and evaluation:

- **RandomSampler** for training helps the model learn robust patterns by randomizing the order of samples during each epoch
- **SequentialSampler** for validation ensures consistent evaluation results
- **DataLoader objects** facilitate iteration over the datasets in batches, which is critical for training the BERT model on a GPU
- **Batch size of 32** balances memory usage and training efficiency

## 10. Freeze BERT Parameters
This step prevents the pre-trained BERT model's weights from being modified during training.

### Purpose:
- **Preserves pre-trained knowledge**: The transformer layers and embeddings retain their learned representations
- **Focuses training on classification layers**: Only the custom dense layers (`fc1`, `fc2`) added in the `BERT_Arch` class are updated
- **Faster training**: Reduces the number of parameters to update, speeding up the training process
- **Feature extraction**: Uses BERT as a sophisticated feature extractor for the spam classification task

### Focus on Classification Layers:
- The `BERT_Arch` class adds dense layers (`nn.Linear(768, 512)`, `nn.Linear(512, 2)`) on top of BERT's output
- Only these custom layers' parameters are updated during training
- You can add multiple linear layers with non-linear activation functions (ReLU, etc.) and dropout for more complex architecturesal will guide you step-by-step through fine-tuning a BERT model for spam detection using the provided Jupyter notebook. Each step is explained in simple English with detailed explanations of the concepts and techniques used.

## 1. Install Required Libraries
First, we install the Hugging Face Transformers library, which provides the pre-trained BERT model and tokenizer. This library makes it easy to work with state-of-the-art transformer models.

```bash
!pip install transformers
```

## 2. Import Libraries and Set Up Environment
Import all necessary Python libraries including:
- **PyTorch**: For deep learning operations and tensor computations
- **Pandas & NumPy**: For data manipulation and numerical operations
- **Scikit-learn**: For data splitting and evaluation metrics
- **Seaborn & Matplotlib**: For data visualization
- **Transformers**: For BERT model and tokenizer
- **GPU Setup**: Configure the device to use CUDA for faster training

## 3. Load the Dataset
Load the spam dataset from the CSV file (`spamdata_v2.csv`) for training and evaluation. This step involves:
- Reading the CSV file using pandas
- Displaying the first few rows to understand the data structure
- Checking the dataset shape and class distribution
- Understanding the balance between spam and non-spam messages

## 4. Split the Dataset into Train, Validation, and Test Sets
This section splits the dataset into three parts for proper model training and evaluation:
- **Training Set (70%)**: Used to train the model
- **Validation Set (15%)**: Used to tune hyperparameters and select the best model
- **Test Set (15%)**: Used for final evaluation on unseen data

The split is stratified to maintain the same class distribution across all sets.

## 5. Import BERT Model and BERT Tokenizer
Load the pre-trained BERT model and tokenizer from Hugging Face:
- **BERT Model**: `bert-base-uncased` - a pre-trained transformer model
- **Tokenizer**: Converts text into tokens that BERT can understand
- Demonstrate tokenization with sample text to show how it works

## 6. Analyze Text Length Distribution
Before tokenization, analyze the length distribution of messages in the training set to determine an appropriate maximum sequence length. This helps in setting the `max_seq_len` parameter efficiently.

## 7. Tokenize the Dataset
This block tokenizes the training, validation, and test texts using the BERT tokenizer with the following specifications:

### Purpose of max_seq_len = 25
- **Tokenization Constraint**: The `max_seq_len` variable defines the maximum number of tokens (including special tokens like [CLS] and [SEP]) that the BERT tokenizer will include in the tokenized representation of each input text.

### Padding and Truncation:
- **Padding**: If a text sequence has fewer tokens than `max_seq_len`, the tokenizer adds padding tokens ([PAD]) to make the sequence length equal to `max_seq_len`.
- **Truncation**: If a text sequence has more tokens than `max_seq_len`, the tokenizer truncates the sequence to fit within this limit.

The tokenization process converts text into:
- **Input IDs**: Numerical representation of tokens
- **Attention Masks**: Indicates which tokens are actual words vs padding

## 8. Convert Integer Sequences to Tensors
Convert the tokenized sequences and labels into PyTorch tensors using `torch.tensor()`. This format is required for model training and ensures compatibility with PyTorch operations.

## 8. Create DataLoaders
Wrap the tensors into DataLoader objects for efficient batching and shuffling during training and validation. Use random sampling for training and sequential sampling for validation.

## 9. Freeze BERT Parameters
Freeze the pre-trained BERT model’s parameters so only the new classification layers are trained. This uses BERT as a feature extractor and speeds up training.

## 10. Define the Model Architecture
Create a custom neural network class (`BERT_Arch`) that adds new dense (fully connected) layers on top of BERT’s output. These layers are responsible for classifying messages as spam or not spam.

## 11. Set Up the Optimizer
Use the AdamW optimizer, which is well-suited for training deep learning models like BERT.

## 12. Handle Class Imbalance
Calculate class weights to address the imbalance between spam and non-spam messages. This ensures the model pays more attention to the minority class (spam).

## 13. Define the Loss Function
Use the Negative Log-Likelihood Loss (NLLLoss), which works well with classification tasks and the model’s output format.

## 15. Fine-Tune BERT - Training and Evaluation Functions

### Training Function:
Trains the BERT-based model on the training dataset for one epoch:
- **Forward Pass**: Processes batches through the model
- **Loss Calculation**: Computes training loss using NLL loss
- **Backward Pass**: Calculates gradients using backpropagation
- **Gradient Clipping**: Prevents exploding gradients (clipped to 1.0)
- **Parameter Updates**: Updates model weights using the optimizer

### Evaluation Function:
Evaluates the model on the validation dataset without updating parameters:
- **Model Evaluation Mode**: Disables dropout layers
- **No Gradient Computation**: Uses `torch.no_grad()` for efficiency
- **Validation Loss**: Computes loss on validation data for model selection

## 16. Start Model Training
Execute the complete training loop:

### Training Process:
- **10 Epochs**: Trains for 10 complete passes through the training data
- **Best Model Selection**: Saves the model with the lowest validation loss as `saved_weights.pt`
- **Loss Tracking**: Monitors both training and validation losses
- **Early Stopping**: Prevents overfitting by saving the best performing model

### Performance Monitoring:
- **Training Loss**: Decreases from ~0.196 to ~0.076 over 10 epochs, indicating effective learning
- **Validation Loss**: Fluctuates with best performance around epoch 3 (~0.091)
- **Overfitting Indicators**: Some fluctuation suggests potential overfitting due to frozen BERT parameters and small dataset

## 17. Load the Best Saved Model
Load the model weights that achieved the lowest validation loss during training. This ensures we use the best performing version of our model for final evaluation.

## 18. Evaluate Model Performance on Test Data
Test the trained model on the unseen test dataset:

### Evaluation Metrics:
- **Classification Report**: Provides precision, recall, F1-score, and support for each class
- **Confusion Matrix**: Visual representation of prediction accuracy
- **Overall Accuracy**: Typically achieves ~99% accuracy on the test set
- **Balanced Performance**: Good performance on both spam and non-spam classes

### Visualization:
Create a heatmap of the confusion matrix to visualize model performance and identify any classification patterns or errors.

## 19. Create Prediction Function for New Messages
Implement a function to predict whether new, unseen messages are spam or not:

### Prediction Process:
1. **Tokenize New Text**: Apply the same tokenization process used during training
2. **Convert to Tensors**: Transform tokenized text into PyTorch tensors
3. **Model Inference**: Pass through the trained model to get predictions
4. **Class Prediction**: Convert model output to binary classification (spam/not spam)

### Example Usage:
Test the model with sample messages to demonstrate its practical application:
- Spam-like message: "Congratulations! You've won a free iPhone!"
- Normal message: "Hey, how are you doing today?"

## 20. Additional Model Validation
Perform additional validation on the validation set to ensure robust performance assessment and verify that the model generalizes well across different data splits.

---

## Key Takeaways

This comprehensive tutorial demonstrates:
- **Transfer Learning**: Leveraging pre-trained BERT for spam detection
- **Efficient Fine-tuning**: Freezing BERT parameters and training only classification layers
- **Class Imbalance Handling**: Using weighted loss functions for better performance
- **Proper Evaluation**: Using separate validation and test sets for unbiased performance assessment
- **Practical Application**: Creating a function for real-world spam detection

The final model achieves excellent performance (99% accuracy) while being computationally efficient due to the frozen BERT parameters approach.

For detailed code implementation and step-by-step execution, see the `Fine_Tuning_BERT.ipynb` notebook in this repository.