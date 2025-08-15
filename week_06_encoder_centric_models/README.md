# Week 6 - Sentiment Classification with DistilBERT

## 📌 Objective
This project implements a **binary toxicity classification model** using **DistilBERT** as the encoder, trained on the Civil Comments dataset.  
The goal is to detect toxic comments from text with high recall and balanced precision, despite a heavily imbalanced class distribution (~8% toxic).  

The project explores and compares two pooling strategies — **[CLS] token representation** and **mean pooling** — combined with a **custom classification head**.  
All training, evaluation, and optimization steps are implemented from scratch, with full experiment control through a configuration file.  

## 🧩 Skills & Concepts
- **Transformer-based NLP** – Fine-tuning DistilBERT for binary classification.
- **Custom Model Architecture** – Implemented custom classification head with support for both [CLS] token and mean pooling representations.
- **Class Imbalance Handling** – Applied class weighting to address ~8% toxic class prevalence.
- **Custom Training & Evaluation Loops** – Implemented from scratch with logging, checkpointing, and resume support.
- **Optimization Techniques** – AdamW optimizer, learning rate scheduling with patience/factor.
- **Threshold Tuning** – Selected decision threshold on a dedicated set to maximize F1 score (β = 1).
- **Early Stopping** – Prevented overfitting by monitoring validation loss.
- **Config-Driven Workflow** – All parameters and settings controlled via YAML config file.

## 📦 Dataset
- **Source:** [Civil Comments Dataset](https://www.tensorflow.org/datasets/catalog/civil_comments)
- **Size:** ~1.8M train comments, 50k validation comments, 50k threshold tuning comments, 100k test comments
- **Target:**: `toxicity` column, binarized (toxic vs. non-toxic) with ~8% prevalence in training set.
- **Preprocessing:**  
  - Selected `max_length` to cover 95–97.5% of text lengths.
  - Tokenized using DistilBERT tokenizer.
  - Lowercasing and special token handling via tokenizer.
  - Preserved full punctuation and casing where relevant.

## 📂 Project Structure
- `src/` – Core source code for data loading, preprocessing, data class, model class, trainer class
- `models/` – Contains outputs and metrics for each of the model runs
- `logs/` – Contains logging information recorded during training and evaluation
- `main.py` - Script to run the training and evaluation of the translation model
- `requirements.txt` – Dependencies  

## ⚙️ Setup
```bash
# Create environment
conda create -n week6 python=3.10 -y
conda activate week6

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run 
```bash 
python3 main.py
```

## 🧪 Configuration
All experiment settings are stored in a single YAML config file, enabling reproducible runs and easy parameter tuning.  
Key configurable areas include:  

- **Model Parameters** – Base model name (e.g., `distilbert-base-uncased`), classifier head dimension, dropout, pooling strategy (CLS token or mean pooling), option to freeze/unfreeze BERT.  
- **Dataset Settings** – Maximum token length (covers ~95–97.5% of sentences), target column name, random seed, threshold validation split.  
- **Training Settings** – Batch size, number of epochs, early stopping patience, checkpoint/resume capability, class weighting strategies.  
- **Optimization** – Choice of optimizer (AdamW), learning rate, scheduler configuration (patience, factor, mode).  
- **Threshold Optimization** – Step size and beta parameter for F-beta score maximization on a held-out threshold set.  
- **Paths** – Model, checkpoint, training logs, and results directories.  

This structure makes it straightforward to adjust hyperparameters, switch pooling strategies, enable/disable freezing, and run multiple experiments with minimal code changes.

This setup allows you to adjust experiments without modifying the code, ensuring reproducibility and easier iteration.

## 📊 Results
The table below summarizes the test set performance for both model variants — one using the CLS token and the other using mean pooling — across key evaluation metrics.

| Model Variant              | Threshold | Accuracy | Precision | Recall | Specificity | F1 Score | AUROC  | AUPRC  |
|----------------------------|-----------|----------|-----------|--------|-------------|----------|--------|--------|
| DistilBERT (CLS Token, Frozen BERT)  | 0.82      | 0.9120   | 0.4574    | 0.5421 | 0.9441      | 0.4962   | 0.8930 | 0.5077 |
| DistilBERT (Mean Pooling, Frozen BERT) | 0.82      | 0.9208   | 0.5042    | 0.5685 | 0.9514      | 0.5344   | 0.9129 | 0.5706 |


### 📌 Key Takeaways
- **Mean pooling outperformed the CLS token** across nearly all metrics, with notable gains in **AUPRC (+6.3%)** and **AUROC (+2.2%)**.  
- **Recall and F1-score improvements** with mean pooling indicate better sensitivity to the minority (toxic) class.  
- **High specificity (>94%)** in both variants demonstrates strong ability to avoid false positives despite class imbalance.  
- Optimal decision threshold for both models was **0.82**, balancing precision and recall.  
- Class weighting combined with threshold tuning was crucial for handling the ~8% prevalence rate in the dataset.  
- Freezing DistilBERT still yielded strong performance, suggesting that lightweight fine-tuning can be effective with well-designed pooling strategies.

## 🧠 Engineering Notes
- **Modular class-based design:** All components (dataset loader, classifier, trainer, optimizer, scheduler, checkpointing, etc.) are implemented as classes for reusability and maintainability.  
- **Config-driven experiments:** All hyperparameters, paths, and training options are stored in a YAML config file, making it easy to reproduce runs or swap configurations.  
- **Checkpointing & resuming:** Training supports saving/restoring model weights, optimizer state, scheduler state, and training history to continue from the last checkpoint.  
- **Early stopping & LR scheduling:** Implemented patience-based early stopping and `ReduceLROnPlateau` learning rate scheduling for efficient convergence.  
- **Custom pooling strategies:** Both CLS token extraction and mean pooling are custom implemented to compare downstream classification performance.  
- **Threshold optimization set:** A dedicated validation subset is used to select the decision threshold that maximizes F1-score (β=1).  
- **Imbalance handling:** Integrated class weighting (`balanced` strategy) directly into the loss function to account for ~8% positive prevalence.  
- **High-performance training setup:** Ran on a **Google Cloud Platform VM** with an **NVIDIA T4 GPU** for accelerated training.  

## 🗺️ Next Steps
- **Hyperparameter tuning:** Explore additional settings for `max_length`, learning rate, and dropout to further improve generalization.  
- **Unfreeze DistilBERT layers:** Experiment with partial or full fine-tuning of the encoder to potentially boost performance.  
- **Data augmentation:** Investigate NLP-specific augmentation techniques (e.g., back-translation, synonym replacement) to increase diversity in the training data.  
- **Advanced pooling strategies:** Try attention-based pooling or learnable pooling layers for richer sentence representations.  
- **Explainability:** Integrate SHAP or LIME to interpret token-level contributions to predictions.  
- **Deployment:** Containerize with Docker and serve via FastAPI for real-time inference.  

## 🔗 References
- **Key Papers**
  - Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).  
    *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.  
    [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

- **Hugging Face Resources**
  - [Transformers Documentation – BERT Fine-Tuning Tutorials](https://huggingface.co/docs/transformers/tasks/sequence_classification)  
  - [Hugging Face Course – Fine-tuning BERT on GLUE Tasks](https://huggingface.co/course/chapter3)

- **Tutorials & Blogs**
  - [Fine-Tuning BERT for Text Classification](https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894) – Detailed guide with PyTorch and Hugging Face.
  - [BERT for Named Entity Recognition](https://huggingface.co/docs/transformers/tasks/token_classification) – Official HF documentation for token classification tasks.

- **Datasets**
  - [GLUE Benchmark – SST-2](https://gluebenchmark.com/tasks)  
  - [CoNLL-2003 NER Dataset](https://huggingface.co/datasets/conll2003)