# Week 4 - Sentiment Analysis with Deep Learning

## 📌 Objective
This project builds upon earlier work from **Week 1**, where the same sentiment analysis dataset was modeled using **traditional machine learning methods** (e.g., logistic regression, SVM, and random forests).  
In Week 4, we revisit the dataset but use deep learning approaches — a standard LSTM and an LSTM augmented with a simple attention mechanism — to compare performance and interpretability against the Week 1 baseline.

The goal is twofold:
1. Assess whether deep learning (with or without attention) can outperform traditional ML on this dataset.
2. Examine how attention affects both accuracy and interpretability in sequence modeling.

## 🧩 Skills & Concepts
- **NLP Preprocessing**: Tokenization, sequence padding, train/validation/test splits.  
- **Word Embeddings**: Learned embeddings for dense word representations.  
- **LSTM Models**: Sequence modeling with configurable hidden layers, dropout, and bidirectionality.  
- **Attention Mechanism**: Simple attention layer for interpretability and performance gains.  
- **Model Training**: Custom PyTorch loops, loss computation, and accuracy tracking.  
- **Visualization**: Training curves and metric curves for interpretability.
- **Modular Code Structure**: Separate utilities for data handling, models, and training.

## 📦 Dataset
- **Source:** 
- **Size:**
- **Preprocessing:** 

## 📂 Project Structure
- `src/` – Core source code and model / data logic  
- `notebooks/` – Prototyping and data exploration  
- `results/` – Overall performance metrics for the models along with loss curves, confusion matrices, etc. 
- `main.py` – Script to run models and perform evaluation
- `requirements.txt` – Dependencies  

## ⚙️ Setup
```bash
# Create environment
conda create -n week4 python=3.10 -y
conda activate week4

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run 
```bash 
python3 main.py
```

<!-- ## 🧪 Configuration  -->

## 📊 Results
| Metric      | Traditional ML | Base Model | Attention Model |
|-------------|----------------|------------|-----------------|
| Accuracy    | 0.812          | 0.878      | 0.874           |
| Precision   | 0.815          | 0.859      | 0.865           |
| Recall      | 0.807          | 0.904      | 0.887           |
| Specificity | 0.817          | 0.852      | 0.862           |
| F1          | 0.811          | 0.881      | 0.876           |
| AUROC       | 0.896          | 0.948      | 0.945           |
| AUPRC       | 0.896          | 0.945      | 0.942           |


### 📌 Key Takeaways
- **Deep Learning Performance**: Both the base LSTM model and the LSTM with a simple attention mechanism performed strongly, achieving ~88% accuracy and high AUROC/AUPRC scores. The base model slightly outperformed the attention model in most metrics for this dataset.
- **Attention Mechanism Impact**: While attention improved **specificity** and **precision** slightly, it did not outperform the base LSTM in recall or overall F1 score. This suggests that for this well-predicted dataset, attention may not significantly boost performance over a well-tuned LSTM.
- **Traditional ML Comparison**: Compared to the Week 1 baseline (traditional ML methods such as Logistic Regression, Naive Bayes, etc.), deep learning approaches achieved higher performacne, demonstrating better ability to capture complex linguistic patterns.
- **Few-Shot Learning Implications**: The main improvement over traditional ML likely comes from the deep model’s ability to learn contextual relationships in text. However, diminishing returns can occur without significant architectural changes (e.g., more complex attention or transformer-based models).
- **Dataset Considerations**: The Amazon Reviews dataset is large and balanced enough that deep learning models can shine; results may differ for smaller or more imbalanced datasets.

## 🧠 Engineering Notes
- **Modular Structure**:  
  - `src/data_utils.py`: Loads, cleans, tokenizes text data.  
  - `src/models.py`: Defines Base LSTM and LSTM + Attention architectures.  
  - `src/model_utils.py`: Handles model creation, saving, and loading.  
  - `src/train_utils.py`: Training loop, evaluation, metrics, and plotting.  
  - `main.py`: Runs the full training and evaluation pipeline.

- **Model Saving**:  
  - All trained models are saved as PyTorch `.pth` objects for later use.

- **Results & Artifacts**:  
  - Metrics saved per model in individual pickle files, with an aggregated summary CSV.  
  - Training loss curves, AUROC, AUPRC, and confusion matrices saved for each model.

- **Reproducibility**:  
  - Seeds set for consistent runs.

- **Attention Layer**:  
  - Implements token-level weighting to highlight important input features.

## 🗺️ Next Steps
- Hyperparameter tuning (hidden size, layers, LR, dropout)
- Try BiLSTM, GRU, or Transformer encoder
- Add subword tokenization (WordPiece/BPE)
- Create simple inference + deployment script

## 🔗 References
- Stanford CS224n Lectures 5–7: RNNs, LSTMs, GRUs.  
- Blog: “Illustrated Guide to Attention” by Lilian Weng.