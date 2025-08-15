# Week 2 – Named Entity Recognition

## 📌 Objective
This project compares two NER approaches on the **CoNLL-2003** dataset:

1. **CRF Model** – Uses handcrafted lexical, POS, and neighboring token features, plus BOS/EOS flags, to learn BIO tagging.  
2. **spaCy Transformer (`en_core_web_trf`)** – Generates entity spans, maps them to CoNLL labels, and converts to BIO format for evaluation.

**Goal:**  
Evaluate both models on the same test set and compare a feature-based CRF with a pre-trained transformer.

## 🧩 Skills & Concepts
- **Named Entity Recognition (NER)** using both statistical (CRF) and transformer-based models.
- **BIO Tagging Scheme** for entity span labeling.
- **Feature Engineering** for CRFs:
  - Token shape
  - POS tags and prefixes
  - Contextual features from neighboring tokens
  - BOS/EOS indicators
- **Entity Mapping** from spaCy labels to CoNLL categories.
- **Data Preprocessing** for CoNLL-2003 format.
- **Model Evaluation** with precision, recall, and F1 scores.

## 📦 Dataset
- **Source:** [CoNLL-2003 Dataset](https://www.kaggle.com/datasets/juliangarratt/conll2003-dataset)
- **Size:** 14,041 training sentences and 6,703 testing sentences
- **Preprocessing:** Minimal preprocessing was done for this project. Sentences that were empty were removed from the dataset. 

Below is an example from the train dataset: 

[('EU', 'NNP', 'B-ORG'), ('rejects', 'VBZ', 'O'), ('German', 'JJ', 'B-MISC'), ('call', 'NN', 'O'), ('to', 'TO', 'O'), ('boycott', 'VB', 'O'), ('British', 'JJ', 'B-MISC'), ('lamb', 'NN', 'O'), ('.', '.', 'O')]

The dataset contains token-level labels in the BIO format for four entity types:

- **PER** – Person names
- **LOC** – Locations
- **ORG** – Organizations
- **MISC** – Miscellaneous entities

Each sentence is annotated with word, POS tag, and entity label.  We only focus on the entity label for this project. 

## 📂 Project Structure
- `src/` – Core source code and utilities for creating and evaluating the SpaCy and CRF models
- `notebooks/` – Prototyping and data exploration  
- `results/` - Contains performance metrics for the CRF and SpaCy models evaluated on the test set
- `main.py` – Script to run models and log metrics
- `requirements.txt` – Dependencies  

## ⚙️ Setup
```bash
# Create environment
conda create -n week2 python=3.10 -y
conda activate week2

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run 
```bash 
python3 main.py
```

## 📊 Results
We evaluated both a Conditional Random Field (CRF) model and SpaCy’s `en_core_web_trf` transformer-based NER model on the **CoNLL-2003** dataset.  
Metrics are reported as **F1-scores** for each entity type and aggregated averages.  

The CRF model outperformed SpaCy in most categories, particularly for `ORG` and `MISC`, while SpaCy had a slight edge on `PER`.

| Label        | CRF F1  | SpaCy F1 | Δ F1 (CRF − SpaCy) |
|--------------|---------|----------|--------------------|
| LOC          | 0.897   | 0.783    | **+0.114**         |
| MISC         | 0.835   | 0.703    | **+0.132**         |
| ORG          | 0.792   | 0.540    | **+0.252**         |
| PER          | 0.880   | 0.889    | -0.009             |
| **Micro Avg**| 0.857   | 0.754    | **+0.103**         |
| **Macro Avg**| 0.851   | 0.729    | **+0.122**         |
| **Weighted** | 0.856   | 0.741    | **+0.115**         |


## 📌 Key Takeaways
- **CRF consistently outperforms SpaCy** across most entity types, with the largest improvements in `ORG` (+25.2 F1) and `MISC` (+13.2 F1).  
- **SpaCy only slightly leads on `PER`** (+0.9 F1), likely benefiting from its transformer-based contextual embeddings for personal names.  
- **Feature engineering in CRF remains competitive**, especially for domains with limited or domain-specific training data.  
- The large gap in `ORG` suggests that SpaCy’s general-purpose model struggles with certain organization names without domain-specific fine-tuning.  
- **Label translation was required** for SpaCy predictions (e.g., `GPE → LOC`, `NORP → MISC`), introducing potential inconsistencies—especially where CoNLL uses a single label for multiple semantic types (e.g., `MISC`).  
- **Macro and weighted averages** both favor CRF by over 12 F1 points, showing stronger overall performance despite translation challenges.  

## 🧠 Engineering Notes
- **Models:**
  - **CRF:** Hand-crafted lexical, orthographic, POS, and context features.
  - **SpaCy (`en_core_web_trf`):** Transformer-based NER without fine-tuning.

- **Label Mapping:**
  - SpaCy outputs mapped to CoNLL-2003 tags (e.g., `GPE → LOC`, `NORP → MISC`).
  - Some categories (e.g., `MISC`) are broad, leading to potential misalignments.

- **Processing:**
  - CoNLL-2003 dataset (Kaggle) in BIO format.
  - SpaCy predictions converted to BIO for evaluation.
  - JSON outputs required converting NumPy types to native Python.

- **Environment:**
  - Python 3.10, local execution.
  - Transformer model on CPU; CRF training fast due to lightweight features.

## 🗺️ Next Steps
- Fine-tune the SpaCy transformer model on the CoNLL-2003 dataset to improve performance.  
- Refine the label mapping to better align SpaCy outputs with CoNLL tags and reduce inconsistencies.  
- Enhance CRF features by adding character n-grams, embeddings, and other lexical signals.  
- Expand evaluation to include confusion matrices and span-level F1 scores.  
- Package both models for easy inference and side-by-side evaluation on new data.

## 🔗 References
- Jurafsky, D., & Martin, J. H. *Speech and Language Processing* – Chapters on sequence labeling (HMMs, CRFs) and topic modeling.  
- Bird, S., Klein, E., & Loper, E. *Natural Language Processing with Python* – CRFsuite examples.  
- Stanford CS124 and CS224N archived lectures (older editions with classical tagging content).  

