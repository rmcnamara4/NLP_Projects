# Week 3 – Word Embeddings

## 📌 Objective
The goal of this project is to build and evaluate domain-specific word embeddings for biomedical text. Using a large collection of PubMed abstracts, I train two different Skip-Gram models with Gensim’s Word2Vec implementation: one based on standard word tokenization, and another based on subword-level Byte Pair Encoding (BPE) tokenization. The objective is to compare how tokenization choices impact the quality of learned embeddings in a medical domain.  

To assess embedding quality, I use both **intrinsic evaluation** methods (cosine similarity checks, UMAP visualization, word analogy tasks, nearest neighbor queries) and **extrinsic evaluation** against the **UMNSRS dataset** (a human-annotated benchmark of biomedical term similarity and relatedness). Performance is quantified using **Pearson** and **Spearman** correlations with gold-standard similarity scores.  

## 🧩 Skills & Concepts
- **NLP & Embeddings**: Skip-gram training, BPE vs. word tokenization, similarity/analogy evaluation.  
- **Biomedical Text Processing**: Domain-specific embeddings from PubMed abstracts; evaluation on UMSRS dataset.  
- **Evaluation Methods**: Cosine similarity, Pearson/Spearman correlations with human-annotated scores.  
- **Visualization & Analysis**: Dimensionality reduction (UMAP) for embedding space exploration.  
- **Tools & Libraries**: Python, gensim, scikit-learn, UMAP-learn.  

## 📦 Dataset
- **Source:**  
  - **Training Data:** 10,000 PubMed abstracts  
  - **Evaluation Data:** [UMNSRS](https://huggingface.co/datasets/bigbio/umnsrs) (566 term pairs annotated for semantic similarity and relatedness)

- **Size:**  
  - **Training:** 10k abstracts  
  - **Evaluation:** 566 samples

- **Preprocessing:**  
  - Tokenization with both **standard word-level** and **Byte Pair Encoding (BPE)** strategies  
  - Lowercasing and basic text cleaning for abstracts  
  - Evaluation pairs scored by comparing cosine similarity of embeddings against human annotations

## 📂 Project Structure
- `src/` – Core source code and utilities data loading, preprocessing / cleaning, model design, and evaluation
- `notebooks/` – Prototyping and data exploration  
- `results/` - Contains performance metrics, analogy results, and visualizations for each of the two models
- `get_data.py` - Script to collect the abstracts from PubMed
- `main.py` – Script to run the algorithm 
- `requirements.txt` – Dependencies  

## ⚙️ Setup
```bash
# Create environment
conda create -n week3 python=3.10 -y
conda activate week3

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run 
```bash 
python3 main.py
```

## 🧪 Configuration
Both the Skip-gram and BPE-based models were trained with the same configuration:

- **Vector size:** 200  
- **Context window:** 5  
- **Architecture:** Skip-gram (`sg=1`)  
- **Minimum frequency:** 20 (`min_count=20`)  
- **Epochs:** 10  
- **Workers:** 4 (parallel threads)  
- **Tokenization:**  
  - Skip-gram model: whitespace tokenization  
  - BPE model: byte-pair encoding (BPE) tokenization  

## 📊 Results
I evaluated embeddings on the **UMNSRS** dataset by computing Spearman and Pearson correlations between cosine similarity of embeddings and human similarity scores. Since Byte-Pair Encoding (BPE) reduces the out-of-vocabulary (OOV) problem compared to skip-gram, I also performed a filtered evaluation restricted to the subset of word pairs where both models have representations.

| Model                   | Spearman | Pearson | Num Evaluated |
|--------------------------|----------|---------|---------------|
| Skip-Gram (Full Eval)    | 0.348    | 0.462   | 33            |
| BPE (Full Eval)          | 0.118    | 0.172   | 566           |
| Skip-Gram (Filtered)     | 0.348    | 0.462   | 33            |
| BPE (Filtered)           | **0.449** | **0.534** | 33         |

I also evaluated on hand-crafted medical analogies to qualitatively compare model behavior. Selected examples are shown below:

| Analogy | Skip-gram Predictions | BPE Predictions |
|---------|------------------------|-----------------|
| heart : cardiovascular :: lung : ? | bladder, non-small, thrombosis, vte, adenocarcinoma | pancreatic, bladder, oncologic, hereditary, oncological |
| bacteria : antibiotic :: virus : ? | rsv, prophylaxis, influenza, antiviral, vaccination | prophylaxis, antibiotics, pap, neonates, pylori |
| psychiatry : mental :: cardiology : ? | cardiovascular, heart, dhts, ascvd, diabetes | hrqol, heart, coronary, diabetes, cardiovascular |
| cancer : oncology :: pneumonia : ? | pseudomonas, endophthalmitis, uti, rescue, septic | septic, urinary, bronch, cough, throat |

Other results, including word similarities, UMAP plots, and cosine similarity distributions, can be found in the `results/` directory. 

### 📌 Key Takeaways
- **Coverage vs. Precision Tradeoff**  
  - Skip-gram embeddings suffer from limited vocabulary coverage (only 33 UMNSRS pairs evaluated).  
  - BPE embeddings cover far more terms (566 UMNSRS pairs), but raw scores are much lower.

- **UMNSRS Evaluation**  
  - **Skip-gram (Full/Filtered)**: Spearman = 0.348, Pearson = 0.462 (on 33 pairs).  
  - **BPE (Full Eval)**: Spearman = 0.118, Pearson = 0.172 (on 566 pairs).  
  - **BPE (Filtered to same 33 pairs)**: Spearman = **0.449**, Pearson = **0.534** → **outperforms skip-gram when vocabulary is aligned.**  
  - Suggests BPE captures semantics well but introduces noise on rare/fragmented terms.

- **Qualitative Analogy Evaluation**  
  - **Both models show medically plausible associations**, but their tendencies differ:  
    - Skip-gram: more domain-specific (e.g., “antiviral, vaccination” for virus : antibiotic).  
    - BPE: broader and sometimes noisy, but can capture related biomedical concepts (e.g., “oncological, hereditary”).  
  - Example patterns:  
    - *heart : cardiovascular :: lung : ?*  
      - Skip-gram → bladder, thrombosis (mixed relevance).  
      - BPE → pancreatic, oncological (somewhat related, but diffuse).  
    - *bacteria : antibiotic :: virus : ?*  
      - Skip-gram → antiviral, vaccination (solid match).  
      - BPE → antibiotics, pylori (less precise).  
    - *psychiatry : mental :: cardiology : ?*  
      - Both models predict “cardiovascular/heart” terms → correct association.  
    - *cancer : oncology :: pneumonia : ?*  
      - Skip-gram → septic, pseudomonas, UTI (plausible).  
      - BPE → septic, bronch, cough, throat (closer to real-world pulmonary context).

- **Overall Insights**  
  - **Skip-gram** = higher precision but poor coverage.  
  - **BPE** = broad coverage, and when vocabulary aligns, can outperform skip-gram quantitatively.  
  - **Hybrid Approach Potential**: Using skip-gram for precision + BPE for coverage could yield the best results.  
  - Qualitative analogies confirm both models encode meaningful medical relationships, though skip-gram leans more specific while BPE is more general.

## 🧠 Engineering Notes
- **Repo layout:** `data_utils` (stream/clean PubMed abstracts), `preprocessing_utils` (tokenization pipelines), `model_utils` (gensim Word2Vec build/save/load), `evaluation_utils` (cosine, analogies, UMNSRS scorer), `plot_utils` (UMAP/nearest-neighbor viz), scripts: `get_data.py` (corpus prep), `main.py` (train + eval).
- **Corpus handling:** streamed ~10k abstracts; normalized Unicode, lowercased, stripped punctuation/extra whitespace; optional number masking to reduce sparsity.
- **Two tokenization tracks:**
  - **Word tokens:** simple whitespace + basic normalization; keeps clinical terms intact but suffers OOV.
  - **BPE tokens:** trained a subword vocab (configurable vocab size/merges); serialized tokenizer to reuse across runs; ensured the same BPE model is used for train/eval.
- **OOV policy:** for word-level model, pairs with missing terms are excluded from metrics; for BPE model, subword decomposition eliminates most OOV—logged which pairs were filtered to compare fairly.
- **Analogy evaluation:** standard vector algebra `b - a + c`; removed source tokens from candidate set; limited to top-k; case-normalized queries to match training.
- **Similarity evaluation:** used cosine similarity; ensured L2 normalization before batch scoring for speed; vector lookup wrapped with graceful fallbacks.
- **UMAP viz:** reduced on a consistent sample of vocab (min freq threshold) to avoid bias; cached 2D coords to avoid recompute.

## 🗺️ Next Steps
- **Scale the corpus**: Increase PubMed abstracts (≥100k) and add domain corpora (PMC OA); expect big gains for Skip-gram.
- **Phrase mining**: Learn bigrams/trigrams (e.g., `Phrases` in gensim) so terms like “myocardial infarction” become single tokens before training.
- **Hyperparam sweeps**: Grid/Optuna over `dim`, `window`, `min_count`, `negative`, `epochs`, and BPE merges/vocab size; log UMNSRS + analogy metrics.
- **Compare baselines**: Train CBOW and GloVe (via `glove-python-binary` or `gensim` wrapper) for a stronger baseline panel.
- **Downstream eval**: Plug embeddings into a simple classifier (e.g., PubMed sentiment/topic) to get extrinsic metrics, not just intrinsic similarity.
- **Better analogies**: Curate biomedical analogy sets; evaluate top-k accuracy systematically (not just qualitative lists).

## 🔗 References
- Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft). Chapter: Vector Semantics and Embeddings. [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
- Stanford CS224n: Natural Language Processing with Deep Learning (2019, 2021). Lectures on Word Vectors. [http://web.stanford.edu/class/cs224n/](http://web.stanford.edu/class/cs224n/)
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv preprint [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
- Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP. [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf)
- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). *Enriching Word Vectors with Subword Information.* TACL. [https://arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606)
- Gensim Documentation: [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)
- fastText Documentation: [https://fasttext.cc/](https://fasttext.cc/)

