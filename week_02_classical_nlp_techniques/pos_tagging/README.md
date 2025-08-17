# Week 2 – Part-of-Speech Tagging

## 📌 Objective
The goal of this project is to implement a **Part-of-Speech (POS) tagging system** using a **Hidden Markov Model (HMM)**. The system learns from a tagged training corpus to estimate the probabilities of word–tag (emission) relationships and tag–tag (transition) sequences. With these probabilities, it applies the **Viterbi algorithm** to predict the most likely sequence of POS tags for unseen sentences.  

This project demonstrates:  
- How probabilistic models can be applied to sequence labeling tasks.  
- The end-to-end process of training, decoding, and evaluating an HMM-based POS tagger.  
- A baseline approach to POS tagging that highlights the trade-offs between simplicity, interpretability, and accuracy compared to modern deep learning methods.  

## 🧩 Skills & Concepts
This project reinforces the following skills and concepts:  

- **Hidden Markov Models (HMMs):** Understanding probabilistic models for sequence data.  
- **Viterbi Algorithm:** Dynamic programming for decoding the most likely tag sequence.  
- **Probability Estimation:** Computing transition and emission probabilities from a training corpus.  
- **Part-of-Speech Tagging:** Applying HMMs to a core NLP task.  
- **Sequence Labeling:** General principles of labeling sequences in structured prediction problems.  
- **Python Implementation:** Writing modular, maintainable code with clear separation of logic (`main.py` for workflow, `viterbi_algorithm.py` for core algorithm).  

## 📦 Dataset
- **Source:** [Brown Corpus](https://www.nltk.org/nltk_data/) (via NLTK)
- **Size:** 45,872 training sentences and 11,468 testing sentences
- **Preprocessing:** The dataset was used with the `universal` POS tagset. Minimal preprocessing was done — empty or malformed sentences were removed.  

[('The', 'DET'), ('Fulton', 'NOUN'), ('County', 'NOUN'), ('Grand', 'ADJ'), ('Jury', 'NOUN'), ('said', 'VERB'), ('Friday', 'NOUN'), ('an', 'DET'), ('investigation', 'NOUN'), ('of', 'ADP'), ("Atlanta's", 'NOUN'), ('recent', 'ADJ'), ('primary', 'NOUN'), ('election', 'NOUN'), ('produced', 'VERB'), ('``', '.'), ('no', 'DET'), ('evidence', 'NOUN'), ("''", '.'), ('that', 'ADP'), ('any', 'DET'), ('irregularities', 'NOUN'), ('took', 'VERB'), ('place', 'NOUN'), ('.', '.')]

The dataset contains token-level part-of-speech (POS) tags in the **Universal POS tagset** format, which includes the following categories:

- **NOUN** – Nouns
- **VERB** – Verbs
- **ADJ** – Adjectives
- **ADV** – Adverbs
- **PRON** – Pronouns
- **DET** – Determiners
- **ADP** – Adpositions
- **CONJ** – Conjunctions
- **PRT** – Particles
- **NUM** – Numerals
- **X** – Other (foreign words, typos, etc.)
- **.** – Punctuation

Each sentence is annotated at the token level with a word and its corresponding POS tag. This dataset is used to train and evaluate the Hidden Markov Model (HMM) for part-of-speech tagging.

## 📂 Project Structure
- `src/` – Core source code and utilities calculating transition and emission probabilities, running the Viterbi algorithm, and evaluating the model
- `notebooks/` – Prototyping and data exploration  
- `results/` - Contains performance metrics and estimated probabilities that went in to the model
- `main.py` – Script to run the algorithm 
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
The model achieved strong performance on the Brown Corpus POS tagging task using the Universal tagset:

- Accuracy: 94.6%

This demonstrates that the Hidden Markov Model (HMM) with Viterbi decoding was effective in learning sequential dependencies and predicting POS tags with high precision. The results indicate that the model generalizes well on unseen data, maintaining consistent accuracy between training and test splits.

### 📌 Key Takeaways
- **Sequential modeling works well:** The HMM with Viterbi decoding effectively captured word-to-tag and tag-to-tag dependencies, achieving high accuracy.
- **Data sparsity matters:** Even with a relatively modest dataset like the Brown Corpus, careful handling of emissions and transitions enabled strong generalization.
- **Universal tagset simplifies evaluation:** Using a standardized, reduced tagset allowed for clearer performance measurement and easier comparison to other approaches.
- **Strong baseline for POS tagging:** This project demonstrates how classical statistical models (HMMs) can still achieve competitive results, serving as a useful benchmark against more advanced neural models. 

## 🧠 Engineering Notes
- **Corpus Handling:** Used the Brown corpus via NLTK with the universal tagset, reducing complexity by mapping fine-grained POS tags to a simplified set of categories.
- **Data Splitting:** Training set (~45k sentences) and test set (~11k sentences) were created for balanced evaluation.
- **Model Implementation:** Implemented a Hidden Markov Model (HMM) from scratch, estimating both emission and transition probabilities directly from the training data.
- **Decoding:** Applied the Viterbi algorithm with smoothing for sequence tagging, ensuring efficient decoding through dynamic programming.
- **Numerical Stability:** Computed log probabilities to prevent underflow during probability multiplications.
- **Modularity:** Organized the codebase into preprocessing, training, decoding, and evaluation modules for clarity and extensibility.

## 🗺️ Next Steps
- Explore higher-order HMMs (trigram dependencies).  
- Experiment with smoothing techniques for emission probabilities.  
- Incorporate additional corpora for improved generalization.  
- Extend implementation to support semi-supervised or unsupervised training.  

## 🔗 References
- *Speech and Language Processing* – Jurafsky & Martin  
- *NLTK Book* – POS tagging & CRFs  
- [Gensim LDA Tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html)  
- Stanford CS224N archived notes  

