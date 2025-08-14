# Week 5 - Transformer Architecture and Translation

## 📌 Objective
The goal of this project was to implement a complete Transformer architecture **from scratch** in PyTorch, including the Encoder, Decoder, and Multi-Head Attention mechanisms, without relying on high-level sequence-to-sequence libraries. This custom implementation was then trained to translate text from **English to French**, demonstrating a full understanding of the Transformer’s inner workings—positional encoding, scaled dot-product attention, feed-forward networks, and layer normalization.

The project showcases both **theoretical comprehension** of the original *Attention is All You Need* paper and **practical engineering skills** in building a fully functional neural machine translation (NMT) system end-to-end.

## 🧩 Skills & Concepts
- **Deep Learning Foundations**
  - Implemented core Transformer components: Encoder, Decoder, Multi-Head Attention, Feed-Forward layers, and Positional Encoding.
  - Applied residual connections, layer normalization, and masking strategies for sequence modeling.
- **Natural Language Processing**
  - Learned and applied sequence-to-sequence modeling for Neural Machine Translation (NMT).
  - Incorporated tokenization, vocabulary building, and handling of special tokens (e.g., `<sos>`, `<eos>`).
- **PyTorch Engineering**
  - Built reusable, modular PyTorch classes for each Transformer subcomponent.
  - Managed custom training loops, loss functions, and evaluation metrics without high-level frameworks.
- **Data Processing**
  - Preprocessed bilingual English–French datasets.
  - Handled batching, padding, and attention masks to accommodate variable-length sequences.
- **Machine Translation**
  - Trained the Transformer on an English-to-French dataset to learn direct translation mappings.
  - Evaluated model performance using BLEU score and qualitative translation examples.
- **Reproducible Research**
  - Structured code for clarity and scalability, enabling future experiments.
  - Documented architecture design choices and hyperparameter configurations.

## 📦 Dataset
- **Source:** Hugging Face `opus_books` dataset (`en-fr` translation pair)
- **Size:** Automatically downloaded; split into train (64%), validation (16%), and test (20%) from the original training split
- **Preprocessing:**  
  - Tokenization using a custom regex-based tokenizer that lowercases text and separates punctuation  
  - Conversion of tokens to numerical IDs via a constructed vocabulary, with `<SOS>` and `<EOS>` tokens added by default  
  - Padding of sequences for batching

## 📂 Project Structure
- `src/` – Core source code for data loading, preprocessing, transformer architecture modules, and training and evaluation modules
- `results/` – Contains corpus BLEU score for the test samples
- `logs/` – Contains logging information recorded during training and evaluation
- `main.py` - Script to run the training and evaluation of the translation model
- `requirements.txt` – Dependencies  

## ⚙️ Setup
```bash
# Create environment
conda create -n week5 python=3.10 -y
conda activate week5

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run 
```bash 
python3 main.py
```

## 🧪 Configuration
All experiment settings are stored in a single `config.yaml` file, organized into sections:

- **Model** – Transformer architecture hyperparameters such as hidden size, attention head count, number of layers, and dropout rates.  
- **Training** – Batch size, learning rate, number of epochs, early stopping patience, and checkpoint resume behavior.  
- **Evaluation** – Beam search parameters including beam width, maximum sequence length, and length penalty.  
- **Dataset** – Name of the dataset and source/target language codes.  
- **Paths** – File and directory locations for saving models, checkpoints, losses, and logs.  

This setup allows you to adjust experiments without modifying the code, ensuring reproducibility and easier iteration.

## 📊 Results
**Corpus BLEU:** `8.94%` on the held-out test set.  

**Example outputs:**  

| Example | Reference | Hypothesis | Notes |
|---------|-----------|------------|-------|
| 1 | `<SOS> - - oui .` | `<SOS> - - oui .` | Exact match |
| 2 | `<SOS> je dois dire toutefois que le pain et le vin manquaient totalement .` | `<SOS> cependant , je me sens de l'avis de ce pain et de vin fut définitivement brutalement définitivement fatiguée .` | Captures “pain” and “vin” but drifts semantically |
| 3 | `<SOS> - - je ne demande qu'à vous obéir , répondit - il .` | `<SOS> – je suis rassurée , repartit l ’ avantage de l ’ impolitesse de l ’ impolitesse des tasses .` | Retains sentence form but content diverges |
| 4 | `<SOS> - - de mon malheur .` | `<SOS> – l ’ avantage de mon malheur royale . l ’ avantage de l ’ avantage de l ’ ambition l l ’ avantage de l ’ ambition l ’ intelligence le tort de l ’ official don cesare l ’ avantage de l ’ intelligence . l ’ avantage de l ’ intelligence le tort de l l l l l ’ avantage de l l l l l l l l l l l l ’ avantage de l veuillez veuillez veuillez veuillez veuillez veuillez veuillez l ’ avantage de l l l l l l l l` | Severe repetition and hallucination |
| 5 | `<SOS> la séance était finie ; la foule se dispersa ; et , maintenant que les discours étaient lus , chacun reprenait son rang et tout rentrait dans la coutume : les maîtres <UNK> les domestiques , et ceux - ci frappaient les animaux , <UNK> <UNK> qui s ’ en retournaient à l ’ étable , une couronne verte entre les cornes .` | `<SOS> la réunion était finie , la foule , et maintenant que les ouvriers avaient lu , chacun retomba dans sa place , et tout rentra dans la vieille classe ; les maîtres juraient les domestiques , les domestiques et les domestiques , les animaux , font les fêtes , les fêtes , se précipitèrent sur les remises de la propontide .` | Partial structure match, but unrelated concepts (“propontide”) |

### 📌 Key Takeaways
- The model demonstrates it can **perfectly translate very short, simple sentences**, showing it has learned basic token mappings.  
- On **medium-length sentences**, it often includes some correct words but fails to maintain full semantic accuracy.  
- **Long sentences** suffer from severe repetition, hallucinations, and topic drift, indicating weaknesses in long-range dependency handling.  
- The low BLEU score is consistent with a **Transformer trained from scratch** on a **small, domain-specific dataset** without pretrained embeddings.  
- Improvements could likely be made through:
  - Using pretrained models (e.g., mBART, MarianMT).  
  - Expanding dataset size and diversity.  
  - Adding regularization and better decoding strategies (beam search tuning, repetition penalty).  

## 🧠 Engineering Notes
- Implemented a complete Transformer architecture from scratch, including:
  - Encoder and Decoder stacks with Multi-Head Attention.
  - Positional encoding without external library dependencies.
  - Custom training loop with teacher forcing.
  - Beam search decoding implemented from scratch for improved sequence generation.
- Managed checkpoints and training state saving/loading for resuming experiments.
- Dataset preprocessing included custom tokenization, padding, and vocabulary management.
- Model training and evaluation were conducted on a **T4 NVIDIA GPU** provisioned through a **GCP VM**, enabling faster experimentation compared to CPU-based training.
- Implemented custom logging to track BLEU scores, sample outputs, and training losses during experiments.
- Modularized code into reusable components (train.py, evaluation.py, checkpoint.py, main.py) and managed experiment runs via a configuration file for improved maintainability and reproducibility.

## 🗺️ Next Steps
- Experiment with **larger Transformer models** or pretrained sequence-to-sequence architectures (e.g., T5, mBART) to improve BLEU score.
- Fine-tune on a **cleaned and domain-specific dataset** to reduce hallucinations and improve semantic accuracy.
- Apply **subword tokenization** (e.g., Byte-Pair Encoding) to handle rare words and morphology more effectively.
- Introduce **label smoothing** and regularization to reduce overfitting and improve generalization.
- Implement **mixed-precision training** for faster computation and lower memory usage.
- Deploy the trained model via **FastAPI + Docker** for interactive translation demos.

## 🔗 References
- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper introducing self-attention and encoder–decoder architecture.
- Alammar, J. (2018). [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual and intuitive explanation of how Transformers work.