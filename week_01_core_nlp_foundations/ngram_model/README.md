# Week 1 – N-Gram Language Models

## 📌 Objective
This project implements unigram, bigram, and trigram language models using the Penn Treebank dataset (Wall Street Journal text).  
The goal was to build intuition for one of the simplest and most fundamental NLP approaches — predicting tokens from probability distributions — which serves as a conceptual foundation for more advanced models later in the program.

## 🧩 Skills & Concepts
- N-gram modeling & probability estimation  
- Perplexity and fallback perplexity  
- Text preprocessing & tokenization  

## 📦 Dataset
- **Source:** [Penn Treebank – Hugging Face](https://huggingface.co/datasets/penn_treebank)  
- **Size:** 42,068 sentences (80% train / 20% test)  
- **Preprocessing:** Minimal cleaning (dataset already tokenized/normalized). Added `<s>` and `</s>` sentence delimiters and split on whitespace.  

## 📂 Project Structure
- `src/` – Core source code and model logic  
- `notebooks/` – Prototyping and data exploration  
- `results/` – Perplexity scores and sample generations  
- `main.py` – Script to run models and calculate perplexity  
- `requirements.txt` – Dependencies  

## ⚙️ Setup
```bash
# Create environment
conda create -n week1 python=3.10 -y
conda activate week1

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run 
```bash 
python3 main.py
```

<!-- ## 🧪 Configuration  -->

## 📊 Results
| Model | Perplexity |
|-------|------------|
| Unigram | 622.14 |
| Bigram | 341.82 |
| Trigram | 5,355.64 |
| Laplace-smoothed Trigram | 3,116.60 |
| Fallback Perplexity | **77.85** |

**Example Generations**  
_Unigram:_  
> gulf to from came 's `</s>` basis concerns negotiable morgan worried of $ `<s>` as future `<unk>` the have in nine fees credit mr. cents deficit `</s>` in last 's month the `<s>` vietnamese look because union `</s>` of the oil far of energy and customer in being with high

_Bigram:_  
> `<s>` there were sell a new york-based `<unk>` in junk bonds are being acquired `<unk>` within the analyst at N billion of well above $ N one of cocaine consumed at N to do some other of senior subordinated notes sold its spokesman would n't presented his consulting and other

_Trigram:_  
> `<s>` the idea N years old will be able to put it a canadian newspaper publisher said it named its interim location sources say it just has n't yet found a way to go national with pizza which it should be doing more to private investors the refinancing of campeau

## 📌 Key Takeaways
- **Bigrams significantly reduced perplexity** compared to unigrams, showing that adding immediate context improves predictive power.
- **Trigram perplexity increased sharply** without smoothing due to data sparsity — many trigram combinations never appeared in training.
- **Laplace smoothing** reduced trigram perplexity but was still higher than bigrams, highlighting the trade-off between context length and data availability.
- **Fallback models** (using lower-order n-grams when higher-order counts are missing) gave the best perplexity, balancing context with coverage.
- **Language is inherently sparse**, and any model that tries to predict the next word must find clever ways to deal with uncertainty, data sparsity, and variability in expression.
- Generated samples from higher n-grams were more coherent than unigrams but still lacked grammatical structure — a limitation of pure statistical models. Trigram model could formulate simple phrases, like "will be able to", but lack coherence overall. 


## 🧠 Engineering Notes
- Modularized model training, generation, and evaluation for reusability in future language modeling projects.  
- Implemented smoothed probability estimation to handle unseen n-grams.  
- Added fallback perplexity to balance accuracy and coverage.
- Saved results in JSON format. 

## 🗺️ Next Steps
- Experiment with Kneser-Ney smoothing.  
- Compare performance with neural language models (Week 5 onward).  

## 🔗 References
- Jurafsky & Martin, *Speech and Language Processing*, Ch. 3–4.  
- Hugging Face Datasets – Penn Treebank.  


