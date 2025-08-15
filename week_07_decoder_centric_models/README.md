# Week 7 - PubMed Article Summarization using GPT-2

## 📌 Objective
The goal of this project is to fine-tune **GPT-2** to generate concise, high-quality summaries of PubMed articles, using each article’s **abstract** as the reference summary.  
The system is designed for **hierarchical summarization**, where long articles are first broken into chunks and summarized individually, then those chunk-level summaries are combined and summarized again to produce the final output.  

The implementation emphasizes **reproducibility, modularity, and configurability** by using **Hydra** for experiment management and **PyTorch Lightning** for training orchestration, while incorporating:

- Custom collation and padding
- Custom prompt formatting
- Attention and label masking
- Gradual unfreezing of GPT-2 layers
- Multiple decoding strategies (beam search, sampling, etc.)

## 🧩 Skills & Concepts
This project demonstrates a wide range of **NLP, deep learning, and engineering skills**, including:

- **Abstractive Summarization** with transformer-based language models.
- **Prompt Engineering** for effective conditioning of GPT-2 (`"Summarize this: ... TL;DR:"`).
- **Hierarchical Summarization** to handle long-form biomedical text.
- **Custom Data Collation & Attention Masking** for prompt-label separation and variable-length sequences.
- **Model Fine-Tuning** with gradual layer unfreezing for better adaptation.
- **Experiment Management** with **Hydra** for full configuration control.
- **Training Optimization**:
  - Early stopping and model checkpointing.
  - Gradient clipping and batch accumulation for small-batch training stability.
  - Mixed-precision training (FP16) for speed and memory efficiency.
  - Learning rate scheduling.
- **Evaluation Metrics** for summarization: ROUGE-1, ROUGE-2, ROUGE-L.
- **Engineering Practices**:
  - Modular code structure (`src` and script separation).
  - Custom PyTorch Lightning `DataModule` and `LightningModule` for clean training loops.
  - Logging and reproducible experiment runs.

## 📦 Dataset
This project uses the **PubMed Summarization Dataset** from [Hugging Face Datasets](https://huggingface.co/datasets/pubmed).  
The dataset consists of biomedical research articles, where the **abstract** serves as the reference summary for the **article body**.

### Data Split
- **Training set:** 13,000 articles
- **Validation set:** 2,000 articles
- **Test set:** 2,000 articles

### Chunking Strategy
- Articles are split into **chunks** to fit within GPT-2’s maximum context window.
- Each article can have at most **6 chunks**.
- For consistency and better coverage of the paper’s core content, **the middle 6 chunks** are used when available.

### Hierarchical Summarization Workflow
1. **Chunk Summarization:** Each chunk is summarized individually using GPT-2.
2. **Final Summarization:** The chunk summaries are concatenated and passed back into GPT-2 for a final, concise summary.

This hierarchical approach allows summarization of **long-form biomedical texts** while respecting GPT-2’s input length limitations.

## 📂 Project Structure
- `src/` – Core source code for lightning data module, model module, evaluation, etc. 
- `outputs/` – Contains full outputs from each model run including logs, Hydra configs, evaluation metrics, predictions, etc. 
- `scripts/` – Contains the full scripts to train and evaluate a model 
- `configs/` - Contains all of the Hydra configuration files for determining the settings of the modeling and evaluation process
- `requirements.txt` – Dependencies  

## ⚙️ Setup
```bash
# Create environment
conda create -n week7 python=3.10 -y
conda activate week7

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run 
```bash 
python3 -m scripts.train 
python3 -m scripts.evaluate
```

## 🧪 Configuration (Hydra)
This project uses **Hydra** for fully modular configuration, enabling flexible selection of optimizers, schedulers, generation strategies, and trainer settings without modifying code.  All config files are in the `configs/` directory. 

**Main configuration highlights:**  
- **Optimizer & Scheduler:** Choose from predefined configs (e.g., `adamw`, `cosine_annealing`).  
- **Model:** GPT-2 LightningModule with options for freezing/unfreezing layers, loss aggregation, and logging.  
- **Trainer:** Controls epochs, precision (mixed FP16), gradient clipping, and gradient accumulation.  
- **Callbacks:** Early stopping and checkpointing with custom save paths.  
- **Generation:** Separate configs for chunk-level and final summary generation (beam search, sampling, etc.).  
- **Paths & Logging:** Organized output, checkpoint, and log directories based on project name/version.  
- **Seed & Resume:** Set experiment seed and optionally resume from checkpoints.  

You can change the individual configs within the subdirectories to adjust specific parameters and / or change the main `config.yaml` to use different architectures (e.g., Adam optimizer vs AdamW).

## 📊 Results
Model performance was evaluated on the PubMed test set (2,000 articles) using ROUGE metrics, with abstracts serving as the reference summaries. Below are the results for the model `summarization_v1`.

| Metric  | Score  |
|---------|--------|
| ROUGE-1 | 0.280  |
| ROUGE-2 | 0.044  |
| ROUGE-L | 0.154  |

While the model successfully generated coherent summaries, it often included verbose restatements of source content and occasionally introduced minor factual inconsistencies compared to the reference abstracts. Below are two sample predictions:

**Example 1**

**Prediction:**  
*backgrounddepressive and anxiety disorders are common comorbidities that affect cognitive function , and cognitive performance is affected by the severity and severity of depression and anxiety symptoms . here \n , we aimed to compare cognitive performance among patients with moderate cognitive impairment with those with severe cognitive impairment.subjects and methodsthe data were analyzed using the spss software ( version 16.0 , chicago , il ; spss inc . , ibm , usa ) . the data were statistically analyzed using one - way anova followed by mann - whitney u - test.resultsdecreases in the level of depression symptoms were associated with a significant increase in the levels of anxiety symptoms , and a significant decrease in global cognition and mmse.conclusionsdecrease in anxiety symptoms was associated with an increase in global cognitive performance , but this effect was not significant in either group.conclusiondecreased anxiety symptoms were related to a significant change in global functioning , but the effect was*

**Reference:**  
*research on the implications of anxiety in parkinson 's disease ( pd ) has been neglected despite its prevalence in nearly 50% of patients and its negative impact on quality of life . \n previous reports have noted that neuropsychiatric symptoms impair cognitive performance in pd patients ; however , to date , no study has directly compared pd patients with and without anxiety to examine the impact of anxiety on cognitive impairments in pd . \n this study compared cognitive performance across 50 pd participants with and without anxiety ( 17 pda+ ; 33 pda ) , who underwent neurological and neuropsychological assessment . \n group performance was compared across the following cognitive domains : simple attention / visuomotor processing speed , executive function ( e.g. , set - shifting ) , working memory , language , and memory / new verbal learning . \n results showed that pda+ performed significantly worse on the digit span forward and backward test and part b of the trail making task ( tmt - b ) compared to the pda group . \n there were no group differences in verbal fluency , logical memory , or tmt - a performance . in conclusion , \n anxiety in pd has a measurable impact on working memory and attentional set - shifting .*

---

**Example 2**

**Prediction:**  
*background . in this cross - sectional study \n we investigated the prevalence and risk factors of c. faecalis infection among hospitalized patients in the republic of saudi arabia ( riyadh ) of india . \n\n\n methods . we conducted a retrospective study of hospitalized patients who were diagnosed with an acute respiratory tract infection ( ards ) during january 2009 to january 2012 , who had been hospitalized for more than 6 months , and who were suspected of having an ards outbreak in the past 6 months. \n \n results . of the hospitalized patients , 81.3% were mdr infected , and 44.6% were ccd infected . among those with a diagnosis of ards , the prevalence was higher in mdr than in ccd cases ( p < 0.001 ) . however , there was no statistically significant difference between the two groups ( p > 0.05).conclusionsthe prevalence of ais was higher among mdr*

**Reference:**  
*background and objective . \n antimicrobial resistance is now a major challenge to clinicians for treating patients . \n hence , this short term study was undertaken to detect the incidence of multidrug - resistant ( mdr ) , extensively drug - resistant ( xdr ) , and pandrug - resistant ( pdr ) bacterial isolates in a tertiary care hospital . \n material and methods . \n the clinical samples were cultured and bacterial strains were identified in the department of microbiology . \n the antibiotic susceptibility profile of different bacterial isolates was studied to detect mdr , xdr , and pdr bacteria . \n results . the antibiotic susceptibility profile of 1060 bacterial strains was studied . \n 393 ( 37.1% ) bacterial strains were mdr , 146 ( 13.8% ) strains were xdr , and no pdr was isolated . \n all ( 100% ) gram negative bacterial strains were sensitive to colistin whereas all ( 100% ) gram positive bacterial strains were sensitive to vancomycin . \n conclusion . \n close monitoring of mdr , xdr , or even pdr must be done by all clinical microbiology laboratories to implement effective measures to reduce the menace of antimicrobial resistance .*

---

These examples show the model’s tendency to capture the overall structure and topic while occasionally drifting in specificity and introducing unrelated or fabricated details.

### 📌 Key Takeaways
- **Performance in expected range for GPT-2:** ROUGE-1 of 0.280, ROUGE-2 of 0.044, and ROUGE-L of 0.154 are on par with what is typically seen for GPT-2 on domain-specific summarization tasks without large-scale fine-tuning.
- **Moderate unigram overlap:** Captures key terms and phrases from references, but not always in the same structure.
- **Low bigram overlap:** Difficulty preserving multi-word sequences from the reference summaries.
- **Tendency toward verbosity:** Predictions often exceed reference length, sometimes repeating points or including filler.
- **Occasional factual drift:** Model sometimes introduces content not in the source, especially specific numbers or locations.
- **Maintains abstract-like structure:** Outputs generally follow scientific abstract conventions (Background → Methods → Results → Conclusion), even when details diverge.

## 🧠 Engineering Notes
- **Training Infrastructure:** Model trained on Google Cloud Platform (GCP) using a single NVIDIA L4 GPU for both training and generation.
- **Frameworks & Tools:** Implemented in PyTorch Lightning with Hydra-based configuration management for flexible experiment control.
- **Precision & Performance:** Mixed precision training (FP16) with gradient accumulation enabled to effectively increase batch size given GPU memory constraints.
- **Training Efficiency:** Gradient clipping and early stopping used to stabilize training and prevent wasted compute cycles.
- **Checkpointing & Logging:** Automatic checkpoint saving for best validation loss, with custom directory structures for logs and model artifacts.
- **Data Processing:** Custom tokenization, collation, and masking logic optimized for hierarchical summarization workflow.
- **Generation Flexibility:** Configurable beam search, sampling, and hybrid generation strategies available without code changes.

## 🗺️ Next Steps
- **Model Variants:** Experiment with larger GPT-2 models (Medium, Large, XL) to evaluate performance gains over base GPT-2.
- **Hyperparameter Tuning:** Run Optuna or Ray Tune sweeps for learning rate, dropout rates, and generation parameters.
- **Domain Adaptation:** Pretrain GPT-2 on a larger biomedical corpus (e.g., PubMed abstracts, PMC full text) before fine-tuning for summarization.
- **Evaluation Expansion:** Add human evaluation metrics and task-specific assessments beyond ROUGE (e.g., factual consistency checks).
- **Pipeline Optimization:** Profile preprocessing and generation steps to reduce latency and memory usage.
- **Deployment:** Package the trained summarizer into a FastAPI or Streamlit app for interactive use, possibly containerized with Docker.
- **Compare Architectures:** Benchmark against encoder–decoder architectures like BART, PEGASUS, and BioBART for biomedical summarization tasks.

## 🔗 References
**Papers & Blog Posts**
- Radford, A., et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) – GPT-2 technical report & OpenAI blog.
- Brown, T. B., et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) – GPT-3 paper introducing few-shot capabilities.
- OpenAI Blog. (2019). [Better Language Models and Their Implications](https://openai.com/research/better-language-models) – GPT-2 release blog with safety considerations.
- Hugging Face Blog. [How to Generate Text](https://huggingface.co/blog/how-to-generate) – Practical guide on decoding strategies (greedy, beam search, sampling).
- Hugging Face Blog. [GPT-2 for Text Generation](https://huggingface.co/blog/gpt2) – Walkthrough on using GPT-2 for generation tasks.

**Open-Source Models & Code**
- [Hugging Face GPT-2 Model Page](https://huggingface.co/gpt2) – Base model with weights and usage examples.
- [EleutherAI GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-1.3B) – Open GPT-like model trained on The Pile.
- [Transformers Documentation](https://huggingface.co/docs/transformers/index) – API reference for model loading, fine-tuning, and generation.