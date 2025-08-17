# Week 8 - PubMed Article Summarization using Pegasus (Encoder - Decoder) 

## 📌 Objective
The goal of this project is to develop and evaluate an abstractive text summarization system using the **PEGASUS** model, fine-tuned on biomedical and scientific articles. This work directly builds on my previous GPT-2 summarization project, enabling a **comparative analysis of summarization quality and efficiency** between the two architectures.  

Specifically, the objectives are to:  
- Implement **middle and dynamic chunking strategies** to handle long input documents.  
- Explore **dynamic chunking based on embedding similarity**, leveraging abstracts during training and mean chunk embeddings during inference.  
- Fine-tune PEGASUS with **LoRA (Low-Rank Adaptation)** for parameter-efficient training, while maintaining the option for **gradual unfreezing** as an alternative.  
- Train and manage experiments using **PyTorch Lightning** with Hydra-driven configuration, detailed logging, and checkpointing.  
- Package and deploy the model with **FastAPI** inside a Docker container, and publish the fine-tuned model on **Hugging Face Hub** for reproducibility and public access.  

## 🧩 Skills & Concepts
This project demonstrates applied expertise across several areas of modern NLP and machine learning:  

- **Abstractive Summarization** – Using PEGASUS, a Transformer model pre-trained with the gap-sentence generation objective, to generate fluent and concise summaries.  
- **Biomedical NLP** – Fine-tuning on PubMed-style articles to adapt summarization for the healthcare/clinical domain.  
- **Parameter-Efficient Fine-Tuning (LoRA)** – Applying low-rank adaptation for efficient training without updating the full model.  
- **Model Training Frameworks** – Leveraging PyTorch Lightning for modularity, reproducibility, and built-in features like checkpointing and early stopping.  
- **Configuration Management** – Using Hydra to organize and control experiments through structured YAML configuration files.  
- **Chunking Strategies** – Implementing middle chunking, dynamic chunking, and embedding-based chunking to handle long input sequences.  
- **Experiment Tracking & Logging** – Recording metrics and managing runs to compare fine-tuning strategies.  
- **Model Deployment** – Serving the model via a FastAPI application inside Docker, with endpoints for summarization.  
- **Model Sharing** – Publishing the fine-tuned PEGASUS model to Hugging Face Hub for reproducibility and accessibility.  

## 📦 Dataset
This project uses the **PubMed Summarization Dataset** from [Hugging Face Datasets](https://huggingface.co/datasets/pubmed).  
The dataset consists of biomedical research articles, where the **abstract** serves as the reference summary for the **article body**.

### Data Split
- **Training set:** 13,000 articles
- **Validation set:** 2,000 articles
- **Test set:** 2,000 articles

### Chunking Strategy
- Articles are split into **chunks** to fit within Pegasus' maximum context window.
- Each article can have at most **6 chunks**.
- For consistency and better coverage of the paper’s core content, **the middle 6 chunks** are used when available.
- I also experiment with **dynamic chunking** where the chunks with the highest similarity scores to the abstract (training) or article (inference) embeddings are chosen

### Hierarchical Summarization Workflow
1. **Chunk Summarization:** Each chunk is summarized individually using Pegasus.
2. **Final Summarization:** The chunk summaries are concatenated and passed back into Pegasus for a final, concise summary.

This hierarchical approach allows summarization of **long-form biomedical texts** while respecting Pegasus' input length limitations.

## 📂 Project Structure
The core structure of the repo is as follows: 
- `src/` – Core source code for lightning data module, model module, evaluation, etc. 
- `outputs/` – Contains full outputs from each model run including logs, Hydra configs, evaluation metrics, predictions, etc. 
- `scripts/` – Contains the full scripts to train and evaluate a model 
- `configs/` - Contains all of the Hydra configuration files for determining the settings of the modeling and evaluation process
- `requirements.txt` – Dependencies  

Additional: 
- `api/` – Source code for the FastAPI application, including request handling, inference logic, and route definitions. Containerized to serve the summarization model as an API.  
- `huggingface_models/` – Directory containing the trained PEGASUS model files prepared for upload and versioning on Hugging Face Hub.  
- `Dockerfile` – Instructions to build a Docker image for the API, defining the runtime environment, dependencies, and execution steps for containerized deployment.  
- `requirements_docker.txt` – Dependency list used specifically for the Docker build to ensure a consistent environment inside the container.    

## ⚙️ Setup
```bash
# Create environment
conda create -n week8 python=3.10 -y
conda activate week8

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run 
```bash 
python3 -m scripts.train 
python3 -m scripts.evaluate
python3 -m scripts.save_to_hf
```

## 🧪 Configuration (Hydra)
This project uses **Hydra** for modular, composable configs. The root `config.yaml` wires together sub-configs in `configs/` so you can swap components (optimizer, scheduler, LoRA, generation strategy, etc.) without touching code.

**What’s in the main config**
- **Defaults block:** selects the active sub-configs  
  (`optimizer/adamw`, `scheduler/cosine_annealing`, `datamodule/default`,  
  `model/default`, `trainer/default`, `lora/default`,  
  callbacks for `early_stopping` + `model_checkpoint`, and generation presets for chunk vs. final).
- **Hydra output dirs:** runs log to `outputs/${project.name}_${project.version}`;  
  checkpoints go to `checkpoints/${project.name}_${project.version}`.
- **Experiment controls:** `seed` for reproducibility; `resume: true` to pick up from the latest checkpoint.

**Key sub-configs you can swap**
- **`optimizer/*`** – e.g., `adamw` (lr, weight_decay).  
- **`scheduler/*`** – e.g., `cosine_annealing` (ties to `trainer.max_epochs`).  
- **`datamodule/*`** – batch sizes, workers, and **chunking strategy**  
  (middle-chunking vs. dynamic chunk selection via embedding similarity; max/min lengths, stride, padding value).  
- **`model/*`** – PEGASUS base, dropout, **gradual unfreezing** toggles.  
- **`lora/*`** – enable/disable LoRA and set rank/alpha/target modules.  
- **`trainer/*`** – epochs, accelerator/devices, mixed precision, grad clip, accumulation, validation frequency.  
- **`callbacks/*`** – early stopping (monitor `val_loss`, patience), checkpointing (top-k, filename, dir).  
- **`generation/*`** – separate settings for **chunk summaries** and **final summary**  
  (beams, sampling, length/repetition penalties, max tokens, no-repeat n-gram).  
- **`save_model/*`** – toggle saving, format (HF/`state_dict`), output path.  
- **`project/*`** – name/version used to namespace logs and checkpoints.

## 📊 Results
Model performance was evaluated on the PubMed test set (2,000 articles) using ROUGE metrics, with abstracts serving as the reference summaries. Below are the results for the middle chunking and dynamic chunking models compared with the best performing GPT-2 model: 

| Model Variant | Chunking Strategy | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------------|------------------|---------|---------|---------|
| **GPT-2**     | Middle Chunking  | 0.280   | 0.044   | 0.154   |
| **PEGASUS**   | Middle Chunking  | **0.430**   | **0.151**   | **0.232**   |
| **PEGASUS**   | Dynamic Chunking | 0.424   | 0.141   | 0.223   |

Below are a few examples of generated summaries using GPT-2 vs Pegasus with middle chunking. 

**Example 1**

**Reference:**  
*research on the implications of anxiety in parkinson 's disease ( pd ) has been neglected despite its prevalence in nearly 50% of patients and its negative impact on quality of life . \n previous reports have noted that neuropsychiatric symptoms impair cognitive performance in pd patients ; however , to date , no study has directly compared pd patients with and without anxiety to examine the impact of anxiety on cognitive impairments in pd . \n this study compared cognitive performance across 50 pd participants with and without anxiety ( 17 pda+ ; 33 pda ) , who underwent neurological and neuropsychological assessment . \n group performance was compared across the following cognitive domains : simple attention / visuomotor processing speed , executive function ( e.g. , set - shifting ) , working memory , language , and memory / new verbal learning . \n results showed that pda+ performed significantly worse on the digit span forward and backward test and part b of the trail making task ( tmt - b ) compared to the pda group . \n there were no group differences in verbal fluency , logical memory , or tmt - a performance . in conclusion , \n anxiety in pd has a measurable impact on working memory and attentional set - shifting .*

**GPT-2 Prediction:**  
*backgrounddepressive and anxiety disorders are common comorbidities that affect cognitive function , and cognitive performance is affected by the severity and severity of depression and anxiety symptoms . here \n , we aimed to compare cognitive performance among patients with moderate cognitive impairment with those with severe cognitive impairment.subjects and methodsthe data were analyzed using the spss software ( version 16.0 , chicago , il ; spss inc . , ibm , usa ) . the data were statistically analyzed using one - way anova followed by mann - whitney u - test.resultsdecreases in the level of depression symptoms were associated with a significant increase in the levels of anxiety symptoms , and a significant decrease in global cognition and mmse.conclusionsdecrease in anxiety symptoms was associated with an increase in global cognitive performance , but this effect was not significant in either group.conclusiondecreased anxiety symptoms were related to a significant change in global functioning , but the effect was*

**Pegasus Prediction:**
*background : anxiety is the most common neuropsychiatric symptom in patients with parkinson 's disease ( pd ) . however , the neural basis of anxiety and its impact on cognition in pd remain poorly understood . the present study examined the relationship between anxiety , depression , apathy , and cognitive performance in a sample of pd patients . anxiety was measured using the hospital anxiety and depression scale ( hads ) and depression using the hamilton depression rating scale ( ham - d ) while cognitive performance was assessed using the rapid assessment battery for the assessment of neuro.materials and methods : twenty - two patients with pd and 22 healthy controls ( hc ) were recruited for this study . all participants were assessed for anxiety using the hads and objective . cross - sectional data was taken from a patient database that has been compiled over the past 8 years ( since 2008 ) at the university of sydney.results:there were no group differences in age , gender , disease duration , or depression . there were significantly higher total hads scores in the pda+ group compared to the hc group ( p  0.001 ) but no significant group differences were observed on the digit span forward and immediate recall tests . patients with dementia had significantly worse performance on the trail making test part a and the logical memory i immediate recall test compared to healthy controls.conclusions:the present study suggests that*

---

**Example 2**

**Reference:**  
*background and objective . \n antimicrobial resistance is now a major challenge to clinicians for treating patients . \n hence , this short term study was undertaken to detect the incidence of multidrug - resistant ( mdr ) , extensively drug - resistant ( xdr ) , and pandrug - resistant ( pdr ) bacterial isolates in a tertiary care hospital . \n material and methods . \n the clinical samples were cultured and bacterial strains were identified in the department of microbiology . \n the antibiotic susceptibility profile of different bacterial isolates was studied to detect mdr , xdr , and pdr bacteria . \n results . the antibiotic susceptibility profile of 1060 bacterial strains was studied . \n 393 ( 37.1% ) bacterial strains were mdr , 146 ( 13.8% ) strains were xdr , and no pdr was isolated . \n all ( 100% ) gram negative bacterial strains were sensitive to colistin whereas all ( 100% ) gram positive bacterial strains were sensitive to vancomycin . \n conclusion . \n close monitoring of mdr , xdr , or even pdr must be done by all clinical microbiology laboratories to implement effective measures to reduce the menace of antimicrobial resistance .*

**GPT-2 Prediction:**  
*background . in this cross - sectional study \n we investigated the prevalence and risk factors of c. faecalis infection among hospitalized patients in the republic of saudi arabia ( riyadh ) of india . \n\n\n methods . we conducted a retrospective study of hospitalized patients who were diagnosed with an acute respiratory tract infection ( ards ) during january 2009 to january 2012 , who had been hospitalized for more than 6 months , and who were suspected of having an ards outbreak in the past 6 months. \n \n results . of the hospitalized patients , 81.3% were mdr infected , and 44.6% were ccd infected . among those with a diagnosis of ards , the prevalence was higher in mdr than in ccd cases ( p < 0.001 ) . however , there was no statistically significant difference between the two groups ( p > 0.05).conclusionsthe prevalence of ais was higher among mdr*

**Pegasus Prediction:**
*background : multidrug resistant ( mdr ) staphylococcus aureus and escherichia coli are the major causes of nosocomial infections . hence , this short - term study was undertaken to detect the incidence of mdr bacterial isolates in a tertiary care hospital of central india.materials and methods : bacterial strains were isolated from different clinical samples and were identified by conventional methods . antibiotic susceptibility test of bacterial strains was done by kirby bauer disc diffusion method as per clinical laboratory standard institute ( clsi ) guidelines.results:among 250 samples , 127 ( 6.7% ) were positive for mdr and 41 ( 2.1% ) for extended spectrum beta - lactamase ( xdr ) . e. coli was the commonest isolate ( 31.6% ) followed by klebsiella pneumoniae ( 26.6% ) and pseudomonas aeruginosa ( 20% ) among mdr isolates.conclusion:the prevalence of xdr mdr among gram negative bacilli ( gnb ) isolates was found to be 15.1% . xdr gram positive cocci ( gpc ) are the most common cause of infection in intensive care units ( icus ) of tertiary care teaching hospital . therefore , it is important to screen all patients admitted to icu with gpc to prevent mdr infection . early identification and treatment of such isolates is essential to reduce the morbidity and mortality associated with*

### 📌 Key Takeaways
- **PEGASUS > GPT-2**: PEGASUS consistently outperformed GPT-2 in both ROUGE scores and clinical quality.  
- **Best Chunking Strategy**: Middle-chunk summaries performed the strongest; dynamic-chunking was competitive but not superior.  
- **Content Distribution Insight**: Central sections of biomedical articles tend to hold the most summary-relevant information.  
- **Practical Training**: LoRA fine-tuning achieved strong results with minimal parameter overhead.  

## 🧠 Engineering Notes
- **Compute**: Fine-tuned PEGASUS on **NVIDIA L4 GPU** provisioned via **GCP VM** for scalable cloud training.  
- **Chunking Strategies**: Implemented both fixed middle-chunking and dynamic chunking pipelines for long biomedical texts.  
- **Trainer Setup**: Used PyTorch Lightning `Trainer` with mixed precision (FP16), gradient accumulation, and early stopping.  
- **LoRA Fine-tuning**: Applied parameter-efficient fine-tuning to reduce compute/memory overhead while adapting PEGASUS.  
- **Evaluation**: Automated **ROUGE** scoring pipeline, paired with qualitative review of generated summaries.  
- **Hugging Face Integration**: Packaged and pushed fine-tuned models with inference-ready configuration for sharing and deployment.  
- **Deployment**: Wrapped inference in a **FastAPI** service, containerized with **Docker** for reproducibility and portability.  
- **Secrets Management**: Used a `.env` file to securely load the Hugging Face Hub token into the Dockerized FastAPI app.  
<!-- 
## 🔎 Deployment
You can build and run the FastAPI summarization service with Docker:

```bash
# 1. Clone the repository and navigate into it
git clone <your-repo-url>
cd <your-repo-name>

# 2. Build the Docker image
docker build -t summarization-service .

# 3. Run the container (pass your Hugging Face token from a .env file if required)
docker run -p 8000:8000 summarization-service
```
**Example Request:**
```bash
curl -X POST "http://localhost:8000/summarize" \
    -H "Content-Type: application/json" \
    -d '{"article": "The quick brown fox jumps over the lazy dog. This sentence is often used as a typing test."}'
``` -->

<!-- **Response:**
```bash
{
  "summary": "The quick brown fox jumps over a lazy dog."
}
``` -->

## 🗺️ Next Steps
- **Add Deployment Guide** – Document how to run the Docker container locally and (optionally) deploy to a cloud service like AWS, Azure, or GCP.  
- **Expand Evaluation** – Compare generated summaries against baselines (e.g., lead-3, BART, or PEGASUS) using metrics like ROUGE, BLEU, and BERTScore.  
- **Enhance Preprocessing** – Experiment with better chunking strategies and domain-specific text cleaning.  
- **Model Improvements** – Try fine-tuning a larger summarization model or applying techniques like LoRA for efficiency.  
- **Monitoring & Logging** – Add logging, request tracking, and simple monitoring to capture usage stats and errors.  
- **Frontend or API Client** – Build a lightweight UI or notebook client for easier interaction with the summarization API.  

## 🔗 References
- **Key Papers**
  - T5: *Exploring the Limits of Transfer Learning...* (Raffel et al.)
  - BART: *Denoising Sequence-to-Sequence Pre-training...* (Lewis et al.)
  - LoRA: *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al.)
- **Parameter-Efficient Tuning**
  - [AdapterHub](https://adapterhub.ml) – Library for Adapters
- **Tutorials**
  - Hugging Face: Summarization with BART/T5