# Week 10 - RAG-Based LLM System for Alzheimer's Q&A

## 📌 Objective
The goal of this project is to build a **Retrieval-Augmented Generation (RAG)** system that can answer **natural language questions related to Alzheimer’s disease** by grounding responses in **peer-reviewed biomedical literature**.

Using documents from **PubMed Central** as the knowledge base and large language models (LLMs) for generation, this pipeline enables **accurate, faithful, and explainable question answering** for use cases such as:

- Supporting researchers in literature review  
- Providing evidence-backed insights to clinicians or caregivers  
- Demonstrating how RAG can mitigate hallucinations in high-stakes domains like healthcare

The system was evaluated using both retrieval and generation metrics, including **hit@5**, **MRR**, **faithfulness**, and **correctness**, with strong performance across all dimensions.

## 🧩 Skills & Concepts
- Retrieval-Augmented Generation (RAG) architecture  
- Dense passage retrieval with FAISS  
- Prompt engineering for biomedical Q&A  
- LLM-as-a-judge evaluation (faithfulness, correctness, hit@5, MRR)  
- Integration with AWS S3 and Bedrock  
- Pipeline modularization and result tracking  
- Handling hallucination and grounding in high-stakes domains  
- Biomedical NLP with PubMed Central as a corpus  

## 📦 Dataset
- **Source:** [PubMed Central Open Access Subset](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/)  
- **Size:** 50 full-text biomedical articles related to Alzheimer’s disease  
- **Preprocessing:** Filtered by Alzheimer’s-related keywords and MeSH terms; extracted relevant sections, cleaned, and chunked into passages for dense retrieval  

## 📂 Project Structure
- `src/` – Source code for data collection, chunking / embedding, retrieval, LLM calls, and evaluation
- `prompts/` - Prompt templates 
- `index/` - Holds FAISS index and configuration
- `run_logs/` – Holds parameters used for each script run; documents runs for reproducibility 
- `scripts/` - Holds scripts to run entire RAG pipeline
- `results/` - Holds RAG answers and the evaluation outputs
- `environment.yml` – Holds packages and metadata for creating environment

## ⚙️ Setup
```bash
# Create environment
conda env create -f environment.yml
conda activate rag-env
```

## 🚀 How to Run 
This project includes multiple modular scripts for retrieval, generation, and evaluation — all configurable via command-line arguments.

### Example: run_rag.py
```bash
python3 -m scripts.run_rag \
--chunks_json data/processed/processed_chunks.jsonl \
--retriveal_json data/retrieved_chunks.jsonl \
--prompt_dir prompts/version_1 \
--use_s3 \
--model_id us.anthropic.claude-3-5-haiku-20241022-v1:0
--max_context_char 6000
--max_out_tokens 500
--temperature 0.2
--top_p 0.95
--out ./results/rag_answers.jsonl
```
### Using S3 Storage

To read inputs or write outputs directly to S3 buckets, use the `--use_s3` flag in any script that supports it. You must also ensure that your AWS credentials are properly configured (via `~/.aws/credentials`, environment variables, or IAM roles if on EC2/SageMaker).

### Example: Saving Documents to S3

```bash
python3 -m scripts.fetch_pmc \
--query "Alzheimer* AND (review OR \"meta-analysis\")" \
--max_results 50 \
--date_from 2018 \
--date_to 2025 \
--use_s3 \
--raw_key data/raw/raw_xml.jsonl \
--interim_key data/interim/parsed_xml.jsonl
```
### Other Scripts 
- `fetch_pmc.py` – Fetch and parse articles from PubMed Central
- `run_chunking.py` - Chunk the articles from PubMed Central 
- `build_vector_store.py` - Embed the chunks and create the vector store 
- `retrieve.py` - Run retrieval process to gather context for each of the evaluation queries 
- `run_rag.py` - Pass context and queries to the LLM and produce outputs 
- `evaluate_rag.py` - Evaluate retrieval performance and LLM answer performance using LLM-as-a-judge

Run any script via: 
```bash
python3 -m scripts.<script_name> --arg1 value1 --arg2 value2 ...
```

You can view the full list of arguments using: 
```bash 
python3 -m scripts.<script_name> --help
```

## 🧪 Configuration
This project uses Amazon Bedrock for both LLM inference and text embedding (can also use HuggingFace for embeddings). To run the full pipeline, you’ll need to configure access credentials and (optionally) some runtime options.

Create a .env file in the project root with the following keys:
```
# ✅ Required – AWS credentials for Bedrock access
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# 🔄 Optional – AWS region (default: us-east-1)
AWS_DEFAULT_REGION=us-east-1

# 🔄 Optional – Bedrock model IDs (defaults used if not set)
BEDROCK_MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0

# 🧪 Optional – Email for PubMed API (only needed if using PubMed retrieval)
PUBMED_EMAIL=your_email@example.com

# ☁️ Optional – S3 bucket name (only needed if reading/writing to S3)
S3_BUCKET=your-s3-bucket-name
```

## 📊 Results
Evaluation was performed across both **retrieval** and **generation** stages using automatic metrics and manual review. Below are results from a run on a small sample (50 PubMed articles + manually written Alzheimer’s queries).

### 🔍 Retrieval Performance

#### **Binary Match (exact answer in chunk)**
| Metric             | Value    |
|--------------------|----------|
| Hit@5              | 0.889    |
| Recall@5           | 0.500    |
| Precision@5        | 0.222    |
| Mean Reciprocal Rank (MRR) | 0.643 |
| Coverage           | 0.500    |

#### **Semantic Match (≥ 0.8 cosine similarity)**
| Metric             | Value    |
|--------------------|----------|
| Hit@5              | 1.000    |
| Recall@5           | 0.611    |
| Precision@5        | 0.244    |
| Mean Reciprocal Rank (MRR) | 0.680 |
| Coverage           | 1.000    |

> ℹ️ *Semantic retrieval considers an answer "hit" if the retrieved chunk has cosine similarity ≥ 0.8 with the expected answer.*

---

### 🧠 Generation Performance

| Metric                          | Value |
|---------------------------------|--------|
| Faithfulness Score              | 0.86   |
| Correctness Score               | 0.98   |
| Hallucination Rate              | 0.00   |
| Insufficient Info (Faithfulness) | 0.20   |
| Insufficient Info (Correctness)  | 0.00   |
| Abstain Accuracy (No Gold)      | 1.00   |

> ✅ *All metrics are based on manual annotations comparing model output to gold/reference answers.*

### 📝 Example Outputs 
Example 1
---------
**Query**: 
*How does amyloid-beta accumulation contribute to synaptic dysfunction in Alzheimer's disease?*

**Model Answer**: 
*Amyloid-beta (Aβ) peptide contributes to synaptic dysfunction through multiple mechanisms. It can directly act on neuronal tubulin tau, causing rapid separation of tau from microtubules and leading to axon structure collapse [1]. Aβ also interacts with ryanodine receptors, disrupting intracellular calcium homeostasis and causing structural and functional damage to nerve cells [1]. Additionally, Aβ affects N-methyl-D-aspartate (NMDA) receptors, which leads to abnormal protein synthesis, dendritic spine formation, and glutamate homeostasis disorders, ultimately causing neuronal excitotoxicity [1]. These processes collectively result in synaptic dysfunction and can eventually lead to neuronal death.*

Example 2
---------
**Query**: 
*Compare efficacy of cholinesterase inhibitors vs memantine in mild AD.*

**Model Answer**: 
*Insufficient evidence. The provided context does not contain a direct comparison of cholinesterase inhibitors (CHEIs) and memantine for mild Alzheimer's disease (AD). While the context discusses cholinesterase inhibition and some potential novel approaches to AD treatment [1], it does not provide a comparative efficacy analysis between different AD medications. To accurately answer this question, additional research or context would be needed.*

### 📌 Key Takeaways
- **Semantic retrieval significantly outperforms binary retrieval**, achieving perfect hit@5 and full coverage — highlighting the value of embedding-based similarity over strict keyword matching.
- **Generation outputs are highly faithful and correct**, with 0% hallucination and 98% correctness, showing the effectiveness of grounded summarization over retrieved chunks.
- **Faithfulness gaps are often due to insufficient evidence**, not hallucination — 20% of outputs were limited by retrieval quality, not generation errors.
- **Abstain logic is working as intended**, with 100% abstain accuracy when gold answers are unavailable, reflecting strong alignment with fallback behavior.
- **RAG evaluation benefits from dual perspectives**, combining retrieval performance (hit, recall, MRR) and generation quality (faithfulness, correctness) to holistically assess system behavior.

## 🧠 Engineering Notes
- Developed a **modular RAG pipeline** with interchangeable components for document embedding, retrieval, and generation (Claude, Bedrock Titan Embeddings, FAISS).
- Implemented **query-driven retrieval and summarization**, allowing dynamic prompts from clinicians to pull relevant information from biomedical sources.
- Enabled **chunked retrieval and generation**, optimizing token efficiency for long documents using windowed chunking and top-k passage selection.
- Structured the project with **argparse-based CLI scripts**, supporting flexible usage across retrieval, summarization, and evaluation steps.
- Added **S3 integration and `.env` config management**, supporting both local and cloud-based usage with optional PubMed API querying.
- Standardized **metrics for evaluation**, including faithfulness, correctness, hallucination rate, and retrieval performance (hit@k, recall, MRR).
- Integrated **semantic vs. binary retrieval evaluation**, supporting cosine similarity thresholds for more nuanced quality assessments.
- Designed for **reproducibility and extensibility**:
  - Clear directory structure and run logging per experiment.
  - Optional Bedrock model override via `.env` or script arguments.

## 🗺️ Next Steps
- **Scale to full dataset** — Expand beyond the initial 50 articles to evaluate performance over a larger and more diverse set of real-world biomedical documents.
- **Expand evaluation set** — Increase the number of clinician-style questions and corresponding gold answers to improve evaluation robustness and error analysis.
- **Hyperparameter testing** — Systematically experiment with key decoding and retrieval parameters (e.g., max context length, `k` in top-k retrieval, temperature, top-p) to optimize faithfulness and coverage.
- **Integrate additional models** — Test higher-capacity and domain-specialized LLMs (e.g., Claude 3.5 Sonnet, GPT-4, Med-PaLM, Mixtral) to compare answer quality, cost, and hallucination rates.
- **Expand RAG evaluation** — Incorporate retrieval variants (e.g., hybrid sparse-dense, ColBERT) and assess impact on faithfulness and correctness.
- **Fine-tune retrieval and reranking** — Add trainable rerankers (e.g., monoBERT, cross-encoders) to prioritize clinically relevant context chunks.
- **Batch inference pipeline** — Refactor scripts into a robust pipeline for monthly batch runs across new members and queries, with optional scheduling.
- **Add clinical feedback loop** — Integrate optional physician review or human-in-the-loop scoring to align system answers with medical decision-making needs.
- **Deployment-ready packaging** — Containerize with Docker and define callable endpoints for integration into downstream care management platforms.
- **Monitoring & drift detection** — Use tools like Evidently to detect retrieval or generation drift over time and flag degraded model behavior.
- **Cost tracking & optimization** — Log token usage and latency for Bedrock and open-weight models, evaluating trade-offs between quality and compute.

## 🔗 References
### 🔬 Key Papers
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
  Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. (2020)  
  [📄 arXiv:2005.11401](https://arxiv.org/abs/2005.11401)  
  *Seminal paper introducing the RAG framework combining dense retrieval and generation models.*

---

### 🧰 Tools & Libraries
- **FAISS** – Facebook AI Similarity Search  
  [🔗 GitHub](https://github.com/facebookresearch/faiss)  
  *Efficient similarity search and clustering of dense vectors.*

- **Pinecone** – Managed vector database  
  [🔗 pinecone.io](https://www.pinecone.io/)  
  *Scalable vector search API for building semantic search and RAG applications.*

- **Weaviate** – Open-source vector search engine  
  [🔗 weaviate.io](https://weaviate.io/)  
  *Schema-based, real-time vector search platform.*

- **Milvus** – High-performance vector database  
  [🔗 milvus.io](https://milvus.io/)  
  *Cloud-native vector DB supporting billions of embeddings.*

- **LangChain** – Framework for LLM orchestration  
  [🔗 langchain.com](https://www.langchain.com/)  
  [🔗 GitHub](https://github.com/langchain-ai/langchain)  
  *Useful for chaining vector search with LLMs for RAG-style pipelines.*

---

### 🧪 Datasets (Optional for Evaluation)
- **HotpotQA** – Factoid QA with multi-hop reasoning  
  [🔗 Dataset on HF](https://huggingface.co/datasets/hotpot_qa)

- **Natural Questions (NQ)** – Real search queries with long/short answers  
  [🔗 Dataset on HF](https://huggingface.co/datasets/natural_questions)

- **TriviaQA** – QA from web-scraped trivia sources  
  [🔗 Dataset on HF](https://huggingface.co/datasets/trivia_qa)

---

### 🧑‍💻 Bonus Tutorials
- **RAG from Scratch with FAISS + Hugging Face**  
  [🔗 Tutorial](https://huggingface.co/blog/rag)  
  *Step-by-step guide to building a RAG pipeline using FAISS and transformers.*

- **LangChain RAG Cookbook**  
  [🔗 LangChain Docs](https://docs.langchain.com/docs/use-cases/question-answering/)  
  *Recipes for combining retrievers, chunkers, and LLMs in LangChain.*  