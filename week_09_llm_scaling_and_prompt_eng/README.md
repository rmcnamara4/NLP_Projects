# Week 9 - Prompt Engineering Evaluation

## 📌 Objective
This project evaluates the impact of prompt engineering strategies on the performance of large language models (LLMs) for grade school math word problems.  
The goal was to measure how model size, architecture, and prompting style affect reasoning accuracy in a controlled evaluation.

## 🧩 Skills & Concepts
- Prompt engineering (zero-shot, one-shot, few-shot, Chain-of-Thought)  
- Comparative evaluation of open-source and API-based LLMs  
- JSON-based results logging for reproducibility and analysis  
- Quantized models and scaling laws

## 📦 Dataset
- **Source:** [GSM8K – Hugging Face](https://huggingface.co/datasets/gsm8k)  
- **Size:** 250 grade school math problems (used a random sample from the test set)  
- **Preprocessing:** Standardized prompt format for all models; minimal cleaning.

## 📂 Project Structure
- `src/` – Model loading and generation, evaluation scripts, utilities source code
- `prompts/` - Prompt templates 
- `outputs/` - JSON performance and CSV documentation of generated outputs / predictions for each model and prompt template
- `configs/` – Configuration files using Hydra
- `scripts/run_experiment.py` - Script to run the evaluation using the specified model and prompt template
- `requirements.txt` – Dependencies  

## ⚙️ Setup
```bash
# Create environment
conda create -n week9 python=3.10 -y
conda activate week9

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run 
```bash 
python3 -m scripts.run_experiment
```

## 🧪 Configuration
This project uses **Hydra** for flexible configuration management. 
The main config (*configs/config.yaml*) defines defaults for **model**, **generation**, **prompt**, and **data** settings.

**Directory Layout**
configs/
├── config.yaml          # Main Hydra entry point
├── data/
│   └── default.yaml     # Dataset and preprocessing parameters
├── generation/
│   └── default.yaml     # Inference parameters (temperature, max tokens, etc.)
├── model/
│   ├── openai.yaml      # OpenAI API models
│   ├── quantized.yaml   # Local quantized LLM configs
└── prompt/
    └── default.yaml     # Prompt template and version settings

## 📊 Results
The table below shows accuracy scores for each model–prompt configuration combination on the GSM8K math reasoning benchmark.

| Model          | Zero-Shot | One-Shot | Few-Shot | One-Shot CoT | Few-Shot CoT |
|----------------|-----------|----------|----------|--------------|--------------|
| Mistral-7B     | 0.084     | 0.112    | 0.108    | 0.264        | 0.376        |
| LLaMA-13B      | 0.040     | 0.048    | 0.056    | 0.356        | 0.380        |
| GPT-3.5-Turbo  | 0.268     | 0.284    | 0.268    | 0.812        | 0.820        |

### 📝 Example Outputs 
Below are sample outputs for the same math problem under different prompting strategies (GPT-3.5-Turbo): 

**Problem**
*"Mary is an avid gardener. Yesterday, she received 18 new potted plants from her favorite plant nursery. She already has 2 potted plants on each of the 40 window ledges of her large country home. Feeling generous, she has decided that she will give 1 potted plant from each ledge to friends and family tomorrow. How many potted plants will Mary remain with?"*

| Prompt Style | Model Output | Correct? |
|--------------|-------------|----------|
| Zero-Shot    | "152" | ❌ |
| One-Shot     | "58" | ✅ |
| One-Shot CoT | "Mary has 40 window ledges with 2 potted plants each, so she has a total of 40 * 2 = 80 potted plants already.
She received 18 new potted plants, so she now has 80 + 18 = 98 potted plants in total.
If she gives away 1 potted plant from each of the 40 window ledges, she will give away 40 * 1 = 40 potted plants.
Therefore, Mary will remain with 98 - 40 = 58 potted plants. 
The answer is 58." | ✅ |

### 📌 Key Takeaways
- **Chain-of-Thought prompting (CoT)** drives the largest performance gains across all models, with GPT-3.5-Turbo nearly tripling its accuracy compared to zero-shot.
- **Few-shot prompting without CoT** provides only modest gains over one-shot prompting — the real leap comes from adding reasoning steps.
- **Few-Shot CoT shows diminishing returns** compared to One-Shot CoT; once reasoning steps are introduced, adding more examples yields only small improvements.
- **Prompt style can outweigh model size** — smaller models with CoT prompting often outperform larger models without reasoning prompts.

## 🧠 Engineering Notes
- Designed a modular evaluation pipeline to test multiple LLMs (Mistral-7B, LLaMA-13B, GPT-3.5-Turbo) across various prompt configurations (Zero-Shot, One-Shot, Few-Shot, One-Shot CoT, Few-Shot CoT).
- Implemented configuration management using a main YAML file to control dataset loading, model selection, prompt templates, and evaluation settings, with sub-configs for model-specific parameters.
- Automated prompt generation and injection into the evaluation loop to ensure consistent formatting across models and prompt styles.
- Standardized metric logging, saving results in both CSV and JSON formats for downstream analysis and visualization.
- Built support for both API-based (OpenAI) and local inference (transformer-based models) from the same pipeline.
- Added reproducibility features:
  - Fixed random seeds for consistent dataset sampling.
  - Version-locked model checkpoints and prompt templates.

## 🗺️ Next Steps
- **Expand model coverage** — Test additional open-weight and API-based models (e.g., GPT-4, Claude, Mixtral) to compare reasoning improvements.
- **Add automatic prompt optimization** — Implement algorithms such as Prompt Injection Search, Automatic Chain-of-Thought generation, or LLM-as-a-Judge for refining prompts dynamically.
- **Error type analysis** — Categorize incorrect answers (e.g., reasoning error, calculation slip, misunderstanding of problem statement) to inform prompt adjustments.
- **Multi-turn reasoning** — Explore interactive prompting, where the model can ask clarifying questions before answering.
- **Few-shot selection strategies** — Compare random vs. semantic similarity-based example selection for few-shot and CoT setups.
- **Visualization dashboard** — Build a small UI or notebook visualization to compare performance across prompt types and models more intuitively.
- **Cost-performance trade-off** — Incorporate token usage and API cost tracking alongside accuracy metrics.

## 🔗 References

**Large Language Models (LLMs) and Scaling**
- Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361. [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)  
- EleutherAI. *GPT-Neo and GPT-NeoX Development Blogs*. [https://www.eleuther.ai](https://www.eleuther.ai)  
- DeepSpeed (Microsoft). [https://www.deepspeed.ai](https://www.deepspeed.ai)  
- FairScale (Meta). [https://github.com/facebookresearch/fairscale](https://github.com/facebookresearch/fairscale)  
- Dettmers, T., et al. (2022). *8-bit Optimizers via Blockwise Quantization*. arXiv:2110.02861. [https://arxiv.org/abs/2110.02861](https://arxiv.org/abs/2110.02861)  

**Prompt Engineering & In-Context Learning**
- OpenAI. *Prompt Engineering Guide* (OpenAI Cookbook). [https://platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering)  
- Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. arXiv:2201.11903. [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)  
- Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. arXiv:2203.02155. [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)  
- Hugging Face Spaces / Gradio Demos. [https://huggingface.co/spaces](https://huggingface.co/spaces)  