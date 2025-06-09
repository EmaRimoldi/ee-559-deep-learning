# HateLens: Tiny LLMs for Efficient and Interpretable Hate Speech Detection

## Introduction

HateLens is a lightweight, transparent pipeline designed to detect and explain hate speech on social platforms. By combining the contextual power of decoder-only TinyLLMs with parameter-efficient fine-tuning and post-hoc explainability, HateLens strikes a balance between high accuracy and minimal computational overhead.

Specifically, HateLens:

- **Leverages TinyLLMs**: Fine-tunes compact, decoder-only language models via Low-Rank Adaptation (LoRA), updating less than 0.05% of parameters to keep memory footprint and inference time low.
- **Ensures Interpretability**: Integrates Local Interpretable Model-agnostic Explanations (LIME) to provide token-level attributions both before and after adaptation, making every classification decision transparent.
- **Maintains Generative Capabilities**: Preserves the base model’s generative strengths by storing only a small set of differential weights, allowing seamless reuse for other tasks.
- **Delivers State-of-the-Art Performance**: On the DynaHate benchmark, our best TinyLLM achieves over 80% accuracy; an improvement of more than 25% compared to its pre-adaptation baseline.

With HateLens, researchers and practitioners gain a fast, reliable, and explainable tool to curb the spread of hateful content without sacrificing efficiency or clarity. Perfect for deployment on edge devices, real-time moderation systems, and research environments where both performance and transparency matter.  




## 📂 Project Structure

Below is an overview of the main folders and scripts in the repository, showing how components of the HateLens pipeline are organized:

📦 HateLens/  
├── 📁 **checkpoints/**  
│   ├── **TinyLlama/**  
│   │   ├── **dynahate/**      ← 3 LoRA-tuned checkpoints (3 seeds) on DynaHate  
│   │   └── **hatecheck/**     ← 3 LoRA-tuned checkpoints (3 seeds) on HateCheck  
│   ├── **Phi-2/**  
│   │   ├── **dynahate/**  
│   │   └── **hatecheck/**  
│   └── **OPT-1.3B/**  
│       ├── **dynahate/**  
│       └── **hatecheck/**  
│  
├── 📁 **data/**               ← Raw & preprocessed datasets  
│   ├── **dynahate/**  
│   └── **hatecheck/**  
│  
├── 📁 **experiments/**        ← YAML configs for each model  
│   ├── `TinyLlama.yaml`      ← Training hyperparameters & LoRA settings  
│   ├── `Phi-2.yaml`  
│   └── `OPT-1.3B.yaml`  
│  
├── 📁 **results/**            ← Evaluation outputs & plots  
│   ├── **metrics/**          ← CSVs of accuracy, F1, etc.  
│   └── **lime/**             ← Token-attribution scores (pre/post FT)  
│  
├── 📁 **utils/**              ← Helper modules & utilities  
│   └── `preprocessing.py`    ← Text cleaning, tokenization scripts  
│  
├── 📄 `requirements.txt`      ← Python dependencies (LoRA, LIME, Transformers…)  
│  
├── 🛠️ `run_training_dynahate.sh`   ← Bash wrapper to train on DynaHate  
├── 🛠️ `run_training_hatecheck.sh`  ← Bash wrapper to train on HateCheck  
│  
├── 🏋️‍♂️ `trainer_dynahate.py`    ← PyTorch Lightning trainer for DynaHate  
├── 🏋️‍♂️ `trainer_hatecheck.py`   ← PyTorch Lightning trainer for HateCheck  
│  
├── 🔍 `evaluate_models.py`     ←  
│     • `--dynahate` → evaluate best checkpoints on DynaHate  
│     • `--hatecheck` → evaluate best checkpoints on HateCheck  
│  
└── 🔍 `compute_lime_scores.py` ←  
      • `--dynahate` → compute LIME attributions on DynaHate  
      • `--hatecheck` → compute LIME attributions on HateCheck  



- **checkpoints/** 📦  
  Contains trained model weights organized by architecture and dataset. Each subfolder holds three seed-based LoRA checkpoints.

- **data/** 🗄️  
  Raw and preprocessed text samples for both DynaHate and HateCheck benchmarks.

- **experiments/** ⚙️  
  YAML configuration files capturing all hyperparameters and LoRA settings used during fine-tuning.

- **results/** 📊  
  Evaluation metrics (accuracy, F1-score) and LIME attribution outputs, ready for plotting and analysis.

- **utils/** 🛠️  
  Utility scripts for data preprocessing, tokenization, and other common tasks.

- **run_training_dynahate.sh** 🏃  
  Shell scripts to kick off batch training jobs on dynahate dataset, selecting model via command-line flags.

- **trainer_dynahate*.py** 🏋️‍  
  Training pipelines leveraging PyTorch (or Lightning) to fine-tune each TinyLLM with LoRA.

- **evaluate_models.py** & **compute_lime_scores.py** 🔬  
  Python scripts to run evaluation and explainability analyses pre- and post-fine-tuning.

