![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/pytorch-%EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FF6A00?style=for-the-badge&logo=huggingface&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458?style=for-the-badge&logo=pandas&logoColor=white)



# HateLens: Tiny LLMs for Efficient & Interpretable Hate Speech Detection

## 👥 Group 48
| Student's name | SCIPER |
| -------------- | ------ |
| [Vittoria Meroni](https://github.com/vittoriameroni) | 386722 |
| [Emanuele Rimoldi](https://github.com/EmaRimoldi) | 377013 |
| [Simone Vicentini](https://github.com/SimoVice/) | 378204 |

## 📄 Deliverables
- [Screencase](https://drive.google.com/file/d/1CwCQC62-vEC8ymORb9-4itADhIKu_CwO/view?usp=sharing) (download for better quality)

## 🌍 Description

HateLens is a lightweight, transparent pipeline designed to detect and explain hate speech on social platforms. By combining the contextual power of decoder-only TinyLLMs with parameter-efficient fine-tuning and post-hoc explainability, HateLens strikes a balance between high accuracy and minimal computational overhead.

Specifically, HateLens:

- **Leverages TinyLLMs**: Fine-tunes compact, decoder-only language models via Low-Rank Adaptation (LoRA), updating less than 0.05% of parameters to keep memory footprint and inference time low.
- **Ensures Interpretability**: Integrates Local Interpretable Model-agnostic Explanations (LIME) to provide token-level attributions both before and after adaptation, making every classification decision transparent.
- **Maintains Generative Capabilities**: Preserves the base model’s generative strengths by storing only a small set of differential weights, allowing seamless reuse for other tasks.
- **Delivers State-of-the-Art Performance**: On the DynaHate benchmark, our best TinyLLM achieves over 80% accuracy—an improvement of more than 25% compared to its pre-adaptation baseline.

With HateLens, researchers and practitioners gain a fast, reliable, and explainable tool to curb the spread of hateful content without sacrificing efficiency or clarity. Perfect for deployment on edge devices, real-time moderation systems, and research environments where both performance and transparency matter.  

## Data

### 🧐 Dynahate

The Dynamically Generated Hate Speech Dataset (DynaHate) [1] was selected for its broad coverage of diverse hate expressions and its balanced class distribution (54% hate, 46% not hate). This eliminates the need for oversampling or augmentation, improving training stability.

DynaHate was developed through an iterative human-model co-annotation process, yielding over 41,000 synthetic samples labeled as hate or nothate. Each entry includes numerous metadata, such as the hate type (Animosity, Derogation, Dehumanization, Threatening, and Support for Hateful Entities) and target group. The dataset is licensed under CC-BY 4.0.


🔗 Useful Links:
- 📄 [Official Repository](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset)
  
### 🕵️ Hatecheck

The HateCheck suite [2] was chosen for its fine-grained, functional evaluation of hate speech models across 29 targeted tests and seven protected groups (women, trans people, gay people, Black people, disabled people, Muslims, immigrants). This allows for stable and diagnostic model assessments without the need for data augmentation or oversampling.

HateCheck was constructed via expert-designed templates and manual case crafting to produce 3,901 candidate examples, of which 3,728 were retained after validation by five trained annotators. Each case is labeled as hateful or non-hateful and annotated with rich metadata. The dataset is released under a CC-BY 4.0 license.

🔗 Useful Links:
- 📄 [Official Repository](https://github.com/paul-rottger/hatecheck-data)


📚 Citations:
- [1] Vidgen, B., et al. (2021). Learning from Machines: Dataset Generation with Dynamic Human-Model Co-Annotation. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing.
- [2] Röttger, P., et al. (2021). HateCheck: Functional Tests for Hate Speech Detection Models. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics.

## 📦 Conda Env

To run the provided Python scripts and notebooks, create an environment using python ? , to replicate our results. 
Using the requirements.txt file inside the repo, run the following commands:

```bash

```

Once created, activate the environment with:

```bash

```

Observation: there could be problems with the follwoing packages: bitsandbites, e  scipy, in caso ci fossero problemi utilizzare le seguenti versioni: 






## 🧱 Project Structure

```text
ee-559-deep-learning/
│
├── data/                              # Raw & preprocessed datasets
│   ├── dynahate/
│   └── hatecheck/
│
├── experiments/                       # YAML configs for each model
│   ├── TinyLlama/                     # LoRA & training hyperparameters
│   │   └── config.yaml
│   ├── phi-2/                         # LoRA & training hyperparameters
│   │   └── config.yaml
│   └── opt/                           # LoRA & training hyperparameters
│       └── config.yaml
│
├── results/                           # Evaluation outputs & explainability scores
│   ├── dynahate/                  
│   └── hatecheck/
│
├── utils/                             # Helper modules & scripts
│               
├── requirements.txt                   # Python, PyTorch, Transformers, LoRA, LIME…
│
├── run_training_dynahate.sh           # Bash wrapper to train on DynaHate
├── run_training_hatecheck.sh          # Bash wrapper to train on HateCheck
│
├── trainer_dynahate.py                # PyTorch Lightning trainer for DynaHate
├── trainer_hatecheck.py               # PyTorch Lightning trainer for HateCheck
│
├── evaluate_models.py                 # Evaluate best model performances
│
└── compute_lime_scores.py             # Compute LIME scores displayed in results/      

```

## 🗂️ Folder Highlights

- `data/` — Raw and preprocessed DynaHate & HateCheck datasets ready for training and evaluation.  
- `experiments/` — YAML config files specifying LoRA & training hyperparameters for each model.  
- `results/` —  
  - `dynahate/`: Metrics plots, LIME scores barplots, for dynahate  
  - `hatecheck/`: Metrics plots, LIME scores barplots, for dynahate  
- `utils/` — Helper modules for text dataset creation, visualization, and common utilities.  

- **Root‐level scripts**:  
  - `run_training_dynahate.sh` & `run_training_hatecheck.sh`: launch dataset‐specific training.  
  - `trainer_dynahate.py` & `trainer_hatecheck.py`: PyTorch Lightning training pipelines.  
  - `evaluate_models.py`: evaluate best checkpoints on DynaHate or HateCheck. Use --<datasets_name> to select the dataset to evaluate.  
  - `compute_lime_scores.py`: generate LIME explanations for selected models. Use --<dataset_name> to select the dataset on which to compute LIME scores.
  - `requirements.txt` — Python, PyTorch, Transformers, LoRA, LIME and related dependencies.  

