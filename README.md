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

### 📊 AirBase

Maintained by the European Environment Agency (EEA), [AirBase](https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d?activeAccordion=1087599) is our primary data source. It compiles air quality measurements from EU Member States, EEA countries, and partner nations. The dataset includes a multiyear time series of pollutant levels, along with metadata on monitoring networks and stations.

🧪 Scripts & Notebooks:
- `data/download_eea_air_quality_data.py` — Python script to download AirBase data.
- `analysis/eea_air_quality_data_eda.ipynb` — Initial exploratory data analysis of the dataset.
- `preprocess/*` — Scripts used to aggregate and clean the data into the final file `air_quality_data.json`, which is consumed by the web app. 

🔗 Useful Links:
- 📄 [Official Datasheet](https://www.eea.europa.eu/data-and-maps/data/airbase-the-european-air-quality-database-6/airbase-products/data/file)
- 🐍 [Python Downloader](https://github.com/JohnPaton/airbase)

### 🌍 Global EV Outlook 2025

The Global EV Outlook [1] is an annual report that presents key trends and developments in electric mobility worldwide. It is developed with the support of the Electric Vehicles Initiative (EVI).

For further insights, refer to the dedicated article by Our World in Data [2].

🧪 Scripts & Notebooks:
- `preprocess/build_ev_share_json.ipynb` — Notebook used to clean and process the data into the file electric_car_share_data.json, which is consumed by the web app.

📚 Citations:
- [1] IEA (2025), Global EV Outlook 2025, IEA, Paris. https://www.iea.org/reports/global-ev-outlook-2025 — Licence: CC BY 4.0
- [2] Hannah Ritchie (2024), Tracking Global Data on Electric Vehicles. Published online at OurWorldinData.org. Retrieved from: https://ourworldindata.org/electric-car-sales

## 📦 Conda Env

To run the provided Python scripts and notebooks, use the `ee-559` conda environment.
You can create it by running the following command:

```bash
conda env create -f ee-559.yml
```

Once created, activate the environment with:

```bash
conda activate ee-559
```

## 🧱 Project Structure

```text
ee-559-deep-learning/
│
├── checkpoints/                       # Trained LoRA checkpoints by model & dataset
│   ├── TinyLlama/
│   │   ├── dynahate/                  # 3 LoRA-tuned checkpoints (3 seeds) on DynaHate
│   │   └── hatecheck/                 # 3 LoRA-tuned checkpoints (3 seeds) on HateCheck
│   ├── Phi-2/
│   │   ├── dynahate/
│   │   └── hatecheck/
│   └── opt/
│       ├── dynahate/
│       └── hatecheck/
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
│   ├── dynahate/                      # CSVs of accuracy, F1-score, etc.
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
├── evaluate_models.py                 
│
└── compute_lime_scores.py             

```

## 🗂️ Folder Highlights

- `checkpoints/` — LoRA‐fine-tuned model weights, organized by TinyLlama, Phi-2 and OPT-1.3B, each with DynaHate and HateCheck seeds.

Because the `checkpoints.tar.gz` file is split into parts to comply with the 2 GB Git LFS limit, you can recombine them into a single archive by running:

```bash
cat checkpoints.tar.gz.part-* > checkpoints.tar.gz
```
Once the parts are merged, extract the contents with:
```
tar -xzvf checkpoints.tar.gz
```

- `data/` — Raw and preprocessed DynaHate & HateCheck datasets ready for training and evaluation.  
- `experiments/` — YAML config files specifying LoRA & training hyperparameters for each model.  
- `results/` —  
  - `metrics/`: CSVs with accuracy, F1-score, etc.  
  - `lime/`: Token‐level attribution scores (pre- and post-fine-tuning).  
- `utils/` — Helper modules for text dataset creation, visualization, and common utilities.  
- **Root‐level scripts**:  
  - `run_training_dynahate.sh` & `run_training_hatecheck.sh`: launch dataset‐specific training.  
  - `trainer_dynahate.py` & `trainer_hatecheck.py`: PyTorch Lightning training pipelines.  
  - `evaluate_models.py`: evaluate best checkpoints on DynaHate or HateCheck.  
  - `compute_lime_scores.py`: generate LIME explanations for selected models.  
- `requirements.txt` — Python, PyTorch, Transformers, LoRA, LIME and related dependencies.  

