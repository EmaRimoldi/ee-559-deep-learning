# import torch
# import argparse
# import yaml
# import wandb
# import random
# import pickle
# import numpy as np
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     DataCollatorWithPadding,
#     TrainingArguments,
#     Trainer
# )
# import evaluate
# from datasets import load_dataset
# from utils.datasets import (
#     create_dynahate_dataset,
# )
# from utils.experiments import downsample
# from utils.inference import evaluate_model_inference_post_FT
# from utils.inference import evaluate_model_inference_pre_FT


# # Suppress warnings for a cleaner output
# import warnings
# warnings.filterwarnings("ignore")

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # Set up argument parser to accept the config file as a command-line argument
# parser = argparse.ArgumentParser(description="Train a model with a specified config file.")
# parser.add_argument('config_file', type=str, help="Path to the config YAML file")
# args = parser.parse_args()

# # Load configuration from YAML file specified by the user
# print("\033[1m" + "-"*80 + "\nLoading configuration...\n" + "-"*80 + "\033[0m")  # Bold print and alignment
# with open(args.config_file, "r") as f:
#     config = yaml.safe_load(f)

# # Set model checkpoint from config file
# model_checkpoint = config['model_checkpoint']

# # Initialize WandB with a project name dynamically set from the model checkpoint
# wandb.init(project=f"model-training-{config['model_name']}", entity="emanuelerimoldi7-epfl", config=config)

# # Set random seed for reproducibility
# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# # Call the function to set the seed
# set_seed()

# # Define label maps for classification
# id2label = {
#     0: "nothate",
#     1: "hate"
# }
# label2id = {
#     "nothate": 0,
#     "hate": 1
# }

# # Load the model with classification head
# print("\033[1m" + "-"*80 + "\nLoading model...\n" + "-"*80 + "\033[0m")  
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_checkpoint,
#     num_labels=2,
#     id2label=id2label,
#     label2id=label2id,
# )


# import os
# import torch
# import multiprocessing

# # Check for CUDA and print device details if available
# if torch.cuda.is_available():
#     device = torch.device("cuda")
    
#     # Print CUDA device name and memory details
#     print(f"\033[1m" + "-"*80 + f"\nCUDA is available! Using device: {torch.cuda.get_device_name(device)}\n" + "-"*80 + "\033[0m")
#     allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # in GB
#     print(f"Memory Allocated: {allocated_memory:.2f} GB")
#     cached_memory = torch.cuda.memory_reserved(device) / 1024**3  # in GB
#     print(f"Memory Cached: {cached_memory:.2f} GB")
#     total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # in GB
#     free_memory = total_memory - allocated_memory - cached_memory  # Available memory in GB
#     print(f"Available Memory: {free_memory:.2f} GB")
#     cpu_cores = multiprocessing.cpu_count()
#     print(f"Number of CPU cores available: {cpu_cores}")
#     dataloader_num_workers = min(cpu_cores // 2, 20) 
#     print(f"Using {dataloader_num_workers} workers for data loading\n" + "-"*80 + "\033[0m")


# else:
#     device = torch.device("cpu")
#     print("\033[1m" + "-"*80 + "\nCUDA is not available. Using CPU.\n" + "-"*80 + "\033[0m")


# # Load the tokenizer from the pre-trained model
# print("\033[1m" + "-"*80 + "\nLoading tokenizer...\n" + "-"*80 + "\033[0m")  
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# # Set padding token as eos_token if it's not defined
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # Set pad_token_id in the model
# model.config.pad_token_id = tokenizer.pad_token_id

# # Resize model embeddings to match the tokenizer size
# model.resize_token_embeddings(len(tokenizer))

# # Load the DynaHate dataset
# print("\033[1m" + "-"*80 + "\nLoading dataset...\n" + "-"*80 + "\033[0m")  
# dynahate_dataset = create_dynahate_dataset("data")

# # Tokenization function for the dataset
# def tokenize_function(examples):
#     return tokenizer(
#         examples["text"],  
#         padding="max_length", 
#         truncation=True,
#         max_length=512, 
#         return_tensors="pt"
#     )

# # Tokenizing the dataset
# tokenized_dataset = dynahate_dataset.map(tokenize_function, batched=True)

# # Prepare DataCollator for padding
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# # Load metrics using 'evaluate' library
# accuracy = evaluate.load("accuracy")
# precision = evaluate.load("precision")
# recall = evaluate.load("recall")
# f1 = evaluate.load("f1")

# # Function to compute evaluation metrics
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = torch.argmax(torch.tensor(predictions), dim=-1)  # Convert logits to labels
#     accuracy_score = accuracy.compute(predictions=predictions, references=labels)
#     precision_score = precision.compute(predictions=predictions, references=labels)
#     recall_score = recall.compute(predictions=predictions, references=labels)
#     f1_score = f1.compute(predictions=predictions, references=labels)
#     return {
#         'accuracy': accuracy_score['accuracy'],
#         'precision': precision_score['precision'],
#         'recall': recall_score['recall'],
#         'f1': f1_score['f1']
#     }
# # # --- Add testing pre-finetuning ---
# # print("\033[1m" + "-"*80 + "\nRunning pre-finetuning evaluation...\n" + "-"*80 + "\033[0m")  

# # results_pre_FT = evaluate_model_inference_pre_FT(
# #     model=model,
# #     tokenizer=tokenizer,
# #     dataset_texts=dynahate_dataset['test']['text'],
# #     dataset_labels=dynahate_dataset['test']['label'],
# #     plot_every=100
# # )

# # # Save the results for pre-finetuning
# # print("\033[1m" + "-"*80 + "\nSaving pre-finetuning results...\n" + "-"*80 + "\033[0m") 

# # with open(f"./experiments/{config['model_name']}/results_pre_FT.pkl", "wb") as f:
# #     pickle.dump(results_pre_FT, f)

# # print("\033[1m" + "-"*80 + "\nPre-finetuning results saved successfully!\n" + "-"*80 + "\033[0m")


# # Training arguments
# training_args = TrainingArguments(
#     output_dir=config['output_dir'],             # Directory to save checkpoints
#     learning_rate=config['learning_rate'],       # Learning rate
#     per_device_train_batch_size=8,               # Training batch size
#     per_device_eval_batch_size=8,                # Evaluation batch size
#     num_train_epochs=config['num_train_epochs'], # Number of epochs
#     weight_decay=config['weight_decay'],         # Weight decay
#     eval_strategy="steps",                       # Evaluate after every epoch
#     save_strategy="steps",                       # Save model every X steps
#     save_steps=500,                              # Save model every 500 steps
#     load_best_model_at_end=True,                 # Load best model after training
#     logging_dir=config['logging_dir'],           # Logging directory
#     logging_steps=config['logging_steps'],       # Logging frequency
#     report_to="wandb",                           # Use WandB for logging
#     #fp16=True,                                   # Enable FP16 training
#     gradient_accumulation_steps=8,               # Gradient accumulation steps
#     dataloader_num_workers=20,                   # Number of workers for data loading
#     lr_scheduler_type="cosine",                  # Linear LR scheduler
#     warmup_ratio=0.03,                           # Warmup steps as a ratio of total training steps                         
# )



# # Trainer setup
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["validation"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics
# )

# # Start training the model
# print("\033[1m" + "-"*80 + "\nStarting training...\n" + "-"*80 + "\033[0m")  
# trainer.train()

# # Save the best model after training
# print("\033[1m" + "-"*80 + "\nSaving the best model...\n" + "-"*80 + "\033[0m")  
# best_model_path = f"./experiments/{config['model_name']}/best_checkpoint"

# # Print the path where the model is being saved
# print("\033[1m" + "-"*80 + f"\nModel saved at: {best_model_path}\n" + "-"*80 + "\033[0m")

# trainer.save_model(best_model_path)

# # Try loading the best model and check if it was loaded correctly
# print("\033[1m" + "-"*80 + "\nLoading the best saved model...\n" + "-"*80 + "\033[0m")  
# try:
#     best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
#     print("\033[1m" + "-"*80 + "\nBest model loaded successfully!\n" + "-"*80 + "\033[0m") 
# except Exception as e:
#     print("\033[1m" + "-"*80 + f"\nFailed to load the best model. Error: {e}\n" + "-"*80 + "\033[0m")  

# # --- Add testing ---
# print("\033[1m" + "-"*80 + "\nRunning post-finetuning evaluation...\n" + "-"*80 + "\033[0m")  
# results_post_FT = evaluate_model_inference_post_FT(
#     model=model,
#     tokenizer=tokenizer,
#     dataset_texts=dynahate_dataset['test']['text'],
#     dataset_labels=dynahate_dataset['test']['label'],
#     plot_every=100
# )

# # Save the results
# print("\033[1m" + "-"*80 + "\nSaving results...\n" + "-"*80 + "\033[0m")  
# with open(f"./experiments/{config['model_name']}/results_post_FT.pkl", "wb") as f:
#     pickle.dump(results_post_FT, f)
# print("\033[1m" + "-"*80 + "\nResults saved successfully!\n" + "-"*80 + "\033[0m")  

##################################### NEW VERSION #############################################

import torch
import argparse
import yaml
import wandb
import random
import pickle
import os
import multiprocessing
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import evaluate
from datasets import load_dataset
from utils.datasets import (
    create_hatecheck_dataset,
)
from utils.experiments import downsample
from utils.inference import evaluate_model_inference_post_FT
from utils.inference import evaluate_model_inference_pre_FT
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig


# Suppress warnings for a cleaner output
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Set up argument parser to accept the config file as a command-line argument
parser = argparse.ArgumentParser(description="Train a model with a specified config file.")
parser.add_argument('config_file', type=str, help="Path to the config YAML file")
args = parser.parse_args()

# Load configuration from YAML file specified by the user
print("\033[1m" + "-"*80 + "\nLoading configuration...\n" + "-"*80 + "\033[0m")  # Bold print and alignment
with open(args.config_file, "r") as f:
    config = yaml.safe_load(f)

# Set model checkpoint from config file
model_checkpoint = config['model_checkpoint']

# Initialize WandB with a project name dynamically set from the model checkpoint
wandb.init(project=f"model-training-{config['model_name']}", entity="emanuelerimoldi7-epfl", config=config)

# Set random seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Call the function to set the seed
set_seed(123)

# Define label maps for classification
id2label = {
    0: "nothate",
    1: "hate"
}
label2id = {
    "nothate": 0,
    "hate": 1
}


# Load the model with classification head
print("\033[1m" + "-"*80 + "\nLoading model...\n" + "-"*80 + "\033[0m")  
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# Setting up LoRA configuration
print("\033[1m" + "-"*80 + "\nSetting up LoRA configuration...\n" + "-"*80 + "\033[0m")
peft_config = LoraConfig(
    task_type=config['task_type'],           # Task type for sequence classification
    r=config['r'],                           # Rank of the low-rank matrix
    lora_alpha=config['lora_alpha'],         # Scaling factor for LoRA 
    lora_dropout=config['lora_dropout'],     # Dropout rate for LoRA layers 
    target_modules=config['target_modules'], # Target both k_proj and v_proj modules for LoRA
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Check if LoRA parameters were applied
print("-" * 80)
model.print_trainable_parameters()
print("-" * 80)

# Check for CUDA and print device details if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    
    # Print CUDA device name and memory details
    print(f"\033[1m" + "-"*80 + f"\nCUDA is available! Using device: {torch.cuda.get_device_name(device)}\n" + "-"*80 + "\033[0m")
    allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # in GB
    print(f"Memory Allocated: {allocated_memory:.2f} GB")
    cached_memory = torch.cuda.memory_reserved(device) / 1024**3  # in GB
    print(f"Memory Cached: {cached_memory:.2f} GB")
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # in GB
    free_memory = total_memory - allocated_memory - cached_memory  # Available memory in GB
    print(f"Available Memory: {free_memory:.2f} GB")
    cpu_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {cpu_cores}")
    dataloader_num_workers = min(cpu_cores // 2, 20) 
    print(f"Using {dataloader_num_workers} workers for data loading\n" + "-"*80 + "\033[0m")


else:
    device = torch.device("cpu")
    print("\033[1m" + "-"*80 + "\nCUDA is not available. Using CPU.\n" + "-"*80 + "\033[0m")


# Load the tokenizer from the pre-trained model
print("\033[1m" + "-"*80 + "\nLoading tokenizer...\n" + "-"*80 + "\033[0m")  
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Set padding token as eos_token if it's not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set pad_token_id in the model
model.config.pad_token_id = tokenizer.pad_token_id

# Resize model embeddings to match the tokenizer size
model.resize_token_embeddings(len(tokenizer))

# Load the Hatecheck dataset
print("\033[1m" + "-"*80 + "\nLoading dataset...\n" + "-"*80 + "\033[0m")  
hatecheck_dataset = create_hatecheck_dataset()

# Tokenization function for the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["test_case"],  
        padding="max_length", 
        truncation=True,
        max_length=512, 
        return_tensors="pt"
    )

# Tokenizing the dataset
tokenized_dataset = hatecheck_dataset.map(tokenize_function, batched=True)

# Prepare DataCollator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load metrics using 'evaluate' library
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

# Function to compute evaluation metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=-1)  # Convert logits to labels
    accuracy_score = accuracy.compute(predictions=predictions, references=labels)
    precision_score = precision.compute(predictions=predictions, references=labels)
    recall_score = recall.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels)
    return {
        'accuracy': accuracy_score['accuracy'],
        'precision': precision_score['precision'],
        'recall': recall_score['recall'],
        'f1': f1_score['f1']
    }

# --- Add testing pre-finetuning ---
print("\033[1m" + "-"*80 + "\nRunning pre-finetuning evaluation...\n" + "-"*80 + "\033[0m")  

results_pre_FT = evaluate_model_inference_pre_FT(
    model=model,
    tokenizer=tokenizer,
    dataset_texts=hatecheck_dataset['test']['test_case'],
    dataset_labels=hatecheck_dataset['test']['label'],
    plot_every=100
)

# Save the results for pre-finetuning
print("\033[1m" + "-"*80 + "\nSaving pre-finetuning results...\n" + "-"*80 + "\033[0m") 

with open(f"./experiments/{config['model_name']}/results_pre_FT.pkl", "wb") as f:
    pickle.dump(results_pre_FT, f)

print("\033[1m" + "-"*80 + "\nPre-finetuning results saved successfully!\n" + "-"*80 + "\033[0m")

# Training arguments
import torch
from transformers import TrainingArguments

# Impostare fp16 solo se CUDA è disponibile
training_args = TrainingArguments(
    output_dir=config['output_dir'],             # Directory to save checkpoints
    learning_rate=config['learning_rate'],       # Learning rate
    per_device_train_batch_size=4,               # Training batch size
    per_device_eval_batch_size=4,                # Evaluation batch size
    num_train_epochs=config['num_train_epochs'], # Number of epochs
    weight_decay=config['weight_decay'],         # Weight decay
    eval_strategy="epoch",                       # Evaluate after every epoch
    save_strategy="epoch",                       # Save model every epoch
    load_best_model_at_end=True,                 # Load best model after training
    logging_dir=config['logging_dir'],           # Logging directory
    logging_steps=config['logging_steps'],       # Logging frequency
    report_to="wandb",                           # Use WandB for logging
    fp16=torch.cuda.is_available(),              # Enable FP16 if CUDA is available, otherwise False
    gradient_accumulation_steps=2,               # Gradient accumulation steps
    dataloader_num_workers=20,                   # Number of workers for data loading
    lr_scheduler_type="cosine",                  # Linear LR scheduler
    warmup_ratio=0.03,                           # Warmup steps as a ratio of total training steps                         
)



# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training the model
print("\033[1m" + "-"*80 + "\nStarting training...\n" + "-"*80 + "\033[0m")  
trainer.train()

# Save the best model after training
print("\033[1m" + "-"*80 + "\nSaving the best model...\n" + "-"*80 + "\033[0m")  
best_model_path = f"./experiments/{config['model_name']}/best_checkpoint"

# Print the path where the model is being saved
print("\033[1m" + "-"*80 + f"\nModel saved at: {best_model_path}\n" + "-"*80 + "\033[0m")

trainer.save_model(best_model_path)

# Try loading the best model and check if it was loaded correctly
print("\033[1m" + "-"*80 + "\nLoading the best saved model...\n" + "-"*80 + "\033[0m")  
try:
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    print("\033[1m" + "-"*80 + "\nBest model loaded successfully!\n" + "-"*80 + "\033[0m") 
except Exception as e:
    print("\033[1m" + "-"*80 + f"\nFailed to load the best model. Error: {e}\n" + "-"*80 + "\033[0m")  

# --- Add testing ---
print("\033[1m" + "-"*80 + "\nRunning post-finetuning evaluation...\n" + "-"*80 + "\033[0m")  
results_post_FT = evaluate_model_inference_post_FT(
    model=model,
    tokenizer=tokenizer,
    dataset_texts=hatecheck_dataset['test']['test_case'],
    dataset_labels=hatecheck_dataset['test']['label'],
    plot_every=100
)

# Save the results
print("\033[1m" + "-"*80 + "\nSaving results...\n" + "-"*80 + "\033[0m")  
with open(f"./experiments/{config['model_name']}/results_post_FT.pkl", "wb") as f:
    pickle.dump(results_post_FT, f)
print("\033[1m" + "-"*80 + "\nResults saved successfully!\n" + "-"*80 + "\033[0m")  
