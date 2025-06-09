import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
from tqdm import tqdm

def evaluate_model_inference_pre_FT(
    model,
    tokenizer,
    dataset_texts,
    dataset_labels,
    device=None,
    print_every=0,       # default no printing during evaluation
    plot_every=100
):
    """
    Run inference on dataset texts and labels, collect classification metrics over time.
    
    Args:
        model: PyTorch model to evaluate.
        tokenizer: tokenizer to preprocess text data.
        dataset_texts: list or Series of texts.
        dataset_labels: list or array of true labels.
        device: torch.device, optional (if None, auto-detect).
        print_every: int, interval of steps to print metrics. If 0, no printing.
        plot_every: int, interval of steps to save metrics for plotting.

    Returns:
        dict with keys: 
            'accuracy_list', 'f1_list', 'precision_list', 'recall_list', 'auc_roc_list', 
            'true_labels_list', 'predictions_list', 'probabilities_list'
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    model.to(device)
    model.eval()

    predictions_list = []
    true_labels_list = []
    probabilities_list = []

    accuracy_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    auc_roc_list = []

    print("-------------------------------------------------------------")
    print("\033[1mStarting evaluation of the model on the test dataset...\033[0m")
    print("-------------------------------------------------------------")

    for i, (text, true_label) in enumerate(tqdm(zip(dataset_texts, dataset_labels), total=len(dataset_texts), desc="Evaluating")):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        prob_pos = probs[0, 1].item()  # Assuming binary classification
        probabilities_list.append(prob_pos)

        prediction = torch.argmax(logits, dim=-1).item()
        predictions_list.append(prediction)
        true_labels_list.append(true_label)

        if (i + 1) % plot_every == 0:
            accuracy = accuracy_score(true_labels_list, predictions_list)
            f1 = f1_score(true_labels_list, predictions_list)
            precision = precision_score(true_labels_list, predictions_list)
            recall = recall_score(true_labels_list, predictions_list)
            try:
                auc_roc = roc_auc_score(true_labels_list, probabilities_list)
            except ValueError:
                auc_roc = np.nan
            
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            auc_roc_list.append(auc_roc)

        if print_every > 0 and (i + 1) % print_every == 0:
            print("\033[1m" + f"Step {i + 1} Metrics:" + "\033[0m")
            print(f" - Accuracy:  {accuracy:.4f}")
            print(f" - F1 Score:  {f1:.4f}")
            print(f" - Precision: {precision:.4f}")
            print(f" - Recall:    {recall:.4f}")
            if not np.isnan(auc_roc):
                print(f" - AUC-ROC:   {auc_roc:.4f}")
            else:
                print(" - AUC-ROC:   Not defined at this step")
            print("-------------------------------------------------------------")

    # Final metrics
    accuracy = accuracy_score(true_labels_list, predictions_list)
    f1 = f1_score(true_labels_list, predictions_list)
    precision = precision_score(true_labels_list, predictions_list)
    recall = recall_score(true_labels_list, predictions_list)
    auc_roc = roc_auc_score(true_labels_list, probabilities_list)

    print("\n\033[1mFinal evaluation metrics on the test dataset:\033[0m")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")

    results = {
        'accuracy_list': accuracy_list,
        'f1_list': f1_list,
        'precision_list': precision_list,
        'recall_list': recall_list,
        'auc_roc_list': auc_roc_list,
        'true_labels_list': true_labels_list,
        'predictions_list': predictions_list,
        'probabilities_list': probabilities_list
    }

    return results

def evaluate_model_inference_post_FT(
    model,
    tokenizer,
    dataset_texts,
    dataset_labels,
    device=None,
    print_every=0,    # default no intermediate printing
    plot_every=100
):
    """
    Run inference on dataset texts and labels, collect classification metrics over time.
    
    Args:
        model: PyTorch model to evaluate.
        tokenizer: tokenizer to preprocess text data.
        dataset_texts: list or Series of texts.
        dataset_labels: list or array of true labels.
        device: torch.device, optional (if None, auto-detect).
        print_every: int, interval of steps to print metrics. If 0, no printing.
        plot_every: int, interval of steps to save metrics for plotting.

    Returns:
        dict with keys: 
            'accuracy_list', 'f1_list', 'precision_list', 'recall_list', 'auc_roc_list', 
            'true_labels_list', 'predictions_list', 'probabilities_list'
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    model.to(device)
    model.eval()

    predictions_list = []
    true_labels_list = []
    probabilities_list = []

    accuracy_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    auc_roc_list = []

    print("-------------------------------------------------------------")
    print("\033[1mStarting evaluation of the model on the test dataset after LoRA fine-tuning...\033[0m")
    print("-------------------------------------------------------------")

    for i, (text, true_label) in enumerate(tqdm(zip(dataset_texts, dataset_labels), total=len(dataset_texts), desc="Evaluating")):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        prob_pos = probs[0, 1].item()  # Assuming binary classification
        probabilities_list.append(prob_pos)

        prediction = torch.argmax(logits, dim=-1).item()
        predictions_list.append(prediction)
        true_labels_list.append(true_label)

        if (i + 1) % plot_every == 0:
            accuracy = accuracy_score(true_labels_list, predictions_list)
            f1 = f1_score(true_labels_list, predictions_list)
            precision = precision_score(true_labels_list, predictions_list)
            recall = recall_score(true_labels_list, predictions_list)
            try:
                auc_roc = roc_auc_score(true_labels_list, probabilities_list)
            except ValueError:
                auc_roc = np.nan
            
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
            auc_roc_list.append(auc_roc)

        if print_every > 0 and (i + 1) % print_every == 0:
            print("\033[1m" + f"Step {i + 1} Metrics:" + "\033[0m")
            print(f" - Accuracy:  {accuracy:.4f}")
            print(f" - F1 Score:  {f1:.4f}")
            print(f" - Precision: {precision:.4f}")
            print(f" - Recall:    {recall:.4f}")
            if not np.isnan(auc_roc):
                print(f" - AUC-ROC:   {auc_roc:.4f}")
            else:
                print(" - AUC-ROC:   Not defined at this step")
            print("-------------------------------------------------------------")

    # Final metrics
    accuracy = accuracy_score(true_labels_list, predictions_list)
    f1 = f1_score(true_labels_list, predictions_list)
    precision = precision_score(true_labels_list, predictions_list)
    recall = recall_score(true_labels_list, predictions_list)
    auc_roc = roc_auc_score(true_labels_list, probabilities_list)

    print("\n\033[1mFinal evaluation metrics on the test dataset:\033[0m")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")

    results = {
        'accuracy_list': accuracy_list,
        'f1_list': f1_list,
        'precision_list': precision_list,
        'recall_list': recall_list,
        'auc_roc_list': auc_roc_list,
        'true_labels_list': true_labels_list,
        'predictions_list': predictions_list,
        'probabilities_list': probabilities_list
    }

    return results
