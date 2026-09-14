# Quick model evaluation script — prints AUC, F1, Sensitivity and Specificity
# for both ResNet-50 (DR) and EfficientNet-B4 (CVD) on their respective test splits.
#
#   conda activate retinal_xai
#   python evaluate_models.py

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import Config
from training.dataset_aptos import APTOSDataset, get_aptos_transforms
from training.dataset_odir import ODIRDataset, get_odir_transforms
from models.model_loader import load_models

SEED = 42

def get_test_loader_aptos():
    csv_path = os.path.join(Config.APTOS_DATA_DIR, "train.csv")
    images_dir = os.path.join(Config.APTOS_DATA_DIR, "train_images")
    full_dataset = APTOSDataset(csv_path, images_dir, transform=None)
    n = len(full_dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val
    
    _, _, test_data = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    test_data.dataset.transform = get_aptos_transforms(train=False)
    return DataLoader(test_data, batch_size=Config.BATCH_SIZE, shuffle=False)

def get_test_loader_odir():
    csv_path = os.path.join(Config.ODIR_DATA_DIR, "full_df.csv")
    images_dir = os.path.join(Config.ODIR_DATA_DIR, "preprocessed_images")
    full_dataset = ODIRDataset(csv_path, images_dir, transform=None)
    n = len(full_dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val
    
    _, _, test_data = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    test_data.dataset.transform = get_odir_transforms(train=False)
    return DataLoader(test_data, batch_size=Config.BATCH_SIZE, shuffle=False)

def evaluate_model(model, loader, device, is_binary=False):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            
            if is_binary:
                # EfficientNet output shape is usually [B, 1]
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.extend(probs)
                all_preds.extend(preds)
                
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def print_metrics(name, labels, preds, probs, is_binary):
    print(f"\n--- {name} ---")

    try:
        auc = roc_auc_score(labels, probs) if is_binary else roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        print(f"AUC-ROC: {auc:.4f}")
    except Exception as e:
        print(f"AUC Error: {e}")

    print(f"F1-Score (Macro): {f1_score(labels, preds, average='macro'):.4f}")
    print(f"Sensitivity (Recall): {recall_score(labels, preds, average='macro'):.4f}")

    cm = confusion_matrix(labels, preds)
    if is_binary:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
    else:
        # Macro specificity via one-vs-rest from the confusion matrix
        specificities = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        specificity = np.mean(specificities)

    print(f"Specificity: {specificity:.4f}")

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print("Loading models (this takes a few seconds)...")
    resnet, effnet, _, _, device = load_models()
    
    print("\nLoading test datasets...")
    aptos_loader = get_test_loader_aptos()
    odir_loader = get_test_loader_odir()
    
    print("\nEvaluating ResNet-50 (Diabetic Retinopathy)...")
    r_labels, r_preds, r_probs = evaluate_model(resnet, aptos_loader, device, is_binary=False)
    print_metrics("ResNet-50", r_labels, r_preds, r_probs, is_binary=False)
    
    print("\nEvaluating EfficientNet-B4 (CVD Risk)...")
    e_labels, e_preds, e_probs = evaluate_model(effnet, odir_loader, device, is_binary=True)
    print_metrics("EfficientNet-B4", e_labels, e_preds, e_probs, is_binary=True)

if __name__ == "__main__":
    main()
