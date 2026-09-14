# Training script for the retinal captioner (Encoder-Decoder LSTM).
#
# Uses APTOS 2019 images paired with grade-appropriate template captions.
# WeightedRandomSampler boosts grades 3 and 4 which are heavily underrepresented.
# Early stopping with patience=3 — 15 epochs was more than enough in practice.
#
# Run from project root (build vocab first with training/build_vocab.py):
#   conda activate retinal_xai
#   python training/train_captioner.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from training.build_vocab import Vocabulary
from training.dataset_caption import RetinalCaptionDataset, collate_fn
from ml.retinal_caption_model import EncoderCNN, DecoderRNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def train():
    img_dir         = "data/aptos2019/train_images"
    csv_path        = "data/aptos2019/train.csv"
    vocab_path      = "models/weights/vocab.pkl"
    model_save_path = "models/weights/retinal_captioner.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    embed_size    = 256
    hidden_size   = 512
    num_layers    = 1
    num_epochs    = 15
    batch_size    = 16
    learning_rate = 0.001
    patience      = 3

    print("Loading vocabulary...")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Setting up data loaders (80/20 split, class-balanced sampler)...")
    full_dataset = RetinalCaptionDataset(csv_path, img_dir, vocab, transform)
    train_size   = int(0.8 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Grades 3 and 4 are rare — boost their sampling weight 3× to reduce false negatives
    df           = pd.read_csv(csv_path)
    train_labels = [df.iloc[i]["diagnosis"] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=5).astype(float)
    print(f"Training class distribution: {class_counts.astype(int)}")

    class_weights       = 1.0 / (class_counts + 1e-6)
    class_weights[3]   *= 3.0
    class_weights[4]   *= 3.0
    sample_weights      = [class_weights[lbl] for lbl in train_labels]
    sampler             = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_fn, drop_last=True)

    print("Building models...")
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

    # ignore_index=0 skips <pad> tokens in the loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    params    = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    print(f"Adam lr={learning_rate}, ReduceLROnPlateau scheduler")

    best_val_loss   = float("inf")
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        train_loss = 0.0

        for i, (images, captions, lengths) in enumerate(train_loader):
            images   = images.to(device)
            captions = captions.to(device)

            outputs = decoder(encoder(images), captions)
            loss    = criterion(outputs.view(-1, len(vocab)), captions.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions, lengths in val_loader:
                images   = images.to(device)
                captions = captions.to(device)
                outputs  = decoder(encoder(images), captions)
                val_loss += criterion(outputs.view(-1, len(vocab)), captions.view(-1)).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] — Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"LR reduced: {old_lr:.6f} → {new_lr:.6f}")

        if avg_val < best_val_loss:
            print(f"Val loss improved ({best_val_loss:.4f} → {avg_val:.4f}). Saving...")
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "embed_size":  embed_size,
                "hidden_size": hidden_size,
                "vocab_size":  len(vocab),
            }, model_save_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print(f"Done. Best weights saved to {model_save_path}")


if __name__ == "__main__":
    train()
