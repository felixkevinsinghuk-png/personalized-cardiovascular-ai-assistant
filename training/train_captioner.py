import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import torch
import torch.nn as nn
from torchvision import transforms
from training.build_vocab import Vocabulary
from training.dataset_caption import RetinalCaptionDataset, collate_fn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from ml.retinal_caption_model import EncoderCNN, DecoderRNN
import pandas as pd
import numpy as np

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def train():
    # Directories and Paths
    img_dir = 'data/aptos2019/train_images'
    csv_path = 'data/aptos2019/train.csv'
    vocab_path = 'models/weights/vocab.pkl'
    model_save_path = 'models/weights/retinal_captioner.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_epochs = 15 # Increased max epochs since we now use Early Stopping
    batch_size = 16
    learning_rate = 0.001
    patience = 3 # Stop if val_loss doesn't improve for 3 epochs
    
    # Load vocabulary
    print("Loading vocabulary...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and Validation Split
    print("Initializing Data Loaders with 80/20 split and class balancing...")
    full_dataset = RetinalCaptionDataset(csv_path, img_dir, vocab, transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    # --- Class Balancing via WeightedRandomSampler ---
    # Count samples per grade in the training split
    df = pd.read_csv(csv_path)
    train_indices = train_dataset.indices
    train_labels = [df.iloc[i]['diagnosis'] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=5).astype(float)
    print(f"Training class distribution: {class_counts.astype(int)}")

    # Inverse frequency weights — rare classes get higher weight
    class_weights = 1.0 / (class_counts + 1e-6)
    # Extra boost for severe grades (3 and 4) to reduce false negatives
    class_weights[3] *= 3.0
    class_weights[4] *= 3.0
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # Class-weighted CrossEntropyLoss — penalise errors on severe grades more heavily
    # Weight is applied at the token/word level
    loss_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0,  # special tokens + grade0/1
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0, 1.0, 1.0]).to(device)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              sampler=sampler,  # use weighted sampler instead of shuffle
                              num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=True)
                             
    # Build models
    print("Building models...")
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)
    
    # Loss and optimizer — standard CrossEntropyLoss (word-level weighting not applicable)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad> token in loss calculation
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    print(f"Optimizer: Adam lr={learning_rate} | LR Scheduler: ReduceLROnPlateau")
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Train the model
    print("Starting training...")
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        train_loss = 0.0
        
        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            
            features = encoder(images)
            outputs = decoder(features, captions)
            
            targets = captions
            loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
            
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Train Loss: {loss.item():.4f}")
                
        # Validation Phase
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions, lengths in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                
                features = encoder(images)
                outputs = decoder(features, captions)
                targets = captions
                
                loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        print(f"--- Epoch [{epoch+1}/{num_epochs}] Summary ---")
        print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        
        # Step the LR scheduler based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'embed_size': embed_size,
                'hidden_size': hidden_size,
                'vocab_size': len(vocab)
            }, model_save_path)
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered! Training stopped to prevent overfitting.")
                break

    print(f"Training complete. Best weights saved to {model_save_path}")

if __name__ == '__main__':
    train()
