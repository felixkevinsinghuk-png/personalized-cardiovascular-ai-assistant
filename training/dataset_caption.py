import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from training.build_vocab import Vocabulary, CAPTION_TEMPLATES, tokenize
import pickle

class RetinalCaptionDataset(Dataset):
    def __init__(self, csv_path, img_dir, vocab, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id_code']
        diagnosis = row['diagnosis']
        
        # Load image
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Get caption template based on diagnosis
        caption_text = CAPTION_TEMPLATES[diagnosis]
        tokens = tokenize(caption_text)
        
        # Convert words to vocabulary indices
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        
        target = torch.Tensor(caption)
        
        return image, target

def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).
    We should pad the captions to the maximum length in the batch.
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    # Merge images (from tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    
    # Merge captions (from tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        
    return images, targets, lengths

def get_loader(csv_path, img_dir, vocab, transform, batch_size, shuffle, num_workers):
    dataset = RetinalCaptionDataset(csv_path, img_dir, vocab, transform)
    
    data_loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return data_loader
