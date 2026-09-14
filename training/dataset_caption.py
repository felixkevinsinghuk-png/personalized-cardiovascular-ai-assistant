# Caption dataset — pairs each APTOS image with its grade-appropriate template caption.
# Used only during captioner training, not during app inference.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from training.build_vocab import CAPTION_TEMPLATES, tokenize


class RetinalCaptionDataset(Dataset):
    """Maps each APTOS image to its template caption as a token ID sequence."""

    def __init__(self, csv_path, img_dir, vocab, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.vocab     = vocab
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        diagnosis = row["diagnosis"]

        image = Image.open(os.path.join(self.img_dir, f"{row['id_code']}.png")).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert the template caption to a list of vocab IDs
        tokens  = tokenize(CAPTION_TEMPLATES[diagnosis])
        caption = [self.vocab("<start>")] + [self.vocab(t) for t in tokens] + [self.vocab("<end>")]

        return image, torch.Tensor(caption)


def collate_fn(data):
    """Pad captions in a batch to the same length so they can be stacked."""
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images  = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        targets[i, :lengths[i]] = cap[:lengths[i]]

    return images, targets, lengths


def get_loader(csv_path, img_dir, vocab, transform, batch_size, shuffle, num_workers):
    dataset = RetinalCaptionDataset(csv_path, img_dir, vocab, transform)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn)
