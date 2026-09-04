import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import pickle
import pandas as pd
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Add special tokens
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
        
    def __len__(self):
        return len(self.word2idx)

# Caption templates mapping DR grade to clinical description
CAPTION_TEMPLATES = {
    0: "Fundus image appears normal. Optic disc and macula are grossly preserved. No obvious microaneurysms, haemorrhages, or hard exudates are visible. Vascular caliber is within normal limits.",
    1: "Fundus image shows mild vascular changes. Scattered microaneurysms are present. No obvious haemorrhages or hard exudates. Optic disc appears normal.",
    2: "Fundus image shows moderate vascular abnormalities. Microaneurysms and dot-blot haemorrhages are visible. Mild vascular tortuosity is present. Hard exudates are beginning to form.",
    3: "Fundus image reveals severe vascular damage. Numerous haemorrhages and hard exudates are present throughout the retina. Significant vascular tortuosity and venous beading are visible.",
    4: "Fundus image shows proliferative changes. Neovascularisation is evident at the disc or elsewhere. Pre-retinal haemorrhage and severe exudation are present, indicating high risk."
}

def tokenize(text):
    """Simple tokenization: lowercase and separate punctuation."""
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"\s+", r" ", text)
    return text.strip().split()

def build_vocab(csv_path, vocab_path):
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    vocab = Vocabulary()
    
    print("Building vocabulary from templated captions...")
    for template in CAPTION_TEMPLATES.values():
        tokens = tokenize(template)
        for token in tokens:
            vocab.add_word(token)
            
    print(f"Total vocabulary size: {len(vocab)}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"Saved vocabulary to {vocab_path}")

if __name__ == '__main__':
    # Define paths relative to the project root
    csv_path = 'data/aptos2019/train.csv'
    vocab_path = 'models/weights/vocab.pkl'
    
    build_vocab(csv_path, vocab_path)
