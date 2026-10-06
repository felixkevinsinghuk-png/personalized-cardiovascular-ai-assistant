# Builds and pickles the vocabulary for the retinal captioner.
#
# The vocabulary is built purely from the five clinical caption templates below —
# not from free-text, so the vocab is small (~63 words) and deterministic.
# Run this once before training the captioner.
#
#   python training/build_vocab.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import pickle


class Vocabulary:
    """Bidirectional word <-> index mapping with four special tokens."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        for tok in ["<pad>", "<start>", "<end>", "<unk>"]:
            self.add_word(tok)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def __len__(self):
        return len(self.word2idx)


# Ground-truth captions aligned to each DR grade.
# These are the "labels" for the captioner — the LSTM learns to reproduce them.
CAPTION_TEMPLATES = {
    0: "Fundus image appears normal. Optic disc and macula are grossly preserved. No obvious microaneurysms, haemorrhages, or hard exudates are visible. Vascular caliber is within normal limits.",
    1: "Fundus image shows mild vascular changes. Scattered microaneurysms are present. No obvious haemorrhages or hard exudates. Optic disc appears normal.",
    2: "Fundus image shows moderate vascular abnormalities. Microaneurysms and dot-blot haemorrhages are visible. Mild vascular tortuosity is present. Hard exudates are beginning to form.",
    3: "Fundus image reveals severe vascular damage. Numerous haemorrhages and hard exudates are present throughout the retina. Significant vascular tortuosity and venous beading are visible.",
    4: "Fundus image shows proliferative changes. Neovascularisation is evident at the disc or elsewhere. Pre-retinal haemorrhage and severe exudation are present, indicating high risk.",
}


def tokenize(text: str) -> list:
    """Lowercase and split on whitespace, keeping punctuation as separate tokens."""
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()


def build_vocab(csv_path: str, vocab_path: str):
    vocab = Vocabulary()
    print("Building vocabulary from caption templates...")
    for template in CAPTION_TEMPLATES.values():
        for token in tokenize(template):
            vocab.add_word(token)
    print(f"Vocabulary size: {len(vocab)} words")

    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved to {vocab_path}")


if __name__ == "__main__":
    build_vocab(
        csv_path="data/aptos2019/train.csv",
        vocab_path="models/weights/vocab.pkl",
    )
