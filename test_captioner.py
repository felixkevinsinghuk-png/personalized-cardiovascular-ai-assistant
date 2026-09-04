import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
from training.build_vocab import Vocabulary  # noqa: needed for pickle
from ml.caption_inference import generate_retinal_caption

# Image transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def test_captioner():
    df = pd.read_csv('data/aptos2019/train.csv')

    print(f"\n{'='*62}")
    print("RETINAL CAPTIONING MODEL — LIVE INFERENCE TEST (with Safety Override)")
    print(f"{'='*62}\n")

    for grade in range(5):
        sample = df[df['diagnosis'] == grade].iloc[0]
        img_path = f"data/aptos2019/train_images/{sample['id_code']}.png"
        image = Image.open(img_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)

        # Call the actual inference function (includes Safety Override)
        caption = generate_retinal_caption(tensor, dr_grade=grade)

        print(f"Grade {grade} | Image: {sample['id_code']}.png")
        print(f"Generated Caption: {caption}")
        print(f"{'─'*62}")

if __name__ == '__main__':
    test_captioner()
