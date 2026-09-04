"""
ml/retinal_caption_model.py
Defines the Vision-Language Encoder-Decoder architecture for retinal image captioning.
This file provides the classes required to train the captioning module.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int):
        """
        ResNet-50 encoder.
        Extracts visual features from the fundus image and maps them to embed_size.
        """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Linear layer to map the ResNet output (2048) to the desired embedding size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # features shape: (batch_size, 2048, 1, 1)
        features = self.resnet(images)
        # reshape to (batch_size, 2048)
        features = features.view(features.size(0), -1)
        # map to (batch_size, embed_size)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        """
        LSTM decoder.
        Generates the caption sequence token by token.
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Args:
            features: (batch_size, embed_size)
            captions: (batch_size, seq_length)
        """
        # Embed the captions, drop the last token (usually <end>) to match sequence length
        embeddings = self.embed(captions[:, :-1])
        # Concatenate features as the first input to the LSTM
        # features shape: (batch_size, 1, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # Pass through LSTM
        hiddens, _ = self.lstm(embeddings)
        
        # Map to vocabulary size
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, max_len=20, start_idx=0):
        """
        Generate caption greedily for inference.
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None
        
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            
            # Prepare next input
            inputs = self.embed(predicted).unsqueeze(1)
            
        return sampled_ids


class RetinalCaptioner(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        """
        Combined Vision-Language model.
        """
        super(RetinalCaptioner, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
