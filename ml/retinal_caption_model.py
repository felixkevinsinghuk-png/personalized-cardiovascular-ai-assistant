# Vision-Language Encoder-Decoder for retinal image captioning.
# EncoderCNN extracts visual features; DecoderRNN generates the caption word by word.

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """ResNet-50 backbone that maps fundus image features to an embedding vector."""

    def __init__(self, embed_size: int):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # Strip the final FC layer — we only want the feature extractor
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn     = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)          # (batch, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 2048)
        return self.bn(self.linear(features))   # (batch, embed_size)


class DecoderRNN(nn.Module):
    """LSTM that generates captions token by token from the encoder's feature vector."""

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_size)
        self.lstm   = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Drop the last token (usually <end>) and prepend the image features as step 0
        embeddings = torch.cat((features.unsqueeze(1), self.embed(captions[:, :-1])), dim=1)
        hiddens, _ = self.lstm(embeddings)
        return self.linear(hiddens)

    def sample(self, features, max_len=20, start_idx=0):
        """Greedy decoding for inference."""
        sampled_ids = []
        inputs, states = features.unsqueeze(1), None

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            predicted = self.linear(hiddens.squeeze(1)).max(1)[1]
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)

        return sampled_ids


class RetinalCaptioner(nn.Module):
    """Combined encoder-decoder model — wraps EncoderCNN and DecoderRNN together."""

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        return self.decoder(self.encoder(images), captions)
