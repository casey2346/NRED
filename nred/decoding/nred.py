import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from .baseline import baseline_decode
from .latent_reasoning import latent_reasoning

class ConsistencyHead(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
consistency_head = ConsistencyHead()

def NRED(prompt, tokenizer=None, model=None, device="cpu"):
    baseline = baseline_decode(prompt, tokenizer, model, device)

    # latent reasoning
    latent = latent_reasoning(prompt, tokenizer, model, device)

    # scoring
    emb = embedder.encode([latent], convert_to_tensor=True).detach()
    with torch.no_grad():
        score = consistency_head(emb).item()

    if score < 0.55:
        return {
            "baseline": baseline,
            "output": baseline,
            "latent": latent,
            "score": score,
            "mode": "fallback"
        }

    return {
        "baseline": baseline,
        "output": latent,
        "latent": latent,
        "score": score,
        "mode": "enhanced"
    }
