"""
we will construct the Word2Vec architecture : it needs to take 30-dim as an input, a training SkipGram corpus, and return 30-dim as an output (the fine-tuned embeddings, with better semantic similarity). 
"""
import torch.nn as nn
import torch

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.target_embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.context_embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)

    def forward(self, target_ids):
        # Output scores for all vocabulary words
        target_embeds = self.target_embeddings(target_ids)  # [batch_size, embedding_dim]
        scores = torch.matmul(target_embeds, self.context_embeddings.weight.T)  # [batch_size, vocab_size]
        return scores

