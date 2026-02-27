class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding= nn.Embedding(vocab_size,d_model)

    def forward(self, x_batch):
        x_emb=self.embedding(x_batch)
        return x_emb
