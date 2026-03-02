class PositionalEncodings(nn.Module):
    def __init__(self, max_seq_length, d_model):
        super().__init__() 
        self.lpe = nn.Embedding(max_seq_length, d_model ) 
 
    def forward(self, input_ids): 
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len).to(input_ids.device)  
        positions = positions.unsqueeze(0).expand(batch_size, seq_len) 

        return self.lpe(positions) # Each word position is replaced by a learned vector. so final shape is b,s,d
