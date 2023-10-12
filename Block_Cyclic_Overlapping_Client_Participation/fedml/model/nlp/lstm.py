import torch.nn as nn
import torch

class BiLSTM(nn.Module):
    def __init__(self, embeddings_dim=768, hidden_dim=256, output_dim=4, n_layers=2, dropout=0.5):
        super(BiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(embeddings_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(text, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        hidden = torch.cat((hidden[-2, :, :], hidden[-1,:,:]), dim=1)
        
        return self.fc(self.dropout(hidden))