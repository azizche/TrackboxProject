import torch
import torch.nn as nn

class FootballTrackingLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 2) 
        
    def forward(
        self,
        x,
        #teacher_forcing_ratio = 0.0,
        #target = None
    ):
        """batch_size = x.size(0)
        seq_len = x.size(1)

        outputs = torch.zeros(batch_size, seq_len, 2).to(x.device)

        lstm_out, (hn, cn) = self.lstm(x[:, 0:1, :], )
        outputs[:, 0, :] = self.fc(lstm_out.squeeze(1))
        
        for t in range(1, seq_len):
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing and target is not None:
                prev_output = target[:, t-1, :]
            else:
                prev_output = outputs[:, t-1, :]
            
            current_input = torch.cat([x[:, t, :-2], prev_output], dim=1)
            current_input = current_input.unsqueeze(1)
            
            lstm_out, (hn, cn) = self.lstm(current_input, (hn, cn))
            outputs[:, t, :] = self.fc(lstm_out.squeeze(1))"""
        outputs,_ = self.lstm(x)
        outputs = self.fc(outputs)
        
        return outputs
