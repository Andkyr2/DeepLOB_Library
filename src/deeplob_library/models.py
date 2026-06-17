import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, seq_len = 100,num_classes = 3,num_features = 40, hidden_dim = 512,dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes),
        )
    
    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self, seq_len = 100,num_classes = 3,num_features = 40, hidden_channels = 64,dropout=0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(num_features, hidden_channels, kernel_size=5,padding = 2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5,padding = 2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=5,padding = 2),
            nn.BatchNorm1d(hidden_channels*2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))

        self.fc = nn.Linear(hidden_channels*2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x).squeeze(-1)
        return self.fc(x)




class DeepLOB(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

        self.inp3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

        self.lstm = nn.LSTM(input_size=192, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.cat([self.inp1(x), self.inp2(x), self.inp3(x)], dim=1)

        # (N, 192, T', 1) -> (N, T', 192)
        x = x.permute(0, 2, 1, 3).squeeze(-1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)

        


class VariationalLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers = 1,
        input_dropout = 0.2,
        recurrent_dropout = 0.2,
        batch_first = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout
        self.batch_first = batch_first

        self.cells = nn.ModuleList([nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def _sample_mask(self,batch,dim,p,device,dtype):
        if not self.training or p == 0:
            return None
        mask = torch.empty(batch,dim,device=device,dtype=dtype).bernoulli_(1-p)
        return mask.div_(1-p)

    def forward(self,x,hidden = None):

        if not self.batch_first:
            x = x.transpose(0,1)

        batch,T,_ = x.shape
        device,dtype = x.device,x.dtype

        if hidden is None:
            h = [torch.zeros(batch,self.hidden_size,device=device,dtype=dtype) for _ in range(self.num_layers)]
            c = [torch.zeros(batch,self.hidden_size,device=device,dtype=dtype) for _ in range(self.num_layers)]
        else:
            h,c = hidden

        input_masks, recurrent_masks = [],[]
        for i in range(self.num_layers):
            in_dim = x.size(2) if i == 0 else self.hidden_size
            input_masks.append(self._sample_mask(batch,in_dim,self.input_dropout,device,dtype))
            recurrent_masks.append(self._sample_mask(batch,self.hidden_size,self.recurrent_dropout,device,dtype))


        outputs = []
        for t in range(T):
            inp = x[:,t,:]

            for i, cell in enumerate(self.cells):
                if input_masks[i] is not None:
                    inp = inp * input_masks[i]
                
                h_prev = h[i]
                if recurrent_masks[i] is not None:
                    h_prev = h_prev * recurrent_masks[i]
                
                h[i],c[i] = cell(inp,(h_prev,c[i]))
                inp = h[i]

            outputs.append(h[-1])

        out = torch.stack(outputs,dim=1)
        if not self.batch_first:
            out = out.transpose(0,1)
        return out, (h,c)

class VariationalDeepLOB(nn.Module):
    def __init__(self, num_classes=3,dropout=0.2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

        self.inp3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

        self.lstm = VariationalLSTM(input_size=192, hidden_size=64, num_layers=1, input_dropout=dropout, recurrent_dropout=dropout, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.cat([self.inp1(x), self.inp2(x), self.inp3(x)], dim=1)

        # (N, 192, T', 1) -> (N, T', 192)
        x = x.permute(0, 2, 1, 3).squeeze(-1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)