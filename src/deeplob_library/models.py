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


class BiN(nn.Module):
    '''Bilinear normalizaiton'''
    def __init__(self,num_features,seq_len):
        super().__init__()
        self.d1 = num_features
        self.t1 = seq_len

        self.B1 = nn.Parameter(torch.zeros(seq_len,1))
        self.l1 = nn.Parameter(torch.empty(seq_len,1))

        nn.init.xavier_normal_(self.l1)

        self.B2 = nn.Parameter(torch.zeros(num_features,1))
        self.l2 = nn.Parameter(torch.empty(num_features,1))

        nn.init.xavier_normal_(self.l2)

        self.y1 = nn.Parameter(torch.tensor(0.5))
        self.y2 = nn.Parameter(torch.tensor(0.5))

    def forward(self,x):
        #x: (B,num_features,seq_len)
        y1 = torch.clamp(self.y1,min = 0.0)
        y2 = torch.clamp(self.y2,min = 0.0)

        #normalize along temporal dimension
        x2 = x.mean(dim=2,keepdim=True)
        std2 = x.std(dim=2,keepdim=True)
        #guard against zero std
        std2 = torch.where(std2 < 1e-6,torch.ones_like(std2),std2)
        Z2 = (x - x2) / (std2)
        #learnable affine
        X2 = self.l2 * Z2 + self.B2

        
        #normalize along feature dimension
        x1 = x.mean(dim=1,keepdim=True)
        std1 = x.std(dim=1,keepdim=True)
        #guard against zero std
        std1 = torch.where(std1 < 1e-6,torch.ones_like(std1),std1)
        Z1 = (x - x1) / std1
        #learnable affine
        X1 = self.l1.T * Z1 + self.B1.T

        return y1 * X1 + y2 * X2

class MLPBlock(nn.Module):


    def __init__(self,start_dim,hidden_dim,final_dim):
        super().__init__()
        self.fc = nn.Linear(start_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,final_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(final_dim)

    def forward(self,x):
        res = x
        x = self.fc2(self.gelu(self.fc(x)))
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        x = self.ln(x)
        return self.gelu(x)

def _build_classifier_head(total_dim,num_classes):
    '''repeatedly shrink by 4x until dim < 128 and then classify'''
    layers = nn.ModuleList()
    while total_dim > 128:
        layers.append(nn.Linear(total_dim,total_dim//4))
        layers.append(nn.GELU())
        total_dim //= 4
    layers.append(nn.Linear(total_dim,num_classes))
    return layers

class MLPLOB(nn.Module):
    def __init__(self,hidden_dim = 128, num_layers = 4,seq_len = 100,num_features = 40,num_classes = 3):

        super().__init__()
        assert hidden_dim % 4 == 0 and seq_len % 4 == 0

        self.norm_layer = BiN(num_features,seq_len)
        self.proj = nn.Linear(num_features,hidden_dim)
        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            last = (i == num_layers - 1)
            f_mid,f_out = (hidden_dim*2,hidden_dim//4) if last else (hidden_dim*4,hidden_dim)
            t_mid,t_out = (seq_len*2,seq_len//4) if last else (seq_len*4,seq_len)
            self.blocks.append(MLPBlock(hidden_dim,f_mid,f_out))
            self.blocks.append(MLPBlock(seq_len,t_mid,t_out))


        total_dim = (hidden_dim//4) * (seq_len//4)
        self.final_layers = _build_classifier_head(total_dim,num_classes)


    def forward(self,x):
        #x: (B,seq_len,num_features)
        x = self.norm_layer(x.permute(0,2,1)).permute(0,2,1) #bin expects (B,num_features,seq_len)
        x = self.proj(x)
        for i,block in enumerate(self.blocks):
            if i % 2 == 0:
                x = block(x)
            else:
                x = block(x.permute(0,2,1)).permute(0,2,1)
        x = x.reshape(x.size(0),-1)
        for layer in self.final_layers:
            x = layer(x)
        return x


class ComputeQKV(nn.Module):
    def __init__(self,hidden_dim,num_heads):
        super().__init__()
        self.q = nn.Linear(hidden_dim,num_heads*hidden_dim)
        self.k = nn.Linear(hidden_dim,num_heads*hidden_dim)
        self.v = nn.Linear(hidden_dim,num_heads*hidden_dim)
        
    

    def forward(self,x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q,k,v

class TransformerLayer(nn.Module):
    def __init__(self,hidden_dim,num_heads,final_dim):
        super().__init__()
        self.qkv = ComputeQKV(hidden_dim,num_heads)
        self.attention = nn.MultiheadAttention(hidden_dim*num_heads,num_heads,batch_first=True)
        self.w0 = nn.Linear(hidden_dim*num_heads,hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim,hidden_dim*4,final_dim)

    def forward(self,x):
        res = x
        q,k,v = self.qkv(x)
        x,_ = self.attention(q,k,v)
        
        x = self.norm(self.w0(x) + res)
        x = self.mlp(x)
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x

def sinusoidal_positional_embedding(seq_len,dim, n = 10000.0):

    if dim % 2 != 0:
        raise ValueError("dim must be even")
    pos  = torch.arange(seq_len).unsqueeze(1)
    emb = torch.zeros(seq_len,dim)
    denom = torch.pow(n,torch.arange(0,dim,2)/dim)
    emb[:,0::2] = torch.sin(pos/denom)
    emb[:,1::2] = torch.cos(pos/denom)
    return emb

class TLOB(nn.Module):
    def __init__(self,hidden_dim = 128, num_layers = 4,seq_len = 100,num_features = 40,num_heads = 1,num_classes =3,sinusoidal = True):

        super().__init__()
        assert seq_len % 4 == 0 and hidden_dim % 4 == 0
        self.norm = BiN(num_features,seq_len)
        self.embed = nn.Linear(num_features,hidden_dim)

        if sinusoidal:
            self.register_buffer('pos',sinusoidal_positional_embedding(seq_len,hidden_dim))
        else:
            self.pos = nn.Parameter(torch.randn(1,seq_len,hidden_dim))

        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            last = (i == num_layers - 1)
            h_out = hidden_dim//4 if last else hidden_dim
            t_out = seq_len//4 if last else seq_len
            self.blocks.append(TransformerLayer(hidden_dim,num_heads,h_out))
            self.blocks.append(TransformerLayer(seq_len,num_heads,t_out))

        self.head = _build_classifier_head((hidden_dim//4) * (seq_len//4),num_classes)

    def forward(self,x):
        #x: (B,seq_len,num_features)
        x = self.norm(x.permute(0,2,1)).permute(0,2,1)
        x = self.embed(x) + self.pos
        for i,block in enumerate(self.blocks):
            if i % 2 == 0:
                x = block(x)
            else:
                x = block(x.permute(0,2,1)).permute(0,2,1)
        x = x.reshape(x.size(0),-1)
        for layer in self.head:
            x = layer(x)

        return x

#sanity check
if __name__ == "__main__":
    x = torch.randn(8,128,40)
    print(MLPLOB(hidden_dim = 128,seq_len = 128,num_features = 40,num_classes = 3)(x).shape)
    print(TLOB(hidden_dim = 128,seq_len = 128,num_features = 40,num_classes = 3,sinusoidal = True)(x).shape)


