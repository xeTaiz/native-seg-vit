import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



def tensor2str(t):
    return f'{t.shape} in [{t.min(), t.max()}]'

def log_t(*args):
    for a in args:
        if isinstance(a, dict):
            print('Dict:')
            for k,v in a.items():
                print(f'{k}:')
                log_t(v)
        elif isinstance(a, (list, tuple)):
            print('Iterable:')
            for v in a:
                log_t(a) 
        elif torch.is_tensor(a) or isinstance(a, np.ndarray):
            print(tensor2str(a))
        elif isinstance(a, str):
            print(a)
        else:
            print(f'{a} ({type(a)})')

class LayerNormDim1(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


class GroupingLayer(nn.Module):
    def __init__(self, f_in, f_out, hidden_factor=2, kernel_size=2, iterations=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.f_in = f_in
        self.f_out = f_out

        self.conv = nn.Conv2d(f_in, f_out, kernel_size=kernel_size, stride=kernel_size, padding=0)
        self.k = nn.Linear(f_in, f_in*hidden_factor)
        self.q = nn.Linear(f_out, f_in*hidden_factor)
        self.v = nn.Linear(f_in, f_out)
        self.mlp = nn.Linear(f_out, f_out)

        self.ln_conv = LayerNormDim1(f_out)
        self.ln_v = LayerNormDim1(f_out)
        self.ln_mlp = LayerNormDim1(f_out)

    def forward(self, x_in, temp=1.0):
        # x_in B, f_in, H, W
        B, _, H, W = x_in.shape
        WIN = self.kernel_size * self.kernel_size
        x_out = self.ln_conv(self.conv(x_in))
        x_out = x_out
        x_in  = x_in
        log_t('x_in', x_in)
        log_t('x_out', x_out)
        T_out = x_out.size(-1) * x_out.size(-2)
        k = self.k(x_in.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        v = self.v(x_in.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        fold_args = {'kernel_size': self.kernel_size, 'stride': self.kernel_size}

        k_wins = F.unfold(k, **fold_args) 
        k_wins = k_wins.view(B, self.f_out, WIN, T_out)
        log_t('k', k)
        log_t('v', v)
        log_t('k_wins', k_wins)

        for _ in range(self.iterations):
            q = self.q(x_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            log_t('q', q)
            att = torch.einsum('bfwt,bft->bwt', k_wins, q.view(B, self.f_out, -1))
            att = torch.softmax(att * temp, dim=1)
            att = F.fold(att, x_in.shape[-2:], **fold_args)
            log_t('att', att)   
            update = v * att.expand(-1, self.f_out, -1, -1)
            update = F.unfold(update, **fold_args).view(B, self.f_out, WIN, T_out).sum(dim=2).view(x_out.shape)
            x_out += self.ln_v(update)
            x_out += self.ln_mlp(self.mlp(x_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        ups = F.unfold(att, **fold_args).reshape(B, WIN, *x_out.shape[-2:])
        log_t('ups', ups)

        return x_out






if __name__ == '__main__':
    with torch.no_grad():
        a = torch.rand(2, 16, 100, 100)
        grp = GroupingLayer(f_in=16, f_out=32, kernel_size=2)
        b = grp(a)
    log_t('out', b)
