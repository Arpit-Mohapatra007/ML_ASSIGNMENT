import torch, torch.nn as nn, torch.nn.functional as F

class SE(nn.Module):
    def __init__(s, c, r=8):
        super().__init__()
        s.fc1 = nn.Linear(c, c // r)
        s.fc2 = nn.Linear(c // r, c)
    def forward(s, x):
        z = x.mean((2, 3))
        z = torch.sigmoid(s.fc2(F.relu(s.fc1(z))))
        return x * z[:, :, None, None]

def dw(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU())

class Block(nn.Module):
    def __init__(s, cin, cout, stride, res, se):
        super().__init__()
        s.a = dw(cin, cout, stride)
        s.b = dw(cout, cout, 1)
        s.se = SE(cout) if se else nn.Identity()
        s.res = res
        s.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)) if res else None
    def forward(s, x):
        o = s.se(s.b(s.a(x)))
        if s.res:
            o = o + s.skip(x)
        return F.relu(o)

class ArogyaNet(nn.Module):
    def __init__(s, res=True, se=True, meta=False, drop=0.3):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        ch = [16, 32, 64, 128, 256]
        s.blocks = nn.Sequential(*[Block(ch[i], ch[i + 1], 2, res, se) for i in range(4)])
        s.meta = nn.Sequential(nn.Linear(3, 16), nn.ReLU()) if meta else None
        s.drop = nn.Dropout(drop)
        s.fc = nn.Linear(256 + (16 if meta else 0), 2)
    def feats(s, x, m=None):
        z = s.blocks(s.stem(x)).mean((2, 3))
        if s.meta is not None:
            z = torch.cat([z, s.meta(m)], 1)
        return z
    def forward(s, x, m=None):
        return s.fc(s.drop(s.feats(x, m)))
