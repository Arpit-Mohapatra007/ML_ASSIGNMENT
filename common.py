import os, glob, json, time, copy, random
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from sklearn.metrics import *
from arogya import ArogyaNet

for d in ['reports', 'models', 'cache']:
    os.makedirs(d, exist_ok=True)

seed = 42
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(s):
    os.environ['PYTHONHASHSEED'] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed', s)

set_seed(seed)

df = pd.read_csv('reports/split.csv')
yv = df.y.values

def folds(k):
    return df.index[(df.role == 'pool') & (df.fold != k)].values, df.index[(df.role == 'pool') & (df.fold == k)].values

tr, va = folds(0)
te_idx = df.index[df.role == 'test'].values

if os.path.exists('cache/X.npy'):
    X = np.load('cache/X.npy')
else:
    X = np.stack([cv2.resize(cv2.imread(p, 0), (224, 224)) for p in df.path])
    np.save('cache/X.npy', X)

Xt = torch.tensor(X).float().div(255).unsqueeze(1).to(dev)
Xs = {224: Xt, 96: F.interpolate(Xt, size=96, mode='area')}
mt = np.stack([df['mean'], df.ar, np.log(df.area)], 1)
mt = (mt - mt[tr].mean(0)) / mt[tr].std(0)
M = torch.tensor(mt).float().to(dev)
Y = torch.tensor(yv).to(dev)

def metrics(y, p, t=0.5):
    yh = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, yh), bacc=balanced_accuracy_score(y, yh),
                prec=precision_score(y, yh, zero_division=0), rec=recall_score(y, yh),
                spec=tn / (tn + fp), f1=f1_score(y, yh), roc=roc_auc_score(y, p),
                pr=average_precision_score(y, p), tn=tn, fp=fp, fn=fn, tp=tp)

tab = []
cols = ['acc', 'bacc', 'prec', 'rec', 'spec', 'f1', 'roc', 'pr']
def add(name, y, p, t=0.5):
    d = metrics(y, p, t)
    d['name'] = name
    tab.append(d)
    return d
def addseed(name, res):
    d = pd.DataFrame(res)
    r = {k: f'{d[k].mean():.3f}±{d[k].std():.3f}' for k in cols}
    r['name'] = name
    tab.append(r)
def show():
    return pd.DataFrame(tab).set_index('name')[cols]
def save(fn):
    show().to_csv(fn)

def aug_batch(x):
    n = x.size(0)
    a = (torch.rand(n, device=dev) - .5) * 2 * 0.1745
    s = 1 + (torch.rand(n, device=dev) - .5) * 0.2
    c, sn = torch.cos(a) / s, torch.sin(a) / s
    z = torch.zeros_like(c)
    th = torch.stack([torch.stack([c, -sn, z], 1), torch.stack([sn, c, z], 1)], 1)
    g = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, g, padding_mode='border', align_corners=False)

def bright(x):
    n = x.size(0)
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) - .5) * 0.6
    b = (torch.rand(n, 1, 1, 1, device=dev) - .5) * 0.3
    return (x * c + b).clamp(0, 1)

@torch.no_grad()
def predict_t(model, Xg, Mg, bs=128):
    model.eval()
    out = []
    for b in range(0, len(Xg), bs):
        out.append(F.softmax(model((Xg[b:b + bs] - .5) / .25, Mg[b:b + bs]), 1)[:, 1].cpu())
    return torch.cat(out).numpy()

def predict(model, idx, size):
    ix = torch.as_tensor(idx, device=dev)
    return predict_t(model, Xs[size][ix], M[ix])

def val_loss(model, va, size):
    p = predict(model, va, size)
    y = yv[va]
    return float(-np.mean(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7)))

def fit(model, tr, va, size, aug=False, bc=False, epochs=15, lr=1e-3, bs=32, wd=1e-4, cw=(1., 1.), patience=5):
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = torch.tensor(cw, device=dev).float()
    Xg = Xs[size]
    best, bad, state, hist = 1e9, 0, None, []
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        tl = 0
        for b in range(0, len(perm), bs):
            ix = torch.as_tensor(perm[b:b + bs], device=dev)
            if len(ix) < 2:
                continue
            x = Xg[ix]
            if aug:
                x = aug_batch(x)
            if bc:
                x = bright(x)
            loss = F.cross_entropy(model((x - .5) / .25, M[ix]), Y[ix], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(ix)
        vl = val_loss(model, va, size)
        hist.append([tl / len(perm), vl])
        print('epoch', ep + 1, round(tl / len(perm), 4), round(vl, 4), flush=True)
        if vl < best:
            best, bad, state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(state)
    return model, hist

class Res18(nn.Module):
    def __init__(s, pre=True):
        super().__init__()
        s.m = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pre else None)
        s.m.fc = nn.Linear(512, 2)
    def forward(s, x, m=None):
        return s.m(x.repeat(1, 3, 1, 1))

def build(kind, mk):
    return Res18(**mk) if kind == 'res18' else ArogyaNet(**mk)

def save_model(m, path, mk=None, kind='arogya', size=224):
    torch.save(dict(sd=m.state_dict(), kind=kind, mk=mk or {}, size=size), path)

def load_model(path):
    d = torch.load(path, map_location=dev)
    m = build(d['kind'], d['mk']).to(dev)
    m.load_state_dict(d['sd'])
    return m, d['size']

def run(name, kind='arogya', mk=None, size=224, seeds=(0, 1, 2), **kw):
    res = []
    for s in seeds:
        set_seed(s)
        m, h = fit(build(kind, mk or {}), tr, va, size, **kw)
        res.append(metrics(yv[va], predict(m, va, size)))
        save_model(m, f'models/{name}_{s}.pt', mk, kind, size)
    addseed(name, res)
    return res

@torch.no_grad()
def embed(model, idx, size=224, bs=128):
    model.eval()
    out = []
    for b in range(0, len(idx), bs):
        ix = torch.as_tensor(idx[b:b + bs], device=dev)
        out.append(model.feats((Xs[size][ix] - .5) / .25, M[ix]).cpu())
    return torch.cat(out).numpy()
