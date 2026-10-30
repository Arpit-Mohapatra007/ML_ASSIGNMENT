import os, glob, hashlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import imagehash
from PIL import Image
os.makedirs('reports', exist_ok=True)
root = 'data/chest_xray'

rows = []
for sp in ['train', 'test', 'val']:
    for c in ['NORMAL', 'PNEUMONIA']:
        for p in sorted(glob.glob(f'{root}/{sp}/{c}/*')):
            rows.append([p, sp, c, int(c == 'PNEUMONIA')])
df = pd.DataFrame(rows, columns=['path', 'split', 'cls', 'y'])
print(pd.crosstab(df.split, df.cls))

rows = []
for p in df.path:
    im = Image.open(p)
    a = np.array(im)
    grey = im.mode == 'L' or (a.ndim == 3 and (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all())
    g = a if a.ndim == 2 else a[..., 0]
    rows.append([a.shape[0], a.shape[1], int(grey), g.mean(), g.std()])
df = pd.concat([df, pd.DataFrame(rows, columns=['h', 'w', 'grey', 'mean', 'std'])], axis=1)
df['ar'] = df.w / df.h
df['area'] = df.h * df.w
print(df.groupby('cls')[['h', 'w', 'ar', 'area', 'mean', 'std', 'grey']].agg(['mean', 'std']).round(2).T)
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for k, c in enumerate(['mean', 'area', 'ar', 'h']):
    for cl in ['NORMAL', 'PNEUMONIA']:
        ax[k].hist(df[df.cls == cl][c], bins=40, alpha=.5, density=True, label=cl)
    ax[k].set_title(c)
    ax[k].legend()
plt.savefig('reports/pixel_stats.png', bbox_inches='tight')

df['md5'] = [hashlib.md5(open(p, 'rb').read()).hexdigest() for p in df.path]
print('exact', df.md5.duplicated().sum())
hs = [imagehash.phash(Image.open(p).convert('L')) for p in df.path]
H = np.array([h.hash.flatten() for h in hs]).astype(np.float32)
D = 64 - (H @ H.T + (1 - H) @ (1 - H).T)
np.fill_diagonal(D, 99)
for t in [0, 2, 4, 6]:
    print('pairs with hamming <=', t, np.triu(D <= t).sum())
i, j = np.where(np.triu(D <= 4))
sp = df.split.values
same = sp[i] == sp[j]
cross = ((sp[i] == 'train') & (sp[j] == 'test')) | ((sp[i] == 'test') & (sp[j] == 'train'))
print('near within split', same.sum(), 'near train-test', cross.sum())
ci, cj = i[cross], j[cross]
o = np.argsort(D[ci, cj])[:5]
fig, ax = plt.subplots(5, 2, figsize=(6, 15))
for r, k in enumerate(o):
    ax[r, 0].imshow(Image.open(df.path[ci[k]]).convert('L'), cmap='gray')
    ax[r, 1].imshow(Image.open(df.path[cj[k]]).convert('L'), cmap='gray')
    ax[r, 0].set_title(f'hamming {D[ci[k], cj[k]]:.0f}')
    ax[r, 0].axis('off')
    ax[r, 1].axis('off')
plt.savefig('reports/worst_pairs.png', bbox_inches='tight')

np.save('reports/near_pairs.npy', np.stack([i, j]))
df.to_csv('reports/inventory.csv', index=False)
