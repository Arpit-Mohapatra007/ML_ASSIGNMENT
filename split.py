import os, re
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold
seed = 42
df = pd.read_csv('reports/inventory.csv')
i, j = np.load('reports/near_pairs.npy')
print('95% CI for 16/16:', 0.025 ** (1 / 16), 1.0)
n = len(df)
def pid(p):
    f = os.path.basename(p)
    m = re.match(r'person(\d+)_', f)
    if m:
        return 'P' + m.group(1)
    m = re.match(r'(?:NORMAL2-)?IM-(\d+)-', f)
    if m:
        return 'N' + m.group(1)
    return f

df['pid'] = df.path.map(pid)
rep = df.reset_index().groupby('pid')['index'].transform('first').values
i, j = np.concatenate([i, np.arange(n)]), np.concatenate([j, rep])
A = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
_, grp = connected_components(A, directed=False)
print('largest groups', pd.Series(grp).value_counts().head(5).values, 'n groups', grp.max() + 1)
df['grp'] = grp
a, b = next(StratifiedGroupKFold(6, shuffle=True, random_state=seed).split(df, df.y, df.grp))
df['role'] = 'pool'
df.loc[b, 'role'] = 'test'
df['fold'] = -1
pool = df[df.role == 'pool']
for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(pool, pool.y, pool.grp)):
    df.loc[pool.index[b], 'fold'] = k
df.to_csv('reports/split.csv', index=False)
print(pd.crosstab([df.role, df.fold], df.cls))
