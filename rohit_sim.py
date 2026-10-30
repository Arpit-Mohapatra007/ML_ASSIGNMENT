from common import *
from sklearn.model_selection import train_test_split

def go(nm, a, b):
    res = []
    for s in (0, 1, 2):
        set_seed(s)
        keep = np.arange(len(a)) % 10 != 0
        m, _ = fit(ArogyaNet(), a[keep], a[~keep], 224, aug=True)
        res.append(metrics(yv[b], predict(m, b, 224)))
    addseed(nm, res)

go('kaggle_train_to_test', df.index[df.split == 'train'].values, df.index[df.split == 'test'].values)
a, b = train_test_split(df.index.values, test_size=.2, stratify=yv, random_state=seed)
go('random_split_leaky', a, b)
go('grouped_split', df.index[df.role == 'pool'].values, te_idx)
print(show())
save('reports/rohit_repro.csv')
v = df[df.split == 'val']
print('val n', len(v), '95% CI lower bound for 100% accuracy', 0.025 ** (1 / len(v)))
