from common import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

n0, n1 = (yv[tr] == 0).sum(), (yv[tr] == 1).sum()
cw = (len(tr) / (2 * n0), len(tr) / (2 * n1))
set_seed(0)
base, _ = fit(ArogyaNet(), tr, va, 224, aug=True)
set_seed(0)
wm, _ = fit(ArogyaNet(), tr, va, 224, aug=True, cw=cw)
pb, pw = predict(base, va, 224), predict(wm, va, 224)
rows = []
for t in [.1, .3, .5, .7, .9]:
    for nm, p in [('base', pb), ('weighted', pw)]:
        d = metrics(yv[va], p, t)
        rows.append([nm, t, d['roc'], d['rec'], d['spec']])
print(pd.DataFrame(rows, columns=['model', 'thr', 'roc', 'rec', 'spec']).round(3))
print('prob correlation base vs weighted', np.corrcoef(pb, pw)[0, 1])

def pick_t(y, p, pi=0.1, cfn=20, cfp=1):
    fpr, tpr, th = roc_curve(y, p)
    cost = cfn * pi * (1 - tpr) + cfp * (1 - pi) * fpr
    k = cost[1:].argmin() + 1
    return float(th[k]), fpr[k], tpr[k], cost[k]

t_star, f_s, t_s, c_s = pick_t(yv[va], pb)
print('threshold', t_star, 'fpr', f_s, 'tpr', t_s, 'cost per patient', c_s)
print('PPV at camp prevalence', t_s * .1 / (t_s * .1 + f_s * .9))
fpr, tpr, _ = roc_curve(yv[va], pb)
pr_, rc_, _ = precision_recall_curve(yv[va], pb)
yh = pb >= t_star
pp = (yh & (yv[va] == 1)).sum() / yh.sum()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(fpr, tpr)
ax[0].scatter(f_s, t_s, c='r')
ax[0].set_title('ROC')
ax[1].plot(rc_, pr_)
ax[1].scatter(t_s, pp, c='r')
ax[1].set_title('PR')
plt.savefig('reports/roc_pr.png', bbox_inches='tight')

yt = yv[te_idx]
pt = predict(base, te_idx, 224)
print(metrics(yt, pt, t_star))
cv = []
for k in range(5):
    a, b = folds(k)
    a_val = a[::10]
    a_tr = np.delete(a, np.arange(0, len(a), 10))
    set_seed(0)
    m, _ = fit(ArogyaNet(), a_tr, a_val, 224, aug=True)
    tk = pick_t(yv[a_val], predict(m, a_val, 224))[0]
    cv.append(metrics(yv[b], predict(m, b, 224), tk))
d = pd.DataFrame(cv)[['acc', 'prec', 'rec', 'spec', 'f1', 'roc', 'pr']]
print(d.agg(['mean', 'std']).round(3))

rng = np.random.RandomState(0)
trials = []
for n in range(12):
    cfg = dict(lr=10 ** rng.uniform(-4, -2), bs=int(rng.choice([16, 32, 64])),
               dropout=float(rng.choice([0, .2, .5])), wd=float(rng.choice([0, 1e-4, 1e-2])))
    set_seed(0)
    m, h = fit(ArogyaNet(drop=cfg['dropout']), tr, va, 224, aug=True, epochs=8, lr=cfg['lr'], bs=cfg['bs'], wd=cfg['wd'])
    trials.append({**cfg, 'roc': roc_auc_score(yv[va], predict(m, va, 224))})
res_df = pd.DataFrame(trials).sort_values('roc', ascending=False)
print('budget: 12 random trials x 8 epochs, seed 0')
print(res_df.round(4))
best = res_df.iloc[0]
hp = dict(lr=best['lr'], bs=int(best['bs']), wd=best['wd'])
set_seed(0)
mu_, hu = fit(ArogyaNet(drop=0), tr, va, 224, aug=False, wd=0, epochs=25, patience=999)
set_seed(0)
final, hr = fit(ArogyaNet(drop=best['dropout']), tr, va, 224, aug=True, epochs=25, patience=5, **hp)
hu, hr = np.array(hu), np.array(hr)
print('unregularised: val loss best at epoch', hu[:, 1].argmin() + 1, 'best', hu[:, 1].min().round(4), 'final', hu[-1, 1].round(4))
print('regularised: stopped at epoch', len(hr), 'best epoch', hr[:, 1].argmin() + 1, 'best', hr[:, 1].min().round(4))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, (h, nm) in enumerate([(hu, 'unregularised'), (hr, 'regularised')]):
    ax[k].plot(h[:, 0], label='train')
    ax[k].plot(h[:, 1], label='val')
    ax[k].set_title(nm)
    ax[k].legend()
plt.savefig('reports/curves.png', bbox_inches='tight')
t_star = pick_t(yv[va], predict(final, va, 224))[0]
print('final threshold', t_star)

members = [('arogya_full', 0), ('arogya_full', 1), ('arogya_full', 2), ('resnet18_imagenet', 0),
           ('arogya_no_se', 0), ('arogya_no_res', 0), ('arogya_meta', 0)]
P = {}
for n, s in members:
    m, sz = load_model(f'models/{n}_{s}.pt')
    P[f'{n}_{s}'] = predict(m, te_idx, sz)
P['soft_vote'] = np.mean(list(P.values()), 0)
m0 = load_model('models/arogya_full_0.pt')[0]
Etr, Ete = embed(m0, tr), embed(m0, te_idx)
P['rf_on_embed'] = RandomForestClassifier(300, random_state=seed, n_jobs=-1).fit(Etr, yv[tr]).predict_proba(Ete)[:, 1]
P['ada_on_embed'] = AdaBoostClassifier(n_estimators=200, random_state=seed).fit(Etr, yv[tr]).predict_proba(Ete)[:, 1]
tab.clear()
for k, p in P.items():
    add(k, yt, p)
save('reports/ensembles.csv')

Z = embed(final, te_idx)
pf_ = predict(final, te_idx, 224)
wrong = (pf_ >= t_star) != yt
Z2 = PCA(2).fit_transform(Z)
Z3 = TSNE(2, init='pca', random_state=seed).fit_transform(PCA(30).fit_transform(Z))
fig, ax = plt.subplots(2, 2, figsize=(10, 9))
for r, (zz, nm) in enumerate([(Z2, 'PCA'), (Z3, 't-SNE')]):
    ax[r, 0].scatter(zz[:, 0], zz[:, 1], c=yt, s=6, cmap='coolwarm')
    ax[r, 0].set_title(nm + ' true class')
    ax[r, 1].scatter(zz[:, 0], zz[:, 1], c=wrong, s=6, cmap='coolwarm')
    ax[r, 1].set_title(nm + ' wrong=red')
plt.savefig('reports/embeddings.png', bbox_inches='tight')
km = KMeans(2, n_init=10, random_state=seed).fit_predict(Z)
pur = sum(np.bincount(yt[km == c], minlength=2).max() for c in [0, 1]) / len(yt)
print('ARI', adjusted_rand_score(yt, km), 'purity', pur)
d = df.loc[te_idx]
print(d[wrong][['mean', 'std', 'area', 'ar']].mean())
print(d[~wrong][['mean', 'std', 'area', 'ar']].mean())
print('FN', ((pf_ < t_star) & (yt == 1)).sum(), 'FP', ((pf_ >= t_star) & (yt == 0)).sum())

set_seed(0)
fixed, _ = fit(ArogyaNet(drop=best['dropout']), tr, va, 224, aug=True, bc=True, epochs=25, patience=5, **hp)

save_model(final, 'models/final.pt', dict(drop=float(best['dropout'])))
save_model(fixed, 'models/fixed.pt', dict(drop=float(best['dropout'])))
t_fixed = pick_t(yv[va], predict(fixed, va, 224))[0]
json.dump(dict(t=float(t_star), t_fixed=float(t_fixed)), open('reports/chosen.json', 'w'))
