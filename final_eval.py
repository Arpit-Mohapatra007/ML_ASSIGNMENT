from common import *
import matplotlib.pyplot as plt
final = load_model('models/final.pt')[0]
t_star = json.load(open('reports/chosen.json'))['t']
yt = yv[te_idx]
pt = predict(final, te_idx, 224)
d = metrics(yt, pt, t_star)
print(d)
pd.DataFrame([d]).to_csv('reports/final_test_metrics.csv', index=False)
fpr, tpr, _ = roc_curve(yt, pt)
pr_, rc_, _ = precision_recall_curve(yt, pt)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(fpr, tpr)
ax[0].scatter(d['fp'] / (d['fp'] + d['tn']), d['rec'], c='r')
ax[0].set_title('ROC')
ax[1].plot(rc_, pr_)
ax[1].scatter(d['rec'], d['prec'], c='r')
ax[1].set_title('PR')
plt.savefig('reports/roc_pr_final.png', bbox_inches='tight')

rng = np.random.RandomState(0)
bs = []
for _ in range(1000):
    k = rng.randint(0, len(yt), len(yt))
    if yt[k].min() != yt[k].max():
        bs.append(metrics(yt[k], pt[k], t_star))
print(pd.DataFrame(bs)[['acc', 'prec', 'rec', 'spec', 'f1', 'roc', 'pr']].quantile([.025, .975]).round(3))
cm_ = confusion_matrix(yt, pt >= t_star)
plt.figure(figsize=(3, 3))
plt.imshow(cm_, cmap='Blues')
for a_ in range(2):
    for b_ in range(2):
        plt.text(b_, a_, cm_[a_, b_], ha='center')
plt.xticks([0, 1], ['NORMAL', 'PNEUMONIA'])
plt.yticks([0, 1], ['NORMAL', 'PNEUMONIA'])
plt.savefig('reports/confusion.png', bbox_inches='tight')
