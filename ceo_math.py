from common import *
final = load_model('models/final.pt')[0]
t_star = json.load(open('reports/chosen.json'))['t']
yt = yv[te_idx]
pf_ = predict(final, te_idx, 224)

n, pi = 1000, 0.1
sick, well = n * pi, n * (1 - pi)
for t in [0.5, t_star]:
    d = metrics(yt, pf_, t)
    tp, fn = sick * d['rec'], sick * (1 - d['rec'])
    tn, fp = well * d['spec'], well * (1 - d['spec'])
    print(f't={t:.3f} acc={(tp + tn) / n:.3f} missed={fn:.1f} false_alarms={fp:.1f} ppv={tp / (tp + fp):.3f}')
print('always normal: acc', well / n, 'missed', sick)
print('99% accuracy allows', n * .01, 'errors, and all of them can be missed sick children out of', sick)
print('always pneumonia on the Kaggle test folder:', 390 / 624)

e = pd.read_csv('reports/external.csv', index_col=0)
for nm in e.index:
    r = e.loc[nm]
    tp, fn = sick * r['rec'], sick * (1 - r['rec'])
    tn, fp = well * r['spec'], well * (1 - r['spec'])
    print(nm, 'at camp prevalence: acc', round((tp + tn) / n, 3), 'missed', round(fn, 1), 'false_alarms', round(fp, 1), 'ppv', round(tp / (tp + fp), 3), 'cost per 1000', round(20 * fn + fp, 1))
print('refer everyone cost', well, 'always normal cost', 20 * sick, '99% acc all misses', 20 * n * .01, 'all false alarms', n * .01)
