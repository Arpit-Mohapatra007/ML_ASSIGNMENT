from common import *
final = load_model('models/final.pt')[0]
t_star = json.load(open('reports/chosen.json'))['t']
yt = yv[te_idx]
ix = torch.as_tensor(te_idx, device=dev)
rows = []
for b in [-.2, -.1, 0, .1, .2]:
    for c in [.7, 1, 1.3]:
        x = (Xs[224][ix] * c + b).clamp(0, 1)
        d = metrics(yt, predict_t(final, x, M[ix]), t_star)
        rows.append([b, c, d['spec'], d['rec'], d['acc']])
r = pd.DataFrame(rows, columns=['bias', 'gain', 'spec', 'rec', 'acc'])
print(r.round(3))
r.to_csv('reports/shift_sim.csv', index=False)
