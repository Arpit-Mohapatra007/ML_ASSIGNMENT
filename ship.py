from common import *
final = load_model('models/final.pt')[0]
t_star = json.load(open('reports/chosen.json'))['t']

final.cpu()
torch.save(final.state_dict(), 'arogya.pt')
final.to(dev)
E = embed(final, tr)
mu = E.mean(0)
ic = np.linalg.inv(np.cov(E.T) + 1e-3 * np.eye(E.shape[1]))
dd = np.sqrt(np.einsum('ij,jk,ik->i', E - mu, ic, E - mu))
Ev = embed(final, va)
dv = np.sqrt(np.einsum('ij,jk,ik->i', Ev - mu, ic, Ev - mu))
np.savez('ood.npz', mu=mu, ic=ic, thr=np.percentile(dv, 99.5))
pv, yb = predict(final, va, 224), yv[va]
hi = 0.999
for t in np.linspace(t_star, 0.999, 400):
    tp_, fp_ = (pv[yb == 1] >= t).mean(), (pv[yb == 0] >= t).mean()
    if tp_ * .1 / (tp_ * .1 + fp_ * .9 + 1e-9) >= .8:
        hi = float(t)
        break
hi = max(hi, t_star + .05)
json.dump(dict(t=t_star, hi=hi), open('cfg.json', 'w'))
print(os.path.getsize('arogya.pt') / 1e6, 'MB')
cm = ArogyaNet()
cm.load_state_dict(torch.load('arogya.pt', map_location='cpu'))
cm.eval()
x = torch.randn(1, 1, 224, 224)
torch.set_num_threads(2)
for _ in range(10):
    cm(x)
ts = []
for _ in range(200):
    t0 = time.perf_counter()
    with torch.no_grad():
        cm(x)
    ts.append((time.perf_counter() - t0) * 1000)
print('latency ms median', np.median(ts), 'p95', np.percentile(ts, 95))
