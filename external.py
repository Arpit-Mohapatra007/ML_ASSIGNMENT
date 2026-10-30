from common import *
if os.path.exists('reports/external_done'):
    raise SystemExit('external set already used')

import matplotlib.pyplot as plt

final = load_model('models/final.pt')[0]
fixed = load_model('models/fixed.pt')[0]
t_star = json.load(open('reports/chosen.json'))['t']
t_fixed = json.load(open('reports/chosen.json'))['t_fixed']
yt = yv[te_idx]

ext = 'data/covid/COVID-19_Radiography_Dataset'
ep = glob.glob(f'{ext}/Viral Pneumonia/images/*.png')
en = glob.glob(f'{ext}/Normal/images/*.png')
en = list(np.random.RandomState(0).choice(en, 3300, replace=False))
paths = en + ep
ye = np.array([0] * len(en) + [1] * len(ep))
raw = [cv2.imread(p, 0) for p in paths]
Xe = torch.tensor(np.stack([cv2.resize(r, (224, 224)) for r in raw])).float().div(255).unsqueeze(1).to(dev)
Me = torch.zeros(len(Xe), 3, device=dev)
pe, pfx = predict_t(final, Xe, Me), predict_t(fixed, Xe, Me)
tab.clear()
add('internal test, final', yt, predict(final, te_idx, 224), t_star)
add('external, final', ye, pe, t_star)
add('internal test, bright/contrast aug', yt, predict(fixed, te_idx, 224), t_fixed)
add('external, bright/contrast aug', ye, pfx, t_fixed)
print(show())
print('mean intensity internal', df['mean'].mean(), 'external', np.mean([r.mean() for r in raw]))
plt.hist(df['mean'], bins=40, alpha=.5, density=True, label='internal')
plt.hist([r.mean() for r in raw], bins=40, alpha=.5, density=True, label='external')
plt.legend()
plt.savefig('reports/shift.png', bbox_inches='tight')

save('reports/external.csv')

open('reports/external_done', 'w').write('1')
