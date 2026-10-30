import sys, json, numpy as np, cv2, torch
from arogya import ArogyaNet
cfg = json.load(open('cfg.json'))
ood = np.load('ood.npz')
mu, ic, thr = ood['mu'], ood['ic'], float(ood['thr'])
net = ArogyaNet()
net.load_state_dict(torch.load('arogya.pt', map_location='cpu'))
net.eval()

def predict(path):
    raw = cv2.imread(path)
    if raw is None:
        return 'invalid image file', None
    sat = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)[..., 1].mean()
    g = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    if sat > 25:
        return 'rejected: colour photo, not an X-ray', None
    if g.std() < 12:
        return 'rejected: blank image', None
    x = torch.tensor(cv2.resize(g, (224, 224)) / 255., dtype=torch.float32)[None, None]
    x = (x - .5) / .25
    with torch.no_grad():
        z = net.feats(x)
        p = torch.softmax(net.fc(z), 1)[0, 1].item()
    d = z.numpy()[0] - mu
    if np.sqrt(d @ ic @ d) > thr:
        return 'rejected: does not look like a paediatric chest X-ray', None
    if p < cfg['t']:
        return 'NORMAL', p
    if p < cfg['hi']:
        return 'LOW CONFIDENCE - refer to a clinician', p
    return 'PNEUMONIA', p

if __name__ == '__main__':
    print(*predict(sys.argv[1]))
