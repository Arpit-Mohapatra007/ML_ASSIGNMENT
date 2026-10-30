import numpy as np, cv2, pandas as pd
from predict import predict
df = pd.read_csv('reports/split.csv')
t = df[df.role == 'test']
r = pd.Series([predict(p)[0] for p in t.path], index=t.index)
print(r.value_counts())
print('false reject rate', r.str.startswith('rejected').mean())
print('refer band share', r.str.startswith('LOW').mean())
dec = r[r.isin(['NORMAL', 'PNEUMONIA'])]
print('accuracy outside band', ((dec == 'PNEUMONIA').astype(int) == t.y[dec.index]).mean(), 'n', len(dec))
print('pneumonia share inside band', t.y[r.str.startswith('LOW')].mean())
cv2.imwrite('noise.png', np.random.RandomState(0).randint(0, 255, (512, 512), np.uint8))
cv2.imwrite('white.png', np.full((512, 512), 255, np.uint8))
cv2.imwrite('dog_grey.png', cv2.imread('dog.jpg', 0))
for f in ['noise.png', 'white.png', 'dog.jpg', 'dog_grey.png']:
    print(f, predict(f))
