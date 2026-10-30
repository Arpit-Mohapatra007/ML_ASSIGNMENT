from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from skimage.feature import hog

X16 = np.stack([cv2.resize(x, (16, 16), interpolation=cv2.INTER_AREA) for x in X]).reshape(len(X), -1) / 255.
lr16 = LogisticRegression(max_iter=3000).fit(X16[tr], yv[tr])
add('16x16 pixels + LR', yv[va], lr16.predict_proba(X16[va])[:, 1])
gbm = GradientBoostingClassifier(random_state=seed).fit(mt[tr], yv[tr])
add('meta only (mean, ar, area)', yv[va], gbm.predict_proba(mt[va])[:, 1])
print('chance level', max(yv[va].mean(), 1 - yv[va].mean()))

dm = DummyClassifier(strategy='most_frequent').fit(X16[tr], yv[tr])
add('majority', yv[va], dm.predict_proba(X16[va])[:, 1])
Xh = np.stack([hog(cv2.resize(x, (128, 128)), pixels_per_cell=(16, 16), cells_per_block=(2, 2)) for x in X])
lrh = LogisticRegression(max_iter=3000).fit(Xh[tr], yv[tr])
add('HOG + LR', yv[va], lrh.predict_proba(Xh[va])[:, 1])
save('reports/baselines.csv')
print(show())
