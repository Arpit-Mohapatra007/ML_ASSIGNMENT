from common import *
from torchinfo import summary
import matplotlib.pyplot as plt

s = summary(ArogyaNet(), input_size=(1, 1, 224, 224), col_names=('output_size', 'num_params'), verbose=0)
open('reports/arch.txt', 'w').write(str(s))
print(s)
names = ['input 1x224x224', 'stem conv3x3 s2, 16', 'Block 16>32 (2x DW-sep, SE, skip)', 'Block 32>64',
         'Block 64>128', 'Block 128>256', 'global avg pool', 'dropout', 'fc 2']
fig, ax = plt.subplots(figsize=(3, 9))
ax.axis('off')
ax.set_ylim(0, 1.05)
for k, nm in enumerate(names):
    ax.text(.5, 1 - k * .11, nm, ha='center', bbox=dict(boxstyle='round', fc='lightblue'))
    if k:
        ax.annotate('', xy=(.5, 1 - k * .11 + .03), xytext=(.5, 1 - (k - 1) * .11 - .03), arrowprops=dict(arrowstyle='->'))
plt.savefig('reports/arch.png', bbox_inches='tight')

run('resnet18_imagenet', 'res18', dict(pre=True), aug=True, lr=3e-4)
run('resnet18_scratch', 'res18', dict(pre=False), aug=True, lr=1e-3)
print(show())

a = run('arogya_full', aug=True)
b = run('arogya_no_se', mk=dict(se=False), aug=True)
c = run('arogya_no_res', mk=dict(res=False), aug=True)

for nm, r in [('no_se', b), ('no_res', c)]:
    d = np.array([x['roc'] for x in a]) - np.array([x['roc'] for x in r])
    print(nm, 'delta roc', d.mean().round(4), '±', d.std(ddof=1).round(4))
print(show())
m = run('arogya_meta', mk=dict(meta=True), aug=True)
d = np.array([x['roc'] for x in m]) - np.array([x['roc'] for x in a])
print('meta delta roc', d.mean().round(4), '±', d.std(ddof=1).round(4))
print(show())

run('arogya_96', size=96, aug=True)
run('arogya_224_noaug', aug=False)
print(show())

save('reports/part1_models.csv')
