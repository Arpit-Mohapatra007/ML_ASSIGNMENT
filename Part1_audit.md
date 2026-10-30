# Part 1: Audit of the data and the model

Every number comes from a log in `logs/` or a table in `reports/`. The script that made it is named in each heading. Seeds are printed at the top of each log (42 for splits, 0/1/2 for three-seed runs).

## 1. Inventory and pixel statistics (`audit.py`)

Rohit's folders hold 5,856 images (the brief says 5,863, so seven are unaccounted for). All are true greyscale.

| split | NORMAL | PNEUMONIA |
|---|---|---|
| train | 1,341 | 3,875 |
| test | 234 | 390 |
| val | 8 | 8 |

| | NORMAL | PNEUMONIA |
|---|---|---|
| height (px) | 1379 ± 343 | 820 ± 271 |
| width (px) | 1686 ± 305 | 1195 ± 285 |
| aspect ratio (w/h) | 1.25 ± 0.15 | 1.51 ± 0.25 |
| area (Mpx) | 2.42 ± 1.06 | 1.05 ± 0.61 |
| mean brightness | 122.6 ± 13.5 | 122.8 ± 19.9 |
| contrast (intensity std) | 61.3 ± 5.8 | 55.4 ± 10.0 |

![Pixel statistics by class](assets/pixel_stats.png)

**What it means.** Average brightness is the same in both classes, but image size is not. NORMAL files are 2.3 times larger and squarer. Contrast also differs. File size and shape describe how the image was exported, not the lungs, yet they reveal the class. Resizing everything to 224×224 hides the size but still stretches the wide pneumonia images more. A model can learn "big, squarish image = healthy" and never look at pathology.

## 2. Duplicates and leakage (`audit.py`, `split.py`)

* Exact duplicate files (MD5): 32.
* Near-duplicate pairs by perceptual hash, by Hamming distance: ≤0: 34, ≤2: 50, **≤4: 265**, ≤6: 1,970. The jump from 4 to 6 means ≤6 starts matching different but similar-looking X-rays, so I used 4.
* At ≤4, 238 pairs sit inside one split and **26 pairs cross train and test** (at most 26 of 624 test images, about 4%).
* The Kaggle filenames contain patient IDs (`personNNN_…`, `IM-NNNN-…`). I grouped images that share a patient ID or a near-duplicate link: 2,592 groups for 5,856 images, the largest with 188 images (3.2%).

![Worst five train/test pairs](assets/worst_pairs.png)

## 3. The validation set (`split.py`)

The shipped val folder has 16 images (8 and 8). A perfect score on 16 images has a 95% confidence interval of **0.794 to 1.000** (lower bound 0.025^(1/16)). So "100% on val" is compatible with a true accuracy near 80%.

Replacement, grouped and stratified:

* a **test set of 976 images** (712 pneumonia, 264 normal), never used for any decision;
* the other 4,880 images in **5 folds** of about 976, used for early stopping, tuning, threshold choice and cross-validation.

I rejected k-fold alone because it leaves no untouched final estimate. I rejected one split alone because one validation set is noisy here: validation loss jumps from epoch to epoch (up to 2.7 in `logs/train.txt`) and seed-to-seed accuracy varies by up to ±0.008. Groups keep duplicates and same-patient images on one side of every boundary.

## 4. Shortcut hunt (`baselines.py`)

Validation fold 0. Chance accuracy is 0.730.

| model | acc | bal. acc | sensitivity | specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| majority class | 0.730 | 0.500 | 1.000 | 0.000 | 0.500 | 0.730 |
| 3 metadata numbers (mean, aspect, area) | 0.864 | 0.823 | 0.912 | 0.735 | 0.923 | 0.972 |
| **16×16 pixels + logistic regression** | **0.938** | **0.911** | 0.969 | 0.852 | **0.976** | 0.990 |
| HOG + logistic regression | 0.962 | 0.949 | 0.978 | 0.920 | 0.990 | 0.996 |

A 16×16 thumbnail has 256 numbers and no texture, yet it beats chance by 21 points of accuracy. Three numbers about file size, shape and brightness reach ROC-AUC 0.92. So the data can be mostly solved with cues that are not lung texture, and any CNN is free to use them. That is why the external test in Part 2 matters.

## 5. Baselines

Majority class and HOG + logistic regression are in the table above (precision and F1 in `reports/baselines.csv`). **HOG + LR is the bar deep learning has to clear:** 0.962 accuracy, 0.990 ROC-AUC.

## 6. Transfer baseline (`train.py`, 3 seeds)

| model | acc | bal. acc | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| ResNet18, ImageNet weights | 0.976 ± 0.003 | 0.967 ± 0.008 | 0.995 ± 0.001 | 0.998 ± 0.000 |
| ResNet18, from scratch | 0.954 ± 0.009 | 0.937 ± 0.006 | 0.988 ± 0.006 | 0.995 ± 0.002 |

Expectation: ImageNet's low-level edge and blob filters should carry over to X-rays even though they were learned on colour photos. Colour-specific and high-level filters should not. So pretraining should mostly help fitting and give a modest gain. The numbers agree: +2.2 points accuracy and +0.007 ROC-AUC, with a smaller seed spread. ResNet18 has 11.2 M parameters (about 45 MB), so it does not fit the 25 MB channel.

## 7. ArogyaNet (`arogya.py`, `train.py`)

206,430 parameters (limit 2 M). Full table in `arch.txt`.

| stage | output | parameters |
|---|---|---|
| stem conv 3×3, stride 2 + BN | 16×112×112 | 176 |
| block 16→32 | 32×56×56 | 2,964 |
| block 32→64 | 64×28×28 | 10,536 |
| block 64→128 | 128×14×14 | 39,504 |
| block 128→256 | 256×7×7 | 152,736 |
| global average pool + dropout | 256 | 0 |
| fc | 2 | 514 |
| **total** | | **206,430** |

![ArogyaNet stages](assets/arch.png)

Each block has two depthwise-separable convolutions, a squeeze-and-excitation (SE) module and a 1×1-convolution skip connection. The network also uses depthwise-separable convolutions and global average pooling, but those were not ablated.

**Hypotheses** (from `hypothesis.md`, written before the results):

> 1. Skip connections: with about 8 conv layers and BatchNorm, skips make training stable and faster, but change ROC-AUC only a little (under 0.005), probably inside seed noise.
> 2. Squeeze-and-excitation: channel re-weighting helps the model focus on informative feature maps, giving a small ROC-AUC gain (about 0.005) that may not exceed seed spread.

**Ablations** (3 seeds; ΔROC-AUC = full minus ablated):

| ablation | ROC-AUC full | ROC-AUC ablated | ΔROC-AUC | accuracy ablated (full 0.944) |
|---|---|---|---|---|
| remove SE | 0.987 ± 0.001 | 0.987 ± 0.004 | −0.0006 ± 0.0040 | 0.952 ± 0.008 |
| remove skip | 0.987 ± 0.001 | 0.989 ± 0.001 | −0.0019 ± 0.0005 | 0.954 ± 0.009 |

**Both hypotheses failed.** Removing SE changes nothing outside the noise. Removing skips slightly improved ROC-AUC and accuracy, and did so for all three seeds. The size of the effect (under 0.005) matched hypothesis 1, but the direction did not, and I did not measure training speed. A likely reason: the network is shallow, so skips are not needed, and SE has little to recalibrate when the useful signal is global.

ArogyaNet (0.944, ROC-AUC 0.987) is **below HOG + LR (0.962, 0.990) and pretrained ResNet18 (0.976, 0.995)** on the same fold. It ties ResNet18 from scratch on ROC-AUC at 54 times fewer parameters.

## 8. Two-branch experiment (`train.py`)

Image branch plus a small dense branch for (mean brightness, aspect ratio, log area):

| model | acc | ROC-AUC |
|---|---|---|
| image only | 0.944 ± 0.008 | 0.987 ± 0.001 |
| image + metadata | 0.944 ± 0.023 | 0.984 ± 0.010 |

ΔROC-AUC (meta minus image) = −0.0026 ± 0.0104: no benefit, and a bigger spread.

**Link to the shortcut hunt.** Metadata alone reaches ROC-AUC 0.923, so the information is in the data. Adding it to the CNN gains nothing, so either the pixels already carry it or the network ignores it. This test cannot tell which. Had it helped, the model would be leaning on one hospital's scanner and export habits, which would not hold at a camp with other equipment. The shipped model is image-only.

## 9. Two ablations (`train.py`)

| setting | acc | ROC-AUC |
|---|---|---|
| 224 px, augmentation (reference) | 0.944 ± 0.008 | 0.987 ± 0.001 |
| **96 px**, augmentation | 0.953 ± 0.001 | 0.987 ± 0.002 |
| 224 px, **no augmentation** | 0.942 ± 0.004 | 0.978 ± 0.004 |

* **Resolution:** 96 px matches 224 px with 5.4 times fewer pixels. Fine detail is not what the model uses, which fits the shortcut result.
* **Augmentation:** it adds 0.009 ROC-AUC (about twice the seed spread). Without it, training loss falls to about 0.03 while validation loss gets worse within 3 epochs.

**The medically wrong augmentation is the flip, vertical above all.** My policy is rotation ±10° and zoom ±10%, with no flips. A vertical flip puts the diaphragm and gastric bubble at the top, an anatomy no patient has. A horizontal flip mirrors the body: the heart appears on the right (the pattern of dextrocardia) and the L/R marker is reversed, and both heart shape and lobe location are left-right cues. To be exact, the "pneumonia" label itself mostly does not depend on left or right, so the damage is largest for vertical flips and large rotations. My first draft had a horizontal flip; I removed it before any result in this report.

## 10. Reproducing the failure pattern (`rohit_sim.py`)

Same architecture, three evaluation protocols, three seeds:

| protocol | acc | bal. acc | sensitivity | specificity | ROC-AUC |
|---|---|---|---|---|---|
| Kaggle train folder → Kaggle test folder | 0.790 ± 0.009 | 0.720 ± 0.012 | 0.997 ± 0.003 | 0.443 ± 0.024 | 0.941 ± 0.011 |
| random split of all images (leaky) | 0.952 ± 0.007 | 0.938 ± 0.006 | 0.968 ± 0.008 | 0.907 ± 0.007 | 0.984 ± 0.002 |
| grouped split (ours) | 0.970 ± 0.005 | 0.968 ± 0.003 | 0.971 ± 0.009 | 0.965 ± 0.010 | 0.996 ± 0.001 |

On the official test folder the model finds nearly every sick child but calls 56% of healthy children pneumonia. That is close to the pilot's pattern (7 in 10 healthy children flagged). The leaky split did not score higher than the grouped one, so leakage (26 near-duplicate pairs) is real but is not the main explanation. The two test sets also differ in composition. The 98.4% was not reproduced under any protocol.
