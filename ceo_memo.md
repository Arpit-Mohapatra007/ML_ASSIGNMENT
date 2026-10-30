# Memo to the CEO: the MediScan audit

**From:** ML engineer, Arogya Diagnostics
**About:** what went wrong, what we built, and why "99% accuracy" is the wrong target

## The short version

1. Two of your three instructions are right. The audit explains what was wrong with the old model. The new model fits the camp laptop easily: **0.87 MB** against a 25 MB limit, and **2.2 ms** per image against 200 ms. The 2.2 ms is the model only, on CPU with 2 threads. Time the whole path (open image, resize, predict) on the real laptop before sign-off.
2. "99% accuracy" is the wrong target. Accuracy hides the error that matters, a missed sick child.
3. The new model works well on data like what it learned from, but **it is not proven to work at other hospitals**. On a second hospital's X-rays it raised far too many false alarms. Do not switch it on at a new site until it has been checked there.

## 1. What Rohit's 98.4% was

I cannot reproduce it, and nobody can: no code, no split record, no saved model. What the evidence shows:

* His validation folder had **16 images**. A perfect score on 16 images is compatible with a true accuracy of **79%** (95% interval 79% to 100%).
* Training our model on Kaggle's own train folder and testing on its test folder gave **79% accuracy**, not 98%. It caught 99.7% of sick children but called 56% of healthy children pneumonia (specificity 44%). That is the pilot's failure pattern. A random split of all images gave 95%, still not 98%. So I can show that 98.4% is not what this data supports, but not how he got it.
* The data contain shortcuts. A 16×16 thumbnail scores 94% accuracy. Three numbers about file size and brightness score 86%, with no lung content. When I change brightness and contrast on our own test images, specificity falls from 95% to anywhere between 0.4% and 92% depending on the change (54% when images are darkened by 0.2, 71% when contrast is cut by 30%), while sensitivity stays at or above 97%. This shows the model is sensitive to such changes. It does not prove that is what happened at the two hospitals.

## 2. What the new model achieves

Test set of 976 images, never used for any decision. The cut-off was chosen to minimise expected cost at the camp (a missed pneumonia costs 20 false alarms, camp prevalence 10%):

| measure | value (95% interval) |
|---|---|
| sensitivity (sick children found) | 98.5% (97.5 to 99.3) |
| specificity (healthy children cleared) | 95.1% (92.4 to 97.6) |
| accuracy | 97.5% (96.6 to 98.5) |
| ROC-AUC | 0.995 (0.991 to 0.998) |

Five-fold cross-validation gave lower and less stable specificity, **90.2% ± 5.1**, so plan on 90 to 95%.

**Per 1,000 children at the camp (100 sick, 900 healthy):**

| | missed sick | false alarms | alarms that are real | accuracy |
|---|---|---|---|---|
| always say NORMAL | 100 | 0 | n/a | 90.0% |
| new model, our test data | 1.5 | 44.3 | 69% | 95.4% |
| new model, **other hospital** | 4.2 | 352.9 | 21% | 64.3% |
| retrained with brightness/contrast variation, our test data | 3.5 | 30.7 | 76% | 96.6% |
| retrained with brightness/contrast variation, other hospital | 4.5 | 169.1 | 36% | 82.6% |

The camp tool has a **"low confidence, refer to a clinician" band**: 6.5% of test images fall in it, and accuracy on the rest is 98.5%. It rejects colour photos, blank images, random noise, a greyscale dog photo, and anything else that does not look like a child's chest X-ray. It wrongly rejects 0.5% of real X-rays.

## 3. What it must not be used for

* Adults, babies under one, or children over five. The training data are 1 to 5 year olds from one hospital.
* Any new hospital, scanner or camp without a local check first (specificity fell from 95% to about 61%).
* The only reason to send a child home. Sensitivity is high but not 100%.
* Photos of films or screens, or anything that is not a chest X-ray.
* Telling bacterial from viral pneumonia, or quoting its score as a clinical risk. The score is not calibrated to camp prevalence.

## 4. Why "99% accuracy" is the wrong target

At a camp where 10% of children are sick, **a tool that always says "NORMAL" scores 90%** and finds nobody. A 99% target allows 10 errors per 1,000 children, and all 10 can be missed sick children (10% of the sick). Cost makes this clear. A miss costs 20 and a false alarm costs 1:

| tool | cost per 1,000 children |
|---|---|
| 99% accurate, all 10 errors are misses | **200** |
| 99% accurate, all 10 errors are false alarms | 10 |
| **our model, our test data (95.4% accurate)** | **75.2** |
| retrained variant, our test data | 100.9 |
| our model, other hospital | 437.7 |
| retrained variant, other hospital | 259.8 |
| refer everyone | 900 |
| always NORMAL (90% accurate) | 2,000 |

A 95.4% accurate model costs about 2.7 times less than a 99% accurate one whose errors are misses. So accuracy does not rank tools; what matters is which errors are made. And nobody should promise 99% when a second hospital drops specificity to 61% (81% for the retrained variant).

**Ask for this instead.** At each site, on at least a few hundred locally labelled images and at that site's prevalence, report:

* sensitivity: at least 97%, with the lower end of its 95% interval at least 95%;
* specificity: at least 90%;
* the share of alarms that are real;
* expected cost per child: at most 0.15 (at 20 to 1).

Our model meets these on our own test data (98.5%, lower end 97.5%; 95.1%; cost 0.075) and fails them at the other hospital (specificity 61%, cost 0.438). Do not deploy at a site that misses them.

## 5. Your three instructions

1. **Proof of what Rohit did wrong:** done (section 1 and `Part1_audit.md`). Data problems: a 16-image validation set, image size and brightness that give away the class, 26 near-duplicate train/test pairs, same-patient images across splits. Process problems: no split record, no outside test, accuracy as the only number.
2. **99% accuracy:** declined (section 4).
3. **200 ms and 25 MB:** met with a wide margin. Rohit's ResNet50 (about 98 MB) would not have fit the update channel.

## 6. Next steps

* Run a pilot at one site with a local check before camp use, and compare both models by cost at that site's prevalence. The retrained variant is much more robust at the other hospital (specificity 81% against 61%, cost 0.260 against 0.438 per child) but is somewhat worse on our own data (sensitivity 96.5% against 98.5%, cost 0.101 against 0.075). Neither is a clear winner until it is checked locally.
* Collect a few hundred labelled images per new site to set that site's cut-off.
* Time the full path on the real camp laptop.
