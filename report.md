# The MediScan Audit: technical report

## Summary

Arogya Diagnostics wants to screen children's chest X-rays for pneumonia at rural camps, on a laptop with no GPU and no internet. A former intern's model ("ResNet50, 98.4% accuracy") failed a pilot: 7 of 10 healthy children were called pneumonia and two sick children were missed. This report audits the old data, rebuilds the model, tests it honestly, and says what the company should and should not do with it.

* **The old data cannot support 98.4%.** The 16-image validation set cannot tell 100% from 79%. On Kaggle's own train and test folders our model gets 79% accuracy with specificity 44%. Image size and brightness alone predict the class (ROC-AUC 0.92), and a 16×16 thumbnail scores 94% accuracy.
* **The rebuilt model** (ArogyaNet: 206k parameters, 0.87 MB, 2.2 ms on CPU) gets 97.5% accuracy, 98.5% sensitivity and 95.1% specificity on a held-out test set. It does not beat a simple HOG + logistic-regression baseline on the validation fold, and neither design choice I tested helped.
* **It does not carry over to another hospital.** ROC-AUC fell from 0.995 to 0.876 and specificity to 61%. Brightness/contrast augmentation recovered part of it (ROC-AUC 0.953, specificity 81%).
* **"99% accuracy" is the wrong target.** At camp prevalence, always saying NORMAL scores 90%, and a 99% accurate tool can still miss one in ten sick children.

Every number comes from a script in the repository (map in section 14). Part 1 detail is in `Part1_audit.md`; the CEO memo is `ceo_memo.md`.

## 1. Data audit

There are 5,856 images, all greyscale. NORMAL images are larger (2.42 vs 1.05 megapixels), squarer (aspect 1.25 vs 1.51) and have higher contrast (std 61 vs 55), at the same average brightness (122.6 vs 122.8). These are export and acquisition properties, not lung properties.

![Figure 1. Pixel statistics by class](assets/pixel_stats.png)

Duplicates: 32 exact duplicates and 265 near-duplicate pairs (perceptual hash, Hamming distance ≤4). Of these, 238 lie inside one split and 26 cross train and test. Patient IDs from the Kaggle filenames plus the near-duplicate links give 2,592 groups, the largest with 188 images.

![Figure 2. The worst five train/test pairs](assets/worst_pairs.png)

Validation: 16 out of 16 correct has a 95% interval of 0.794 to 1.000. The replacement is a grouped, stratified design: a 976-image test set never used for decisions, plus five grouped folds on the rest. One split alone would chase noise, and k-fold alone leaves no untouched estimate.

## 2. What Rohit did wrong: evidence

Same architecture (ArogyaNet), three evaluation protocols, three seeds (`rohit_sim.py`):

| protocol | acc | bal. acc | sensitivity | specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Kaggle train folder → Kaggle test folder | 0.790 ± 0.009 | 0.720 ± 0.012 | 0.997 ± 0.003 | 0.443 ± 0.024 | 0.941 ± 0.011 | 0.958 ± 0.008 |
| random split of all images (leaky) | 0.952 ± 0.007 | 0.938 ± 0.006 | 0.968 ± 0.008 | 0.907 ± 0.007 | 0.984 ± 0.002 | 0.994 ± 0.001 |
| grouped split (ours) | 0.970 ± 0.005 | 0.968 ± 0.003 | 0.971 ± 0.009 | 0.965 ± 0.010 | 0.996 ± 0.001 | 0.998 ± 0.000 |

1. The official test folder is much harder than a split inside the pool (79% vs 97% for the same model), so Kaggle's train and test folders are not alike. On that folder the model finds 99.7% of sick children but calls 56% of healthy ones pneumonia, which is the pilot's failure pattern (pilot specificity 30%). Always predicting pneumonia scores 62.5% there, so 79% is only 16.5 points above doing nothing.
2. **The leaky split did not score higher than the grouped one** (95.2% vs 97.0%). I expected leakage to inflate the score, and it did not. Only 26 near-duplicate pairs cross train and test, and the two test sets differ in makeup, so leakage is real but small. It is not the main explanation.
3. 98.4% was not reproduced under any protocol. I cannot say how it was obtained. The 16-image validation set is the most direct candidate.

**Shortcuts** (`baselines.py`, validation fold 0, chance accuracy 0.730):

| model | acc | bal. acc | ROC-AUC |
|---|---|---|---|
| majority class | 0.730 | 0.500 | 0.500 |
| 3 metadata numbers (mean, aspect, area) | 0.864 | 0.823 | 0.923 |
| 16×16 pixels + logistic regression | 0.938 | 0.911 | 0.976 |
| HOG + logistic regression | 0.962 | 0.949 | 0.990 |

A 16×16 thumbnail has no texture, yet it separates the classes. So the data can be largely solved without reading lung texture.

**Sensitivity to brightness and contrast** (`shift_sim.py`: final model, internal test set, cut-off fixed at 0.47). Brightness is in units of the 0-to-1 pixel range; contrast is a multiplier.

| change | specificity | sensitivity |
|---|---|---|
| none | 0.951 | 0.985 |
| brightness −0.2 | 0.542 | 0.994 |
| contrast ×0.7 | 0.705 | 0.994 |
| brightness −0.1 and contrast ×0.7 | 0.076 | 0.997 |
| brightness +0.2 and contrast ×1.3 | 0.212 | 0.989 |

Specificity can collapse while sensitivity stays near 1.0. That is the pilot's pattern. This test is artificial: it shows the weakness exists, not that it caused the pilot failure.

## 3. Models (Part 1)

Validation fold 0, three seeds, mean ± std.

| model | acc | bal. acc | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| ResNet18, ImageNet weights | 0.976 ± 0.003 | 0.967 ± 0.008 | 0.995 ± 0.001 | 0.998 ± 0.000 |
| ResNet18, scratch | 0.954 ± 0.009 | 0.937 ± 0.006 | 0.988 ± 0.006 | 0.995 ± 0.002 |
| ArogyaNet (full) | 0.944 ± 0.008 | 0.936 ± 0.003 | 0.987 ± 0.001 | 0.995 ± 0.000 |
| ArogyaNet, no SE | 0.952 ± 0.008 | 0.934 ± 0.028 | 0.987 ± 0.004 | 0.995 ± 0.001 |
| ArogyaNet, no skip | 0.954 ± 0.009 | 0.934 ± 0.024 | 0.989 ± 0.001 | 0.996 ± 0.000 |
| ArogyaNet + metadata branch | 0.944 ± 0.023 | 0.940 ± 0.017 | 0.984 ± 0.010 | 0.993 ± 0.004 |
| ArogyaNet, 96 px | 0.953 ± 0.001 | 0.931 ± 0.005 | 0.987 ± 0.002 | 0.995 ± 0.001 |
| ArogyaNet, 224 px, no augmentation | 0.942 ± 0.004 | 0.922 ± 0.016 | 0.978 ± 0.004 | 0.991 ± 0.003 |

**Transfer.** I expected ImageNet's low-level filters to carry over despite the colour-to-greyscale gap. They did: pretraining adds 2.2 points of accuracy and 0.007 ROC-AUC. ResNet18 is about 45 MB, too big for the 25 MB channel.

**ArogyaNet** (206,430 parameters, Figure 3): a stem convolution, four blocks, global average pooling, dropout and a 2-way layer. Each block has two depthwise-separable convolutions, a squeeze-and-excitation (SE) module and a skip connection.

![Figure 3. ArogyaNet stages](assets/arch.png)

My hypotheses, committed before any result: skips make training stable with a small ROC-AUC change (under 0.005), and SE gives a small gain (about 0.005). **Both failed.** The change in ROC-AUC (full minus ablated) is −0.0006 ± 0.0040 for SE and −0.0019 ± 0.0005 for skips, so removing either did not hurt, and removing skips helped slightly. The network is shallow, so skips are not needed, and SE has little to fix when the useful signal is global. ArogyaNet is below HOG + LR and below pretrained ResNet18 on this fold. Its advantage is size: 54 times fewer parameters than ResNet18, with ROC-AUC equal to ResNet18 from scratch.

**Two-branch version.** Adding mean brightness, aspect ratio and log area gave −0.0026 ± 0.0104 ROC-AUC: no help. Metadata alone reaches 0.923, so the information exists, but the CNN gains nothing from it. If it had helped, the model would be using one hospital's scanner and export habits, which would not hold at a camp. The shipped model uses the image only.

**Resolution and augmentation.** 96 px matches 224 px (ROC-AUC 0.987 for both) with 5.4 times fewer pixels, so fine detail is not what the model uses. Augmentation (rotation ±10°, zoom ±10%) adds 0.009 ROC-AUC. **The medically wrong augmentation is the flip, vertical above all.** A vertical flip puts the diaphragm and gastric bubble at the top, and a horizontal flip puts the heart on the right (the dextrocardia pattern) and swaps the L/R marker. The pneumonia label itself mostly does not depend on left or right, so the damage is largest for vertical flips and large rotations. No flips are used.

## 4. Imbalance, threshold and cost (Part 2.1)

I compared a class-weighted model with threshold moving on the unweighted model (validation fold 0, `part2.py`):

| threshold | unweighted: sensitivity / specificity | weighted: sensitivity / specificity |
|---|---|---|
| 0.1 | 0.990 / 0.826 | 0.997 / 0.409 |
| 0.3 | 0.962 / 0.920 | 0.985 / 0.742 |
| 0.5 | 0.928 / 0.951 | 0.952 / 0.864 |
| 0.7 | 0.877 / 0.966 | 0.907 / 0.924 |
| 0.9 | 0.757 / 0.996 | 0.795 / 0.962 |
| ROC-AUC | 0.986 at every threshold | 0.970 |

Class weighting **changes the model**: ROC-AUC drops from 0.986 to 0.970 and its probabilities correlate only 0.92 with the unweighted model's. Threshold moving **changes only the decision**: ROC-AUC is the same at every threshold, and sensitivity and specificity just slide along one curve. The weighted model at 0.5 (0.952 / 0.864) is worse than the unweighted model at 0.3 (0.962 / 0.920). This is one seed, and the weighted run was unstable (validation loss spiked to 1.65 at epoch 5), so treat it as a direction, not a precise number.

**Cost matrix.** A missed pneumonia costs 20, a false alarm costs 1, and prevalence at the camp is 10% (9 to 1 NORMAL). Expected cost per patient = 20 × 0.1 × (1 − sensitivity) + 0.9 × (1 − specificity). On validation the best cut-off for the unweighted model is 0.231 (sensitivity 0.979, false-positive rate 0.087, cost 0.1205 per patient, 55.5% of alarms real at camp prevalence). Never screening costs 2.0 per patient and referring everyone 0.9. For the final model the cut-off is **0.470**, chosen on validation. It is marked as the red dot in Figure 4.

![Figure 4. ROC and precision-recall curves with the chosen point (validation, first model)](assets/roc_pr.png)

## 5. Evaluation (Part 2.2)

Final model on the held-out test set (976 images, cut-off 0.470): 701 true positives, 251 true negatives, 13 false positives, 11 false negatives. The intervals come from 1,000 bootstrap resamples.

| measure | value | 95% interval |
|---|---|---|
| accuracy | 0.975 | 0.966 to 0.985 |
| precision | 0.982 | 0.972 to 0.992 |
| sensitivity (recall) | 0.985 | 0.975 to 0.993 |
| specificity | 0.951 | 0.924 to 0.976 |
| F1 | 0.983 | 0.977 to 0.989 |
| ROC-AUC | 0.995 | 0.991 to 0.998 |
| PR-AUC | 0.998 | 0.997 to 0.999 |

![Figure 5. Confusion matrix on the test set](assets/confusion.png)

![Figure 6. ROC and precision-recall curves, final model on the test set](assets/roc_pr_final.png)

The precision-recall curve uses the test set's 73% pneumonia, so its precision (about 0.98) is not the camp precision (0.69).

**Five-fold cross-validation** (grouped, stratified; a new model and a new cut-off inside each fold): accuracy 0.952 ± 0.008, precision 0.965 ± 0.018, sensitivity 0.970 ± 0.020, specificity 0.902 ± 0.051, F1 0.967 ± 0.006, ROC-AUC 0.986 ± 0.003, PR-AUC 0.994 ± 0.001. Specificity is lower and less stable than on the test set, so quote 90 to 95%.

**Why accuracy alone is the wrong headline.** With 73% pneumonia in the data (10% at the camp), always predicting the majority class already scores 73% (90% at camp prevalence), so the number says little about the error that matters. Accuracy also counts a missed sick child the same as a false alarm, though a miss is about 20 times worse.

## 6. Tuning and overfitting (Part 2.3)

Random search with a stated budget: 12 trials of 8 epochs, seed 0, over learning rate (1e-4 to 1e-2, log scale), batch size {16, 32, 64}, dropout {0, 0.2, 0.5} and weight decay {0, 1e-4, 1e-2}. The best was learning rate 0.0014, batch size 16, dropout 0, weight decay 1e-4 (validation ROC-AUC 0.9877). The top four trials span only 0.9862 to 0.9877, less than seed-to-seed noise, so the search found a good region, not a clear winner. Because dropout 0 won, the "regularised" run uses augmentation and weight decay.

![Figure 7. Training and validation loss: unregularised vs regularised](assets/curves.png)

The unregularised run (no augmentation, weight decay or dropout; 25 epochs, no early stopping) reaches its best validation loss, 0.155, at **epoch 3**. **Overfitting begins at epochs 3 to 4**: training loss keeps falling (0.128 to 0.067) while validation loss rises (0.155 to 0.171). By epoch 25 training loss is 0.004 and validation loss 0.256, which is 65% above its best. The regularised run reaches its best validation loss, 0.102 (34% lower), at epoch 14, and early stopping (patience 5) ends it at epoch 19. Early stopping would have spared the unregularised run that 65% rise, and saved six epochs on the regularised one. Validation loss is noisy throughout (spikes above 1), so the chosen epoch is somewhat arbitrary. A learning-rate schedule is the obvious next fix.

## 7. Ensembles (Part 2.4)

The soft vote averages seven Part 1 models: three ArogyaNet seeds, the no-SE, no-skip and metadata variants, and the pretrained ResNet18. Random Forest and AdaBoost are trained on the 256-number embedding from one ArogyaNet seed's last hidden layer. Test set, cut-off 0.5, no member chosen from it.

| name | acc | bacc | rec | spec | roc | pr |
|---|---|---|---|---|---|---|
| arogya_full_0 | 0.952 | 0.959 | 0.944 | 0.973 | 0.993 | 0.997 |
| arogya_full_1 | 0.967 | 0.962 | 0.973 | 0.951 | 0.995 | 0.998 |
| arogya_full_2 | 0.961 | 0.957 | 0.966 | 0.947 | 0.994 | 0.998 |
| resnet18_imagenet_0 | 0.981 | 0.980 | 0.982 | 0.977 | 0.998 | 0.999 |
| arogya_no_se_0 | 0.966 | 0.968 | 0.963 | 0.973 | 0.995 | 0.998 |
| arogya_no_res_0 | 0.963 | 0.945 | 0.985 | 0.905 | 0.995 | 0.998 |
| arogya_meta_0 | 0.972 | 0.967 | 0.979 | 0.955 | 0.996 | 0.999 |
| soft_vote | 0.972 | 0.969 | 0.976 | 0.962 | 0.997 | 0.999 |
| rf_on_embed | 0.959 | 0.952 | 0.968 | 0.936 | 0.993 | 0.997 |
| ada_on_embed | 0.959 | 0.950 | 0.969 | 0.932 | 0.992 | 0.997 |

The soft vote (ROC-AUC 0.9971, accuracy 0.972) beats every ArogyaNet member on ROC-AUC and ties the best on accuracy. **It does not beat its best member, the pretrained ResNet18 (0.9982, 0.981).** Six of the seven members are variants of one small network trained on the same data, so their errors are strongly linked and averaging them removes little. Mixing one stronger model with six weaker ones pulls the result toward the weaker group. ArogyaNet seeds alone differ by 1.5 points of test accuracy (0.952 to 0.967), as much as any voting gain, so differences this small are not established with 976 images. Random Forest and AdaBoost on the embedding (ROC-AUC 0.993 and 0.992) do not improve on the network's own output layer (0.993 for the same seed). A plausible reason is that the embedding is already almost linearly separable and the forests learn from training embeddings the network has already fitted. ResNet18 is the strongest single model on this test set, but it is about 45 MB, over the 25 MB channel.

## 8. Feature space (Part 2.5)

PCA and t-SNE of the final model's embeddings on the test set, coloured by true class and by correct or wrong (Figure 8). K-means with 2 clusters matches the labels with purity 0.921 and adjusted Rand index 0.703, so the classes separate well even without labels. That fits the earlier finding that simple global cues separate them.

![Figure 8. PCA and t-SNE of test embeddings](assets/embeddings.png)

The 24 errors (11 missed pneumonia, 13 false alarms) average 2.13 Mpx and aspect ratio 1.29, against 1.44 Mpx and 1.44 for correct images, with mean brightness 117.8 against 123.4. The wrong images lean toward the NORMAL class's size profile (2.42 Mpx, 1.25), which is consistent with the model using size-like cues. With 24 errors this is a description, not a test.

## 9. Another hospital (Part 2.6)

The COVID-19 Radiography Database (3,300 randomly chosen Normal images and all Viral Pneumonia images) was evaluated once by `external.py`. Each model uses its own cut-off (0.470 and 0.658).

| name | acc | bacc | rec | spec | prec | roc | pr |
|---|---|---|---|---|---|---|---|
| internal test, final | 0.975 | 0.968 | 0.985 | 0.951 | 0.982 | 0.995 | 0.998 |
| external, final | 0.709 | 0.783 | 0.958 | 0.608 | 0.499 | 0.876 | 0.689 |
| internal test, bright/contrast aug | 0.965 | 0.965 | 0.965 | 0.966 | 0.987 | 0.996 | 0.999 |
| external, bright/contrast aug | 0.853 | 0.883 | 0.955 | 0.812 | 0.674 | 0.953 | 0.880 |

![Figure 9. Mean image brightness, internal vs external](assets/shift.png)

At camp prevalence (1,000 children, 100 sick), the original model misses 4.2 and raises 352.9 false alarms, so 21% of alarms are real. The retrained variant misses 4.5 and raises 169.1 (36% real). The drop is almost all false alarms, as in the pilot.

**Why, using the Part 1 statistics.** The training data are children from one hospital, whose two classes differ in image size and contrast (section 1). The external images are a different population (probably adults, not checked), scanner and export, resized to the same 224×224 input. The size and shape cues available at home are absent or misleading. Mean brightness moved only from 122.8 to 128.0, so a plain brightness offset does not explain the drop. The artificial test (section 2) shows the model is far more sensitive to contrast, and the contrast fix recovered much of the lost specificity. I did not log the native size or shape of the external files, and I did not re-run the external set to measure more.

**The fix, honestly.** I tried one fix: retraining with stronger brightness and contrast augmentation (one seed, its own cut-off chosen on internal validation). It helped: ROC-AUC 0.876 to 0.953, specificity 0.608 to 0.812, precision 0.499 to 0.674, with sensitivity unchanged (0.958 to 0.955). It does not close the gap (internal ROC-AUC is 0.996), and it costs something at home. On the internal test set sensitivity fell from 0.985 to 0.965 and specificity rose from 0.951 to 0.966, so at camp prevalence it misses 3.5 sick children per 1,000 instead of 1.5 (expected cost 0.101 per child against 0.075). Whether that is worth it depends on how much a site resembles the training hospital.

## 10. Shipping it (Part 2.7)

`predict.py` takes an image path and prints a class and a score. It works in this order:

1. Reject colour photos (mean saturation above 25) and blank images (intensity spread below 12).
2. Reject images whose embedding is far from the training data (distance cut-off at the 99.5th percentile of validation images).
3. Below 0.470: NORMAL. From 0.470 to 0.929: **"LOW CONFIDENCE, refer to a clinician"**. Above 0.929: PNEUMONIA. The upper limit is the first score at which at least 80% of alarms are real at camp prevalence, on validation.

On the held-out test set, 6.5% of images fall in the refer band, accuracy outside the band is 98.5% (908 images), and 85.7% of band images are pneumonia. 0.5% of real X-rays (5 of 976) are wrongly rejected. A colour dog photo, a blank image, random noise and the same dog in greyscale are all rejected (`ood_check.py`). The greyscale dog is stopped by the embedding check, which the colour filter cannot do.

The model file is 0.87 MB. Inference takes 2.2 ms (median; 95th percentile 2.25 ms) on CPU with 2 threads, not counting image decoding. The Gradio demo is `demo.py`. The score is a ranking score, not a probability calibrated to camp prevalence.

## 11. Why 99% accuracy is the wrong target

Take 1,000 children at camp prevalence (100 sick). Always saying NORMAL scores 90% and finds nobody. A 99% target allows 10 errors, and they can all be missed sick children (cost 200 on the 20-to-1 matrix) or all false alarms (cost 10), so accuracy does not rank the tools. Our model at 95.4% accuracy costs 75.2. The company should ask for these, measured per site on local data at that site's prevalence: sensitivity of at least 97% (lower end of the 95% interval at least 95%), specificity of at least 90%, the share of alarms that are real, and expected cost per child of at most 0.15. Our model meets them on its own test data and fails them at the other hospital. The full memo is `ceo_memo.md`.

## 12. Limits of this work

* **Test set use.** Every decision (hyperparameters, cut-off, model choice) used validation folds or cross-validation. The test set was read only for reporting, by `part2.py` (one model, ensembles, embeddings), `final_eval.py`, `ceo_math.py` and `ood_check.py`.
* **External set.** The external data were used only for evaluation, never for training or tuning. An earlier version of the pipeline (before patient-ID grouping, and before the horizontal flip was removed) ran `external.py` once, and its output was discarded. The reported run came after a full clean re-run. The first attempt of the final pipeline crashed with a NameError before printing anything. ⟦CONFIRM before submitting: state whether any design or hyperparameter decision was changed because of the earlier external numbers; if so, say what⟧
* **Choice after seeing external results.** Recommending a site pilot that includes the brightness/contrast variant follows from external results I had already seen, so its external numbers are a little optimistic as a forecast.
* **Noise.** Three seeds, a very jumpy validation loss, and one seed for the tuning search and for the class-weight comparison. Differences below about 0.005 ROC-AUC should not be read as effects.
* **ArogyaNet is not the strongest model.** HOG + LR and pretrained ResNet18 beat it on the validation fold. On the test set, untuned ResNet18 (seed 0, cut-off 0.5) scores ROC-AUC 0.998 against 0.995. ResNet18 is over the 25 MB limit. Int8 quantisation might fit it, but I did not test that, and I did not measure its external performance.
* **External population.** The external pneumonia cases are probably adults, so part of the drop is a change of population that this study cannot separate from scanner and contrast effects.
* **Calibration.** Scores are not calibrated to camp prevalence.

## 13. Reproducing the work

Python 3.12 with the versions pinned in `requirements.txt`; run on an HP Elite Tower 800 G9 desktop. Seeds are set for Python, NumPy, PyTorch and CUDA (deterministic cuDNN) and printed at the start of each log. `make all` runs `run_all.sh` and writes one log per script to `logs/`. Timings are in the log file timestamps. Do not run it twice with a clean start: `external.py` is meant to run once, and a guard file (`reports/external_done`) stops a second run.

## 14. Where each result comes from

| result | script | output |
|---|---|---|
| inventory, pixel statistics, duplicates, worst pairs | `audit.py` | `pixel_stats.png`, `worst_pairs.png`, `inventory.csv` |
| validation interval, grouped split, folds | `split.py` | `split.csv` |
| shortcuts, baselines | `baselines.py` | `baselines.csv` |
| transfer, ArogyaNet, ablations, two-branch, resolution, augmentation | `train.py` | `part1_models.csv`, `arch.txt`, `arch.png` |
| Rohit's protocols | `rohit_sim.py` | `rohit_repro.csv` |
| imbalance vs threshold, cost, cross-validation, tuning, ensembles, embeddings | `part2.py` | `roc_pr.png`, `curves.png`, `ensembles.csv`, `embeddings.png`, `chosen.json` |
| final test metrics and intervals | `final_eval.py` | `final_test_metrics.csv`, `roc_pr_final.png`, `confusion.png` |
| brightness and contrast sensitivity | `shift_sim.py` | `shift_sim.csv` |
| other hospital | `external.py` | `external.csv`, `shift.png` |
| model file, speed, rejection, refer band | `ship.py`, `ood_check.py` | `arogya.pt`, `ood.npz`, `cfg.json` |
| camp arithmetic | `ceo_math.py` | `logs/ceo_math.txt` |

## Appendix: full tables

Cut-off 0.5 unless stated. `baselines` is validation fold 0; `part1_models` is validation fold 0, mean ± std over three seeds; `rohit_repro` is mean ± std over three seeds; `external` uses each model's own cut-off; `ensembles` is the test set.

### baselines

| name | acc | bacc | prec | rec | spec | f1 | roc | pr |
|---|---|---|---|---|---|---|---|---|
| 16x16 pixels + LR | 0.938 | 0.911 | 0.947 | 0.969 | 0.852 | 0.958 | 0.976 | 0.990 |
| meta only (mean, ar, area) | 0.864 | 0.823 | 0.903 | 0.912 | 0.735 | 0.907 | 0.923 | 0.972 |
| majority | 0.730 | 0.500 | 0.730 | 1.000 | 0.000 | 0.844 | 0.500 | 0.730 |
| HOG + LR | 0.962 | 0.949 | 0.971 | 0.978 | 0.920 | 0.974 | 0.990 | 0.996 |

### part1_models

| name | acc | bacc | prec | rec | spec | f1 | roc | pr |
|---|---|---|---|---|---|---|---|---|
| resnet18_imagenet | 0.976±0.003 | 0.967±0.008 | 0.981±0.008 | 0.986±0.007 | 0.948±0.022 | 0.983±0.002 | 0.995±0.001 | 0.998±0.000 |
| resnet18_scratch | 0.954±0.009 | 0.937±0.006 | 0.964±0.006 | 0.973±0.017 | 0.902±0.017 | 0.969±0.007 | 0.988±0.006 | 0.995±0.002 |
| arogya_full | 0.944±0.008 | 0.936±0.003 | 0.970±0.009 | 0.952±0.022 | 0.920±0.027 | 0.961±0.006 | 0.987±0.001 | 0.995±0.000 |
| arogya_no_se | 0.952±0.008 | 0.934±0.028 | 0.962±0.024 | 0.973±0.016 | 0.894±0.072 | 0.967±0.005 | 0.987±0.004 | 0.995±0.001 |
| arogya_no_res | 0.954±0.009 | 0.934±0.024 | 0.960±0.019 | 0.978±0.010 | 0.890±0.057 | 0.969±0.005 | 0.989±0.001 | 0.996±0.000 |
| arogya_meta | 0.944±0.023 | 0.940±0.017 | 0.974±0.003 | 0.950±0.029 | 0.931±0.008 | 0.961±0.016 | 0.984±0.010 | 0.993±0.004 |
| arogya_96 | 0.953±0.001 | 0.931±0.005 | 0.958±0.005 | 0.978±0.005 | 0.885±0.014 | 0.968±0.000 | 0.987±0.002 | 0.995±0.001 |
| arogya_224_noaug | 0.942±0.004 | 0.922±0.016 | 0.955±0.014 | 0.966±0.011 | 0.878±0.042 | 0.960±0.002 | 0.978±0.004 | 0.991±0.003 |

### rohit_repro

| name | acc | bacc | prec | rec | spec | f1 | roc | pr |
|---|---|---|---|---|---|---|---|---|
| kaggle_train_to_test | 0.790±0.009 | 0.720±0.012 | 0.749±0.008 | 0.997±0.003 | 0.443±0.024 | 0.856±0.005 | 0.941±0.011 | 0.958±0.008 |
| random_split_leaky | 0.952±0.007 | 0.938±0.006 | 0.966±0.003 | 0.968±0.008 | 0.907±0.007 | 0.967±0.005 | 0.984±0.002 | 0.994±0.001 |
| grouped_split | 0.970±0.005 | 0.968±0.003 | 0.987±0.003 | 0.971±0.009 | 0.965±0.010 | 0.979±0.003 | 0.996±0.001 | 0.998±0.000 |

### external

| name | acc | bacc | prec | rec | spec | f1 | roc | pr |
|---|---|---|---|---|---|---|---|---|
| internal test, final | 0.975 | 0.968 | 0.982 | 0.985 | 0.951 | 0.983 | 0.995 | 0.998 |
| external, final | 0.709 | 0.783 | 0.499 | 0.958 | 0.608 | 0.656 | 0.876 | 0.689 |
| internal test, bright/contrast aug | 0.965 | 0.965 | 0.987 | 0.965 | 0.966 | 0.976 | 0.996 | 0.999 |
| external, bright/contrast aug | 0.853 | 0.883 | 0.674 | 0.955 | 0.812 | 0.790 | 0.953 | 0.880 |

### ensembles

| name | acc | bacc | prec | rec | spec | f1 | roc | pr |
|---|---|---|---|---|---|---|---|---|
| arogya_full_0 | 0.952 | 0.959 | 0.990 | 0.944 | 0.973 | 0.966 | 0.993 | 0.997 |
| arogya_full_1 | 0.967 | 0.962 | 0.982 | 0.973 | 0.951 | 0.977 | 0.995 | 0.998 |
| arogya_full_2 | 0.961 | 0.957 | 0.980 | 0.966 | 0.947 | 0.973 | 0.994 | 0.998 |
| resnet18_imagenet_0 | 0.981 | 0.980 | 0.991 | 0.982 | 0.977 | 0.987 | 0.998 | 0.999 |
| arogya_no_se_0 | 0.966 | 0.968 | 0.990 | 0.963 | 0.973 | 0.977 | 0.995 | 0.998 |
| arogya_no_res_0 | 0.963 | 0.945 | 0.966 | 0.985 | 0.905 | 0.975 | 0.995 | 0.998 |
| arogya_meta_0 | 0.972 | 0.967 | 0.983 | 0.979 | 0.955 | 0.981 | 0.996 | 0.999 |
| soft_vote | 0.972 | 0.969 | 0.986 | 0.976 | 0.962 | 0.981 | 0.997 | 0.999 |
| rf_on_embed | 0.959 | 0.952 | 0.976 | 0.968 | 0.936 | 0.972 | 0.993 | 0.997 |
| ada_on_embed | 0.959 | 0.950 | 0.975 | 0.969 | 0.932 | 0.972 | 0.992 | 0.997 |
