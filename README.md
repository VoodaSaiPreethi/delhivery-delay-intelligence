# Delhivery Last-Mile Delay Intelligence Pipeline

Capstone project, AI \& ML Certification Program, E\&ICT Academy, IIT Roorkee.

One connected pipeline answers three operations questions:

1. **How long will a parcel take?** A delivery-hours regression (Part B).
2. **Which parcels will miss their promise?** A delay classifier with an alert threshold (Part C).
3. **Which hubs need which playbook?** K-Means hub segments, checked against the tier labels (Part D).

The data is synthetic and comes from `generate\_data.py`. No real Delhivery records are used. All money is in Indian Rupees (INR).

\---

## How to run

From a fresh clone, run these commands from the repository root, in this order:

```bash
pip install -r requirements.txt
python generate\_data.py          # prints: 24 1224 48  (re-creates data/hubs.csv, raw\_shipments.csv, new\_shipments.csv)
python part\_a\_clean\_eda.py       # writes data/modelling\_table.csv and outputs/part\_a\_\*.csv
python part\_b\_regression.py      # writes outputs/part\_b\_\*.csv
python part\_c\_classification.py  # writes outputs/part\_c\_\*.csv
python part\_d\_segments.py        # writes hub\_profile, part\_d\_separation, hub\_segments, segment\_profile, tier\_vs\_segment\_mismatch
python package\_model.py          # writes models/\*.pkl + models/\*.csv and outputs/part\_d\_reload\_check.csv
python predict\_delay.py          # writes outputs/new\_shipment\_actions.csv
```

**About the model file:** running `python package\_model.py` regenerates `models/delay\_model.pkl`. The committed pickle loads reliably only on the scikit-learn version pinned in `requirements.txt` (scikit-learn 1.9.1). If your installed versions differ, run `package\_model.py` again rather than loading the committed pickle.

Repository layout:

```
README.md  requirements.txt  generate\_data.py
data/      hubs.csv  raw\_shipments.csv  new\_shipments.csv  modelling\_table.csv
src/       shipment\_prep.py   (PROMISED\_HOURS, FEATURES, NOISE\_COLUMNS, clean\_shipments, build\_features)
part\_a\_clean\_eda.py  part\_b\_regression.py  part\_c\_classification.py  part\_d\_segments.py
package\_model.py  predict\_delay.py
models/    delay\_model.pkl  delay\_threshold.csv  feature\_order.csv  impute\_values.csv
outputs/   every CSV named in the brief
```

\---

## Data findings

Source files: `outputs/part\_a\_checks.csv`, `outputs/part\_a\_delay\_by\_group.csv` and `outputs/part\_a\_correlations.csv`.

The raw export had 1,224 rows. Removing 24 exact duplicates left **1,200 unique shipments**. Cleaning fixed 12 different `service\_type` spellings down to 3, filled 18 blank payment modes with `Unknown`, turned 5 sentinel weights (999 kg) into missing values, and filled the gaps with medians. Those medians are 314.95 km for distance, 1.83 kg for weight and INR 1,121.50 for declared value. In total, **278 parcels (delay rate 0.2317)** missed their promise.

1. **Service type.** Express is delayed most often, at a **delay rate of 0.4897** (119 of 243 parcels). Standard is at 0.2056 and Economy at 0.0857. Express has the tightest promise (36 hours), so it has the least slack.
2. **Hub tier.** Delay rate rises steeply with tier: **Tier 1 0.0400 → Tier 2 0.2150 → Tier 3 0.4400**. A Tier 3 parcel is 11 times as likely to be late as a Tier 1 parcel.
3. **Distance.** Delay rate grows with every distance band: **0-200 km 0.0219, 201-400 km 0.1368, 401-600 km 0.4364, 601+ km 0.8523**. More than 85% of parcels going over 600 km are late.
4. **Strongest correlations with delivery\_hours.** Ranked by absolute value, the top three are **promised\_hours 0.8372**, **distance\_km 0.4013** and **n\_stops 0.3247**. All three are positive. Note that service type drives the absolute hours, while distance is what drives *lateness*: corr(is\_delayed, distance\_km) = 0.5455.
5. **Why log-transform declared value.** On the filled column, declared\_value\_inr has a **mean of INR 1,533.05 against a median of INR 1,121.50**. The mean is 37% above the median, which shows a long right tail. After the natural log, the **mean is 7.0277 and the median is 7.0224**, almost identical. The transform makes the column roughly symmetric, so a few very expensive parcels cannot dominate scale-sensitive models.

Other checks: delivery\_hours has Q1 = 43.26 and Q3 = 72.56. That gives 1.5 × IQR bounds of −0.71 and 116.52, and only **1 outlier**. The final modelling table has **0 missing cells** across the 17 features and the target.

\---

## Model results (regression)

Source files: `outputs/part\_b\_\*.csv`. The split is 840 / 180 / 180 rows. The scaler was fitted on the training rows only. The scaled training columns have a maximum absolute mean of 0.0, while the scaled test columns have 0.2187. That non-zero test value proves the scaler never saw the test rows.

|model (validation)|MAE|RMSE|R²|
|-|-|-|-|
|mean\_baseline|15.7952|18.8552|−0.0212|
|linear\_regression|3.9222|**4.8912**|**0.9313**|
|gradient\_boosting|3.9166|4.9165|0.9306|
|random\_forest|4.2505|5.4680|0.9141|
|decision\_tree (depth 6)|5.0971|6.2796|0.8867|

* **Baseline.** Predicting the training mean for every parcel gives **R² = −0.0212** with an RMSE of 18.86 hours. A negative R² means it does slightly worse than the validation set's own mean. Any useful model has to beat this, and every candidate does by a wide margin.
* **Validation winner.** **Linear regression** has the lowest validation RMSE, **4.8912 hours** (R² 0.9313). Gradient boosting is a very close second at 4.9165. This fits the generator, which is mostly additive: base hours plus linear terms in distance, stops, weight and flags.
* **Tree depth: bias versus variance** (`part\_b\_tree\_depth.csv`). Depths 1–3 are **high-bias**. Both errors are large and close together; depth 1 has train 11.63 and validation 12.13, a gap of only 0.50. Depth 6 is the best, with a validation RMSE of 6.2796. From depth 8 onwards the trees are **high-variance**. Training error keeps falling while validation error rises: depth 12 has train 1.36 and validation 7.42, and the unlimited tree has train **0.00** and validation **7.79**, a gap of 7.79 hours. That tree has memorised the training rows.
* **Learning rate** (`part\_b\_gb\_learning\_rate.csv`). **0.1 is best (R² 0.9306)**. At **0.01 (R² 0.7127)** learning is *too slow*: 100 small steps never get close to the target, so the model underfits. At **1.0 (R² 0.8440)** each tree *overshoots*: it corrects the full residual, which includes noise, so the ensemble jumps around the optimum and overfits.
* **Cross-validation** (`part\_b\_cv.csv`, the same 5-fold KFold on the 1,020 train+validation rows). Gradient boosting scores a mean R² of 0.9352 (std 0.0044) and linear regression 0.9261 (std 0.0048). **The CV ranking disagrees with the single validation split.** The validation gap between the two models is only 0.025 hours of RMSE, while CV averages five folds over 1,020 rows. CV is therefore the more reliable signal, and **I would ship gradient boosting** for the regression. It also picks up the interaction terms in the data (sale × Express, rain × long distance) that a linear model cannot represent. Following the brief's rule, the scored final model is still the validation winner, linear regression, which gets **R² 0.9493 and RMSE 4.5235 hours on the test set** (scored once).

\---

## Model results (classification)

Source files: `outputs/part\_c\_\*.csv`. The split is stratified 840 / 180 / 180, with 195 / 42 / 41 delayed parcels.

|model (validation, threshold 0.5)|TN|FP|FN|TP|accuracy|precision|recall|F1|
|-|-|-|-|-|-|-|-|-|
|blind\_all\_on\_time|138|0|42|0|0.7667|0|0|0|
|logistic\_regression|128|10|7|35|0.9056|0.7778|0.8333|0.8046|
|decision\_tree\_depth\_6|128|10|11|31|0.8833|0.7561|0.7381|0.7470|
|random\_forest\_default|132|6|11|31|0.9056|0.8378|0.7381|0.7848|

* **Blind baseline.** Calling every parcel "on time" scores **76.67% accuracy** but catches **0 of the 42** delayed parcels (recall 0). Operations would never get a single alert from it, so accuracy alone is a misleading score for this problem.
* **Noise columns.** The five bookkeeping columns together take **0.2245 (22.5%) of the forest's importance**. The forest spends real splits on random numbers. By F-score, **all five fall outside the top 7**. Their ranks are 8, 12, 14, 16 and 17, and every F-score is below 5. The top 7 are distance\_km, hub\_tier, hub\_daily\_capacity, n\_stops, promised\_hours, rain\_flag and sale\_event\_flag, with F-scores from 389.2 down to 36.7.
* **k table** (`part\_c\_selectk.csv`). Validation F1 is 0.6988 at k=5, **0.8333 at k=7**, 0.7857 at k=10, 0.8049 at k=12 and 0.7848 at k=17. Five columns is too few, because it drops rain and sale effects. Seven is the best. Adding the weak and noise columns afterwards does not help and usually hurts.
* **Tuning** (`part\_c\_tuning.csv`). The grid search ran 27 candidates × 5 folds = 135 fits, and its best was n\_estimators=100, max\_depth=6, min\_samples\_leaf=1 (CV accuracy 0.8952, validation F1 0.7750). The randomized search ran 9 × 5 = 45 fits, and its best was 50 / 6 / 1 (CV accuracy 0.8905, validation F1 0.7848). The **untuned forest on the same 7 columns scored a validation F1 of 0.8333 and won**, so tuning did *not* beat the default here. The searches optimise CV accuracy on the training rows, not F1. And with only 42 delayed parcels in the 180 validation rows, one parcel more or less moves F1 by about 0.02. Gaps this small are within the noise of a single small validation set, so I would not read them as real.
* **Final configuration** (`part\_c\_final\_config.csv`): SelectKBest k=7 + RandomForest(n\_estimators=100, max\_depth unlimited, min\_samples\_leaf=1), chosen from `default\_forest\_same\_7\_features`.
* **Alert threshold** (`part\_c\_thresholds.csv`). The operations rule picks the highest threshold with validation recall ≥ 0.80. That is **t = 0.5**, with validation recall 0.8333 and precision 0.7955. Moving to 0.6 would raise precision to 0.8889 but drop recall to 0.7619, below the rule. Missing a late parcel costs more than an extra call, so 0.5 is kept.
* **Test result** (`part\_c\_final\_test.csv`, scored once): TN 125, FP 14, FN 10, TP 31, **precision 0.6889, recall 0.7561, F1 0.7209**. Test recall is lower than validation recall (0.7561 against 0.8333) and slightly under the 0.80 target. The test set has only 41 delayed parcels, so each one is worth about 2.4 recall points. The gap is about 3 parcels, which is consistent with sampling noise. Still, it means the threshold was tuned somewhat optimistically on one validation split, and recall in production should be watched (see Production plan).

\---

## Operations recommendation

Source files: `outputs/hub\_segments.csv`, `segment\_profile.csv`, `tier\_vs\_segment\_mismatch.csv`, `part\_d\_separation.csv`, `part\_c\_final\_test.csv` and `new\_shipment\_actions.csv`.

**Segments** (K-Means, K = 3, on scaled mean hours, delay rate, mean distance and mean stops):

|segment|hubs|mean hours|delay rate|mean distance km|mean stops|
|-|-|-|-|-|-|
|Standard|7|47.24|0.0257|188.93|55.37|
|Watch|8|56.81|0.1550|314.18|70.72|
|Escalate|9|64.65|0.4600|470.48|84.34|

**Escalate hubs** and their delay rates: HUB19 Guwahati 0.62, HUB24 Agartala 0.54, HUB20 Raipur 0.48, HUB13 Nagpur 0.46, HUB17 Patna 0.46, HUB23 Siliguri 0.44, HUB21 Bhubaneswar 0.40, HUB18 Ranchi 0.38 and HUB15 Kochi 0.36. These hubs need the escalation playbook: hub-manager review of flagged parcels, extra rider capacity and route splitting for long lanes.

**Hubs whose tier label is misleading** (`tier\_vs\_segment\_mismatch.csv`):

* **HUB07 Kolkata** is labelled Tier 1 but behaves like **Watch** (delay rate 0.14, mean lane 304.7 km). The "metro" label understates its risk.
* **HUB13 Nagpur** is labelled Tier 2 but behaves like **Escalate** (delay rate 0.46, 473.3 km, 83.9 stops). It is as stretched as the worst Tier 3 hubs.
* **HUB15 Kochi** is labelled Tier 2 but behaves like **Escalate** (delay rate 0.36, 466.5 km).
* **HUB22 Dehradun** is labelled Tier 3 but behaves like **Watch** (delay rate 0.20, 316.7 km). It performs better than its label.

The tier label should be treated as an administrative or capacity label, not as a performance label. Playbooks, staffing and escalation should follow the measured segment, and the segment should be refreshed every quarter. Note that the classifier still uses `hub\_tier` as a feature, so for these four hubs its probabilities lean towards the label. This is another reason to route actions by segment.

**Alert threshold and what it means:** the threshold is **0.5**. On the held-out test set, precision is 0.6889, so **about 7 of every 10 flagged parcels are really delayed**. Recall is 0.7561, so **about 7.5 of every 10 delayed parcels (roughly 3 in 4) are caught** before they are late.

**This batch of 48 new parcels:** **8 are flagged**. **6 of them need a hub-manager escalation** because their hub is in the Escalate segment (SHP90002 HUB24, SHP90003 HUB18, SHP90011 HUB17, SHP90025 HUB20, SHP90041 HUB21 and SHP90046 HUB21). The other **2 get a proactive customer call** (SHP90005 HUB22 and SHP90007 HUB09). The remaining 40 need no action.

**Why K = 3:** in `part\_d\_separation.csv` the across-to-within distance ratio is 2.894 at K=2 (sizes 15-9), **5.393 at K=3 (9-8-7)**, 6.305 at K=4 (8-7-5-4) and 6.887 at K=5 (8-7-4-3-2). The ratio keeps rising with K, as it always does when clusters get smaller. But K = 4 has a 4-hub cluster and K = 5 has clusters of 3 and 2 hubs. A cluster of fewer than 6 hubs is too small to carry its own playbook. K = 3 gives three balanced segments (9 / 8 / 7) and matches the three playbooks operations actually runs, and the jump from K=2 to K=3 is the largest gain in separation.

\---

## Production plan

**(a) Production-readiness characteristics mapped to this repository**

|characteristic|concrete item|
|-|-|
|Accuracy|Test metrics in `part\_b\_metrics.csv` (R² 0.9493) and `part\_c\_final\_test.csv` (F1 0.7209, recall 0.7561), each scored once on untouched rows|
|Reliability|`predict\_delay.py` asserts there are no missing inputs; unknown payment modes fall back to `Unknown`; the scaler and selector are fitted on training rows only|
|Reproducibility|Pinned `requirements.txt`; fixed `random\_state=42` everywhere; `generate\_data.py` committed verbatim; `package\_model.py` rebuilds the pickle; two runs of `predict\_delay.py` give byte-identical output|
|Performance (speed)|The whole batch is scored in one vectorised `predict\_proba` call; the 7-feature forest scores 48 parcels in well under a second|
|Maintainability|One shared module, `src/shipment\_prep.py`, holds the constants and the two preprocessing functions; each Part is its own script|
|Consistency|`predict\_delay.py` calls the same `clean\_shipments` (with the stored `models/impute\_values.csv` medians) and `build\_features` used in training, and orders columns by `models/feature\_order.csv`|
|Monitorability|`outputs/part\_d\_reload\_check.csv` verifies the saved model; `new\_shipment\_actions.csv` keeps one auditable row per parcel with its probability, flag and action|

**(b) Quantities to monitor, with triggers**

1. **Weekly delayed share of delivered parcels.** The baseline is 0.2317. Alert if it goes above 0.30 or below 0.15 for two consecutive non-sale weeks.
2. **Weekly share of parcels flagged.** Expect about 20–25%; the batch above flagged 8 of 48, or 16.7%. Alert if it goes above 35% or below 10%.
3. **Recall on parcels whose outcome is now known** (a rolling 4-week window). Alert and review the threshold if recall drops below 0.70. Also alert if precision drops below 0.55, because that means the call centre is wasting effort.
4. **Input drift.** Alert if mean `distance\_km` moves more than 15% from the training mean, or if the share of `Unknown` payment modes or median-filled values goes above 5%.

**(c) Retraining and drift.** Retrain **monthly** on a rolling window of recent delivered parcels. Retrain immediately if any trigger in (b) fires for two weeks in a row. Promote a new model only if its held-out recall is at least 0.80 and its F1 is not lower than the current model's. **The 15–28 September sale window (`sale\_event\_flag = 1`)** adds about 7 hours to every parcel and 12 more hours to Express parcels, so delay and flag rates jump during it. During a sale, compare metrics against the *previous sale event*, not against ordinary weeks, so the planned surge does not trigger a false drift alarm. Make sure sale weeks appear in every training window. **Data drift here would look like** new lanes with longer distances, more stops per rider route, a different service mix (for example more Express), hubs changing operating behaviour (another "Nagpur" case), or new rain patterns in the monsoon. It would show up as the feature distributions moving away from the training set and as recall falling on newly delivered parcels.

\---

## Limitations

1. **Synthetic data.** Every row comes from a known formula with Gaussian noise, so the patterns are cleaner and more additive than real logistics data. Real deployments will have messier features (traffic, address quality, rider absence) and lower scores.
2. **A small test set.** The test set has only 41 delayed parcels, so each parcel moves recall by about 2.4 points. The reported test recall of 0.7561 has a wide confidence band.
3. **A threshold and model chosen on one validation split.** Both rest on 180 validation rows (42 delayed). Several candidates were within 0.05 F1 of each other, and test recall came in below the validation recall the threshold was chosen for. A cross-validated threshold would be more robust.
4. **Tier label as a feature.** `hub\_tier` is a feature, but four hubs behave unlike their label. The classifier can under-flag parcels at mislabelled hubs such as Nagpur or Kochi.
5. **No time-based validation.** The splits are random, not chronological, and the data covers only July–September 2026 with one sale event. Behaviour across seasons and future festivals is untested.
6. **Median filling.** Missing distances and declared values are filled with medians computed on the full historical table. That is reasonable for a few gaps, but it would hide a systematic data-capture failure.

