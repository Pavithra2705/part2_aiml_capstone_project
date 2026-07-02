# Part 2 — Supervised Machine Learning (Bike Sharing Dataset)

## Labels and features

Loaded cleaned_data.csv from Part 1. Target column is cnt.

y_reg is the raw cnt column, used for the regression task.

y_clf is a binary version of cnt, made by checking if cnt is above its own median. This gives a roughly 50/50 split by design.

For X, I dropped casual and registered, since cnt is literally casual plus registered added together, so keeping them would let the model just do addition instead of actually learning from weather and calendar features. I also dropped dteday and instant since they are basically a date and a row number, not real predictive features, and dropped the raw season column since a cleaner version of it (season_label) was already available from Part 1.

## Encoding categorical columns

weathersit has a real order to it, 1 is clear weather and 4 is the worst weather, so I left it as a plain number instead of one-hot encoding it, since the order itself carries meaning.

season_label, mnth and weekday have no real order (winter isn't "more" than spring, Monday isn't "more" than Sunday), so turning them into plain numbers would trick the model into thinking there's a ranking that doesn't exist. I one-hot encoded these three using pd.get_dummies with drop_first=True to avoid the dummy variable trap.

yr, holiday and workingday were already 0/1 binary columns, so no encoding was needed for them.

## Leak-free split and scaling

Split X, y_reg and y_clf together using train_test_split with test_size=0.2 and random_state=42.

Fit StandardScaler only on X_train, then used that same fitted scaler to transform both X_train and X_test. Fitting the scaler on the full dataset would mean the scaler's mean and standard deviation are calculated partly from test rows, and that information would sneak into training, which counts as data leakage even though it looks harmless. Fitting on train only keeps the test set truly unseen until evaluation.

## Regression models

| Model | MSE | R2 |
|---|---|---|
| Linear Regression | 683198.44 | 0.8296 |
| Ridge (alpha = 1.0) | 682394.54 | 0.8298 |

Ridge came out very slightly better here. Ridge adds a penalty on large coefficients, so it shrinks them a bit compared to plain Linear Regression. Alpha controls how strong that penalty is, a higher alpha means more shrinkage and simpler, more conservative coefficients. Since the two results are almost identical, it tells me this data isn't suffering much from overfitting or multicollinearity in the first place, so the penalty barely needed to do anything.

## Top 3 coefficients (Linear Regression)

yr: 999.75
temp: 629.996
season_label_Winter: 381.63

These coefficients are in scaled units, meaning a one standard deviation increase in that feature is linked to that many more predicted rentals, holding other features fixed.

yr having the biggest effect makes sense, 2012 had noticeably higher ridership than 2011 across the board, so this coefficient is basically capturing that overall year-over-year growth.

temp being positive matches what I saw in Part 1, warmer days are linked to more rentals.

season_label_Winter being positive is a bit surprising at first, since winter is generally a lower rental season overall. But once temp is already in the model separately, this coefficient is only measuring the extra winter effect after accounting for temperature, so it might be picking up something like steadier registered/commuter demand in winter compared to the baseline season. I wouldn't treat this one as a strong standalone insight, more of a small secondary effect.

## Classification model — class balance check

y_clf_train came out to 299 rows in class 1 and 285 rows in class 0, which is about 51.2% versus 48.8%. Since y_clf was built by splitting at the median, the classes are close to even by construction, there wasn't a real imbalance problem to fix here.

Before / after comparison:

| Stage | Class 0 count | Class 1 count |
|---|---|---|
| Before (raw split) | 285 | 299 |
| After (class_weight='balanced') | 285 | 299 |

The counts are identical before and after because class_weight='balanced' doesn't resample or duplicate any rows, it just adjusts how much each class contributes to the loss function during training. I chose this over SMOTE because the classes were already close to 50/50, so there was nothing meaningful to oversample, using SMOTE here would have been solving a problem that didn't really exist. I kept class_weight='balanced' anyway as a safety net in case the split had come out more uneven.

## Confusion matrix and classification report

Confusion matrix:
[[71 10]
 [9 57]]

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

Class 0: precision 0.89, recall 0.88, f1 0.88
Class 1: precision 0.85, recall 0.86, f1 0.86
Overall accuracy: 0.87

For this dataset, predicting whether a day will have high or low bike demand, I'd say recall matters slightly more for class 1 (high demand days). Missing a high demand day (a false negative) could mean the business under-stocks bikes or staff for a day that actually needed more, which is a more costly mistake than over-preparing for a day that turns out to be average.

## ROC curve and AUC

Plotted the ROC curve, AUC came out to 0.9607.

An AUC that close to 1.0 means the model is very good at telling apart high demand days from low demand days across basically every possible threshold, not just the default 0.5 cutoff. It's a strong result for this dataset.

## Threshold sensitivity

Checked thresholds from 0.3 to 0.7 in steps of 0.1:

Threshold 0.3: precision 0.810, recall 0.970, f1 0.883
Threshold 0.4: precision 0.836, recall 0.924, f1 0.878
Threshold 0.5: precision 0.851, recall 0.864, f1 0.857
Threshold 0.6: precision 0.857, recall 0.818, f1 0.837
Threshold 0.7: precision 0.883, recall 0.803, f1 0.841

Best F1 score is at threshold 0.3 (f1 = 0.883).

Given that recall matters slightly more here (as explained above), I would actually lean toward lowering the threshold from the default 0.5 down toward 0.3, since it pushes recall up to 0.97 while precision only drops to about 0.81. The cost of doing this is more false positives, days predicted as high demand that turn out average, but in this business case, over-preparing for demand is a cheaper mistake than under-preparing for it.

## Regularization experiment (C=1.0 vs C=0.01)

C=1.0: precision 0.851, recall 0.864, AUC 0.961
C=0.01: precision 0.849, recall 0.939, AUC 0.943

C controls how strong the regularization penalty is in logistic regression, and it works opposite to alpha in Ridge, a smaller C means a stronger penalty. Dropping C to 0.01 made the model simpler and pushed recall up noticeably, but it cost some AUC. So the C=0.01 model isn't strictly worse, it trades a bit of overall separation ability for catching more of the actual high-demand days, which could be a reasonable tradeoff depending on what the business cares about more.

## Bootstrap confidence interval for the AUC difference

Ran 500 bootstrap samples on the test set, computing the AUC difference (C=1.0 minus C=0.01) each time.

Mean difference: 0.0182
95% confidence interval: [0.0003, 0.0387]

The interval does not include zero, so the AUC advantage of the C=1.0 model over the C=0.01 model looks statistically reliable, not just random luck from this one train/test split. That said, the interval is fairly close to zero on the lower end, so the advantage is real but not huge.

## Files in this repo

cleaned_data.csv is the input dataset from Part 1. part2.ipynb has all the preprocessing, model training and evaluation code. roc_curve.png is the saved ROC plot. This file is the README.
