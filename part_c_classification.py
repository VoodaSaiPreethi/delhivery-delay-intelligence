"""Part C - delay classifier, feature signal check, tuning and alert threshold."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.shipment_prep import FEATURES, NOISE_COLUMNS  # noqa: E402

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

# 1. stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (y_train.sum(), y_val.sum(), y_test.sum()) == (195, 42, 41)


def cm_row(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# 2. blind baseline + three classifiers at the 0.5 rule
model_rows = [{"model": "blind_all_on_time", **cm_row(y_val, np.zeros(len(y_val), dtype=int))}]

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **cm_row(y_val, logreg.predict(X_val))})

best_d, best_f1 = None, -1.0
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f1:  # strict > keeps the smaller depth on ties
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
model_rows.append({"model": f"decision_tree_depth_{best_d}", **cm_row(y_val, tree.predict(X_val))})

rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **cm_row(y_val, rf_default.predict(X_val))})
pd.DataFrame(model_rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# 3. which columns carry signal?
selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
scores = pd.DataFrame({"feature": FEATURES, "f_score": selector_all.scores_,
                       "rf_importance": rf_default.feature_importances_})
scores["f_rank"] = scores["f_score"].rank(ascending=False, method="first").astype(int)
scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
scores = scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
scores.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

k_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    k_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})
pd.DataFrame(k_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning on the 7 selected columns (selector fitted on training rows only)
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
Xtr7, Xva7 = selector.transform(X_train), selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}

gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5,
                        random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)

DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}


def fmt(params):
    return ";".join(f"{k}={params[k]}" for k in ["n_estimators", "max_depth", "min_samples_leaf"])


tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": len(gs.cv_results_["params"]), "cv_folds": 5,
     "total_fits": len(gs.cv_results_["params"]) * 5, "best_params": fmt(gs.best_params_),
     "best_cv_accuracy": round(gs.best_score_, 4), "val_f1": f1_score(y_val, gs.predict(Xva7))},
    {"search": "randomized_search", "candidates": len(rs.cv_results_["params"]), "cv_folds": 5,
     "total_fits": len(rs.cv_results_["params"]) * 5, "best_params": fmt(rs.best_params_),
     "best_cv_accuracy": round(rs.best_score_, 4), "val_f1": f1_score(y_val, rs.predict(Xva7))},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": fmt(DEFAULT_PARAMS), "best_cv_accuracy": np.nan,
     "val_f1": f1_score(y_val, default7.predict(Xva7))},
])
tuning["val_f1"] = tuning["val_f1"].round(4)
tuning.to_csv("outputs/part_c_tuning.csv", index=False)

# 5. choose the final configuration (ties: default, then grid, then randomized)
params_by_name = {"grid_search": gs.best_params_, "randomized_search": rs.best_params_,
                  "default_forest_same_7_features": DEFAULT_PARAMS}
best_val = tuning["val_f1"].max()
chosen_from = next(n for n in ["default_forest_same_7_features", "grid_search", "randomized_search"]
                   if tuning.loc[tuning["search"] == n, "val_f1"].iloc[0] == best_val)
chosen = params_by_name[chosen_from]
final = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                  ("model", RandomForestClassifier(random_state=42, **chosen))]).fit(X_train, y_train)

# 6. alert threshold from validation probabilities
val_proba = final.predict_proba(X_val)[:, 1]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thr_rows = [{"threshold": t, **cm_row(y_val, (val_proba >= t).astype(int))} for t in THRESHOLDS]
thr = pd.DataFrame(thr_rows)
thr.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
ok = thr.loc[thr["recall"].round(4) >= 0.80, "threshold"]
threshold = float(ok.max()) if len(ok) else 0.1

# 7. one test evaluation
test_proba = final.predict_proba(X_test)[:, 1]
pd.DataFrame([{"threshold": threshold, **cm_row(y_test, (test_proba >= threshold).astype(int))}]).round(4).to_csv(
    "outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": K, "n_estimators": chosen["n_estimators"],
               "max_depth": "unlimited" if chosen["max_depth"] is None else chosen["max_depth"],
               "min_samples_leaf": chosen["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen_from}]).to_csv("outputs/part_c_final_config.csv", index=False)

print("Part C done. tree depth:", best_d, "| chosen:", chosen_from, chosen, "| threshold:", threshold)
