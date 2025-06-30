import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import warnings
from create_feature_set import create_feature_set
import joblib
from tqdm import tqdm

print("Creating feature set...")
df = create_feature_set()

# Define feature columns (all the diff_ columns)
feature_cols = [col for col in df.columns if col.startswith('diff_')]
# Remove any old surface winrate diffs if present, only keep diff_surface_wr
feature_cols = [col for col in feature_cols if not (
    col.startswith('diff_hard_court_wr') or col.startswith('diff_clay_court_wr') or col.startswith('diff_grass_court_wr') or col.startswith('diff_carpet_court_wr'))]

# Select features and target
X = df[feature_cols]
y = df['target']

model = XGBClassifier()

warnings.filterwarnings("ignore")

# to help avoid bias with negative/minority values, since calculating diffs
neg, pos = np.bincount(y)
scale_pos_weight = neg / pos

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6, stratify=y)

# Define the model
base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    n_estimators=1000
)

# Hyperparameter tuning using RandomizedSearchCV
# if needed, decrease some params for slower learning
param_dist = {
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 1, 5],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 1.5, 2]
}

cv = StratifiedKFold(n_splits=5)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=40,
    scoring="accuracy",
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=6
)

# Fit w params
print("Starting hyperparameter search...")
search.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Best model
model = search.best_estimator_

# Print results
print("Best CV Score:", search.best_score_)
print("Best Parameters:", search.best_params_)

# Additional cross-validation - here's where you can use tqdm naturally
print("Running additional cross-validation...")
cv_scores = []
for train_idx, val_idx in tqdm(cv.split(X_train, y_train), total=cv.n_splits, desc="CV Folds"):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model.fit(X_train_fold, y_train_fold)
    score = model.score(X_val_fold, y_val_fold)
    cv_scores.append(score)

cv_scores = np.array(cv_scores)
print(f"Detailed CV scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "xgb_model.joblib")