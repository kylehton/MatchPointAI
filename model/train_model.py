import xgboost as xgb
from create_feature_set import create_feature_set
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

df = create_feature_set()

# Define feature columns (all the diff_ columns)
feature_cols = [col for col in df.columns if col.startswith('diff_')]

# Select features and target
X = df[feature_cols]
y = df['target']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6, stratify=y)

model = xgb.XGBClassifier()

params = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200]
}

grid = GridSearchCV(model, params, cv=6)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)

best_model = grid.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test set score:", test_score)