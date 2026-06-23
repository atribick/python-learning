import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


# Load cleaned data
df = pd.read_csv("../heart_disease_analysis/data/heart_disease_cleaned.csv")

# Drop non-predictive columns
df = df.drop(columns=["id", "dataset"])

# Features and target
X = df.drop(columns=["num"])
y = df["num"]

# One-hot encode categoricals (drop_first avoids multicollinearity)
X = pd.get_dummies(X, drop_first=True)

# 80/20 split, stratify preserves class ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit scaler on train only, then apply to both (prevents data leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("=== Logistic Regression ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")
print(classification_report(y_test, y_pred_lr))
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression — Confusion Matrix")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("=== Random Forest ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(classification_report(y_test, y_pred_rf))
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest — Confusion Matrix")

# Feature importances
feature_names = pd.get_dummies(df.drop(columns=["num"]), drop_first=True).columns
importances = pd.Series(rf.feature_importances_, index=feature_names)
importances.sort_values().plot(kind="barh", figsize=(8, 6))
plt.title("Feature Importances — Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# 5-fold cross-validation scored on recall
lr_cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="recall")
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="recall")

print("=== Cross-Validation Recall Scores (5-Fold) ===")
print(f"Logistic Regression: {lr_cv_scores.mean():.3f} +/- {lr_cv_scores.std():.3f}")
print(f"Random Forest:       {rf_cv_scores.mean():.3f} +/- {rf_cv_scores.std():.3f}")

# Compute ROC curve and AUC for both models
lr_probs = lr.predict_proba(X_test)[:,1]
rf_probs = rf.predict_proba(X_test)[:,1]

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

lr_auc = auc(lr_fpr, lr_tpr)
rf_auc = auc(rf_fpr, rf_tpr)

# Plot ROC curves
plt.figure(figsize=(8,6))
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.3f}")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {lr_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Logistic Regression vs Random Forest")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Logistic Regression AUC: {lr_auc:.3f}")
print(f"Random Forest AUC: {rf_auc:.3f}")