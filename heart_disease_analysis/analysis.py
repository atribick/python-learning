import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# Shared palette
palette = {0: "steelblue", 1: "orange"}

# Load data
df = pd.read_csv("data/heart_disease_uci.csv")

# Drop columns missing more than 50% of values
threshold = 0.5
df = df[df.columns[df.isnull().mean() < threshold]]

# Drop rows with any remaining missing values
df = df.dropna()

# Remove physiologically impossible values
df = df[(df["chol"] > 0) & (df["trestbps"] > 0) & (df["thalch"] > 0)]

# Simplify diagnosis to binary: 0 = no disease, 1 = disease
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

df.to_csv('data/heart_disease_cleaned.csv', index=False)

# Chart 1: Diagnosis distribution
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x="num", hue="num", legend=False, palette=palette)
ax.set_title("Heart Disease Diagnosis Distribution")
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Number of Patients")
ax.set_xticks([0, 1])
ax.set_xticklabels(["No Disease", "Disease"])
plt.tight_layout()
plt.savefig("diagnosis_distribution.png", dpi=150)
plt.show()

# Chart 2: Age distribution by diagnosis
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x="age", hue="num", bins=20, alpha=0.6, palette=palette)
ax.set_title("Age Distribution by Diagnosis")
ax.set_xlabel("Age")
ax.set_ylabel("Number of Patients")
ax.legend(labels=["Disease", "No Disease"])
plt.tight_layout()
plt.savefig("age_distribution.png", dpi=150)
plt.show()

# Chart 3: Chest pain type by diagnosis
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="cp", hue="num", palette=palette,
              order=["asymptomatic", "non-anginal", "atypical angina", "typical angina"])
ax.set_title("Chest Pain Type by Diagnosis")
ax.set_xlabel("Chest Pain Type")
ax.set_ylabel("Number of Patients")
ax.legend(labels=["No Disease", "Disease"])
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("chest_pain_by_diagnosis.png", dpi=150)
plt.show()

# Chart 4: Max heart rate by diagnosis
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x="num", y="thalch", hue="num", legend=False, palette=palette)
ax.set_title("Max Heart Rate by Diagnosis")
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Max Heart Rate (bpm)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["No Disease", "Disease"])
plt.tight_layout()
plt.savefig("max_heart_rate_by_diagnosis.png", dpi=150)
plt.show()

# Chart 5: Cholesterol by diagnosis
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x="num", y="chol", hue="num", legend=False, palette=palette)
ax.set_title("Cholesterol by Diagnosis")
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Serum Cholesterol (mg/dL)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["No Disease", "Disease"])
plt.tight_layout()
plt.savefig("cholesterol_by_diagnosis.png", dpi=150)
plt.show()

# Chart 6: ST slope by diagnosis
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x="slope", hue="num", palette=palette,
              order=["upsloping", "flat", "downsloping"])
ax.set_title("ST Slope by Diagnosis")
ax.set_xlabel("Slope Type")
ax.set_ylabel("Number of Patients")
ax.legend(labels=["No Disease", "Disease"])
plt.tight_layout()
plt.savefig("slope_by_diagnosis.png", dpi=150)
plt.show()