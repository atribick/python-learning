import matplotlib.pyplot as plt
import pandas as pd

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

# Boxplot styling shared across charts
box_style = dict(
    patch_artist=True,
    boxprops=dict(facecolor="steelblue", color="steelblue"),
    medianprops=dict(color="white"),
    whiskerprops=dict(color="steelblue"),
    capprops=dict(color="steelblue"),
    flierprops=dict(markeredgecolor="steelblue"),
)

# Chart 1: Diagnosis distribution
df["num"].value_counts().plot(kind="bar")
plt.title("Heart Disease Diagnosis Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Patients")
plt.xticks(ticks=[0,1], labels=["No Disease", "Disease"], rotation=0)
plt.tight_layout()
plt.savefig("diagnosis_distribution.png", dpi=150)
plt.show()

# Chart 2:Age distribution by diagnosis
df[df["num"] == 0]["age"].plot(kind="hist", alpha =0.5, label ="No Disease", bins=20)
df[df["num"] == 1]["age"].plot(kind="hist", alpha =0.5, label ="Disease", bins=20)
plt.title("Age Distribution by Diagnosis")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.legend()
plt.tight_layout()
plt.savefig("age_distribution.png", dpi=150)
plt.show()

# Chart 3: Chest pain type by diagnosis
cp_counts = df.groupby(["cp", "num"]).size().unstack()
cp_counts.plot(kind="bar")
plt.title("Chest Pain Type by Diagnosis")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45, ha="right")
plt.legend(["No Disease", "Disease"])
plt.tight_layout()
plt.savefig("chest_pain_by_diagnosis.png", dpi=150)
plt.show()

# Chart 4: Max heart rate by diagnosis
no_disease_hr = df[df["num"] == 0]["thalch"]
disease_hr = df[df["num"] == 1]["thalch"]
plt.boxplot([no_disease_hr, disease_hr], tick_labels=["No Disease", "Disease"], **box_style)
plt.title("Max Heart Rate by Diagnosis")
plt.suptitle("")
plt.xlabel("Diagnosis")
plt.ylabel("Max Heart Rate (bpm)")
plt.grid(False)
plt.tight_layout()
plt.savefig("max_heart_rate_by_diagnosis.png", dpi=150)
plt.show()

# Chart 5: Cholesterol by diagnosis
no_disease_chol = df[df["num"] == 0]["chol"]
disease_chol = df[df["num"] == 1]["chol"]
plt.boxplot([no_disease_chol, disease_chol], tick_labels=["No Disease", "Disease"], **box_style)
plt.title("Cholesterol by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Serum Cholesterol (mg/dL)")
plt.grid(False)
plt.tight_layout()
plt.savefig("cholesterol_by_diagnosis.png", dpi=150)
plt.show()

# Chart 6: Slope of peak exercise ST segment by diagnosis
slope_counts = df.groupby(["slope", "num"]).size().unstack()
slope_counts.plot(kind="bar")
plt.title("ST Slope by Diagnosis")
plt.xlabel("Slope Type")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45, ha="right")
plt.legend(["No Disease", "Disease"])
plt.tight_layout()
plt.savefig("slope_by_diagnosis.png", dpi=150)
plt.show()