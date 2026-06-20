import pandas as pd
import matplotlib.pyplot as plt

# Load
df = pd.read_excel("data/recalls_2024.xlsx", header=1)

# Clean
# Convert Date to DateTime and filter to 2024 only
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"].dt.year == 2024]

# Standardize inconsistent Product Area labels
replacements = {
    "Infusion Pumps": "Infusion Pump",
    "Ventilators": "Ventilator",
    "Mobile App": "Software",
    "Anesthesia System": "Anesthesia Machine",
    "Hemodialysis System": "Hemodialysis",
    "Kits and Trays": "Trays and Kits"
}
df["Product Area"] = df["Product Area"].replace(replacements)

# Chart 1: Top 10 Product Areas
top10 = df["Product Area"].value_counts().head(10)

top10.plot(kind="bar")
plt.title("Top 10 Product Areas by Recall Count (2024)")
plt.xlabel("Product Area")
plt.ylabel("Number of Recalls")
plt.tight_layout()
plt.xticks(rotation=45, ha="right")
plt.yticks(range(0, int(top10.max()) + 2, 2))
plt.savefig("top10_product_areas.png", dpi=150)
plt.show()

# Chart 2: Recalls by Month
df["Month"] = df["Date"].dt.month
monthly = df.groupby("Month")["Date"].count()

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

monthly.plot(kind="bar")
plt.title("FDA Medical Device Recalls by Month (2024)")
plt.xlabel("Month")
plt.ylabel("Number of Recalls")
plt.tight_layout()
plt.xticks(ticks=range(12), labels=month_names, rotation=45, ha="right")
plt.savefig("recalls_by_month.png", dpi=150)
plt.show()