import pandas as pd
import matplotlib.pyplot as plt

# Columns to load
cols = [
    "MDR_REPORT_KEY",
    "DATE_RECEIVED",
    "BRAND_NAME",
    "GENERIC_NAME",
    "MANUFACTURER_D_NAME",
    "MANUFACTURER_D_STATE_CODE",
    "MANUFACTURER_D_COUNTRY_CODE",
    "DEVICE_OPERATOR",
    "DEVICE_REPORT_PRODUCT_CODE",
    "DEVICE_EVALUATED_BY_MANUFACTUR",
    "COMBINATION_PRODUCT_FLAG",
    "DEVICE_AVAILABILITY",
    "SERVICED_BY_3RD_PARTY_FLAG",
]

# Load data
df = pd.read_csv(
    "data/DEVICE2024.txt",
    sep="|",
    low_memory=False,
    encoding="latin-1",
    usecols=cols,
)

# Filter to 2024
df["DATE_RECEIVED"] = pd.to_datetime(df["DATE_RECEIVED"], errors="coerce")
df = df[df["DATE_RECEIVED"].dt.year == 2024]

# Optimize memory for low-cardinality columns
cat_cols = [
    "MANUFACTURER_D_STATE_CODE",
    "MANUFACTURER_D_COUNTRY_CODE",
    "DEVICE_OPERATOR",
    "DEVICE_EVALUATED_BY_MANUFACTUR",
    "COMBINATION_PRODUCT_FLAG",
    "DEVICE_AVAILABILITY",
    "SERVICED_BY_3RD_PARTY_FLAG",
]
for col in cat_cols:
    df[col] = df[col].astype("category")

# Shared compact format for number-axis
k_formatter = plt.FuncFormatter(lambda x, _: f"{int(x/1000)}K")

# Chart 1: Top 15 device categories by adverse event volume
top15 = df["GENERIC_NAME"].value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 8))
top15.plot(kind="barh", ax=ax)
ax.set_title("Top 15 Device Categories by Adverse Event Volume (2024)")
ax.set_xlabel("Number of Adverse Events")
ax.set_ylabel("Device Category")
ax.xaxis.set_major_formatter(k_formatter)
plt.tight_layout()
plt.savefig("top15_device_categories.png", dpi=150, bbox_inches="tight")
plt.show()

# Chart 2: Adverse events by month
df["Month"] = df["DATE_RECEIVED"].dt.month
monthly = df.groupby("Month")["MDR_REPORT_KEY"].count()

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(10, 6))
monthly.plot(kind="bar", ax=ax)
ax.set_title("Adverse Events by Month (2024)")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Adverse Events")
ax.yaxis.set_major_formatter(k_formatter)
plt.xticks(ticks=range(12), labels=month_names, rotation=45, ha="right")
plt.tight_layout()
plt.savefig("adverse_events_by_month.png", dpi=150, bbox_inches="tight")
plt.show()

# Chart 3: US vs foreign manufacturers
country_counts = df["MANUFACTURER_D_COUNTRY_CODE"].value_counts()
us = country_counts["US"]
foreign = country_counts[country_counts.index != "US"].sum()

manufacturer_origin = pd.Series({"US": us, "Foreign": foreign})

fig, ax = plt.subplots(figsize=(8, 6))
manufacturer_origin.plot(kind="bar", ax=ax)
ax.set_title("Adverse Events by Manufacturer Origin (2024)")
ax.set_xlabel("Manufacturer Origin")
ax.set_ylabel("Number of Adverse Events")
ax.yaxis.set_major_formatter(k_formatter)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("manufacturer_origin.png", dpi=150, bbox_inches="tight")
plt.show()

# Chart 4: Reporter type
operator_map = {
    "0HP": "Health Professional",
    "0LP": "Patient/Lay Person",
    "000": "No Information",
}
df["DEVICE_OPERATOR_LABEL"] = df["DEVICE_OPERATOR"].map(operator_map)
reporter_counts = df["DEVICE_OPERATOR_LABEL"].value_counts()

fig, ax = plt.subplots(figsize=(8, 6))
reporter_counts.plot(kind="bar", ax=ax)
ax.set_title("Adverse Events by Reporter Type (2024)")
ax.set_xlabel("Reporter Type")
ax.set_ylabel("Number of Adverse Events")
ax.yaxis.set_major_formatter(k_formatter)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reporter_type.png", dpi=150, bbox_inches="tight")
plt.show()

# Chart 5: Device evaluation rate
eval_map = {
    "Y": "Evaluated",
    "N": "Not Evaluated",
    "R": "Returned to Manufacturer",
    "*": "Unknown",
}
df["EVAL_LABEL"] = df["DEVICE_EVALUATED_BY_MANUFACTUR"].map(eval_map)
eval_counts = df["EVAL_LABEL"].value_counts()

fig, ax = plt.subplots(figsize=(8, 6))
eval_counts.plot(kind="bar", ax=ax)
ax.set_title("Device Evaluation Rate by Manufacturer (2024)")
ax.set_xlabel("Evaluation Status")
ax.set_ylabel("Number of Adverse Events")
ax.yaxis.set_major_formatter(k_formatter)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("device_evaluation_rate.png", dpi=150, bbox_inches="tight")
plt.show()

# Chart 6: Device availability after adverse event
availability_map = {
    "N": "Not Available",
    "R": "Returned to Manufacturer",
    "Y": "Available",
    "*": "Unknown",
}
df["AVAILABILITY_LABEL"] = df["DEVICE_AVAILABILITY"].map(availability_map)
availability_counts = df["AVAILABILITY_LABEL"].value_counts()

fig, ax = plt.subplots(figsize=(8, 6))
availability_counts.plot(kind="bar", ax=ax)
ax.set_title("Device Availability After Adverse Event (2024)")
ax.set_xlabel("Availability Status")
ax.set_ylabel("Number of Adverse Events")
ax.yaxis.set_major_formatter(k_formatter)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("device_availability.png", dpi=150, bbox_inches="tight")
plt.show()