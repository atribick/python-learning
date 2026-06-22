# Python Learning Projects

A collection of data analysis projects built while learning Python, pandas, and matplotlib.
Focused on publicly available biomedical and regulatory datasets.

## Projects

### FDA Medical Device Recalls (2024)
**Data:** FDA Medical Device Recall database (2024)
**Tools:** Python, pandas, matplotlib

Analysis of 103 confirmed FDA medical device recalls from 2024. Includes data cleaning,
standardization of inconsistent product area labels, and visualization of recall trends
by product type and month.

**Charts:**
- Top 10 product areas by recall count
- Recall frequency by month

### UCI Heart Disease Dataset
**Data:** UCI Heart Disease dataset via Kaggle (920 patients, 4 clinical sites)
**Tools:** Python, pandas, matplotlib

Exploratory analysis of clinical indicators associated with heart disease diagnosis.
Includes handling of missing data, removal of physiologically impossible values, and
binary classification of diagnosis outcomes.

**Charts:**
- Diagnosis distribution
- Age distribution by diagnosis
- Chest pain type by diagnosis
- Max heart rate by diagnosis
- Serum cholesterol by diagnosis
- ST slope by diagnosis

### FDA MAUDE Adverse Event Analysis (2024)
**Data:** FDA MAUDE (Manufacturer and User Facility Device Experience) database (2024)
**Tools:** Python, pandas, matplotlib

Exploratory analysis of 2.6 million adverse event reports from the FDA MAUDE database.
Includes memory optimization for large dataset handling, column selection, datetime filtering,
and categorical encoding. Analysis covers device categories, reporting trends, manufacturer
origin, reporter type, device evaluation rates, and post-event device availability.

**Charts:**
- Top 15 device categories by adverse event volume
- Adverse events by month
- Adverse events by manufacturer origin (US vs foreign)
- Adverse events by reporter type
- Device evaluation rate by manufacturer
- Device availability after adverse event