import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === CHANGE ONLY THIS PATH IF NEEDED ===
FILE_PATH = r"C:\Users\tanis\OneDrive\Desktop\Market_Prediction_Project\Dataset_DimexRank_36FinancialTimeSeries.csv"

# Load dataset
df = pd.read_csv(FILE_PATH)

# Rename first column as Date (your dataset has date in first column)
df.rename(columns={df.columns[0]: "Date"}, inplace=True)

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Convert wide format → long format (important fix)
long_df = df.melt(id_vars="Date", var_name="Market", value_name="Close")

# Sort
long_df = long_df.sort_values(["Market", "Date"])

# Monthly aggregation
long_df["YearMonth"] = long_df["Date"].dt.to_period("M")
monthly = long_df.groupby(["Market", "YearMonth"])["Close"].last().reset_index()

# Monthly return
monthly["Monthly_Return"] = monthly.groupby("Market")["Close"].pct_change()
monthly.dropna(inplace=True)
# SAVE TO NEW CSV (this is step 1)
monthly.to_csv("monthly_features.csv", index=False)
print("monthly_features.csv saved successfully")
# Features
monthly["Avg_3M"] = monthly.groupby("Market")["Monthly_Return"].rolling(3).mean().reset_index(0, drop=True)
monthly["Avg_6M"] = monthly.groupby("Market")["Monthly_Return"].rolling(6).mean().reset_index(0, drop=True)
monthly["Volatility"] = monthly.groupby("Market")["Monthly_Return"].rolling(3).std().reset_index(0, drop=True)
monthly.dropna(inplace=True)

# Target = next month return
monthly["Target"] = monthly.groupby("Market")["Monthly_Return"].shift(-1)
monthly.dropna(inplace=True)

# Train model
features = ["Monthly_Return", "Avg_3M", "Avg_6M", "Volatility"]
X = monthly[features]
y = monthly["Target"]

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)
print("\nModel R² Score:", r2_score(y, pred))

# ============================
# SAVE MONTHLY FEATURES CSV
# ============================
monthly.to_csv("monthly_features.csv", index=False)
print("\nmonthly_features.csv saved successfully")

# ============================
# ACCURACY
# ============================
print("\nModel Accuracy (R² Score): {:.4f}".format(r2_score(y, pred)))

# ============================
# TOP 5 PREDICTIONS
# ============================
latest = monthly.groupby("Market").tail(1).copy()
latest["Prediction"] = model.predict(latest[features])

top5 = latest.sort_values("Prediction", ascending=False).head(5)

print("\nTop 5 Predicted Markets:")
print(top5[["Market", "Prediction"]])

top5[["Market", "Prediction"]].to_csv("top5_predictions.csv", index=False)
print("top5_predictions.csv saved")
# ============================
# LINE CHART (Required by task)
# ============================
plt.figure()
plt.plot(top5["Market"], top5["Prediction"], marker="o")
plt.title("Predicted Next Month Return (Top 5 Markets)")
plt.xlabel("Market")
plt.ylabel("Predicted Return")
plt.grid(True)
plt.show()

# ============================
# BEST MARKET EACH MONTH
# ============================
best_each_month = monthly.loc[
    monthly.groupby("YearMonth")["Monthly_Return"].idxmax()
]

print("\nBest performing market each month:")
print(best_each_month[["YearMonth", "Market", "Monthly_Return"]])

best_each_month.to_csv("best_market_each_month.csv", index=False)
print("\nbest_market_each_month.csv saved successfully")
# ================================
# BEST MARKET EACH MONTH (REQUIRED)
# ================================

best_each_month = monthly.loc[
    monthly.groupby("YearMonth")["Monthly_Return"].idxmax()
]

print("\nBest performing market each month:")
print(best_each_month[["YearMonth", "Market", "Monthly_Return"]])

best_each_month.to_csv("best_market_each_month.csv", index=False)
print("best_market_each_month.csv saved successfully")