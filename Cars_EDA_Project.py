"""
Exploratory Data Analysis on a Cars Dataset
Dataset: ggplot2 "mpg" dataset (234 observations; 11 variables)
Source: https://ggplot2.tidyverse.org/reference/mpg.html
No machine-learning model is used.

Install requirements:
    pip install pandas matplotlib seaborn plotnine

Run:
    python Cars_EDA_Project.py

The script loads the dataset from plotnine's packaged copy. If plotnine is not
installed, install it using the command above. All charts are saved to the
current working directory.
"""
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the public dataset
try:
    from plotnine.data import mpg
    df = mpg.copy()
except ImportError as exc:
    raise SystemExit("Please install plotnine: pip install plotnine") from exc

# 2. Inspect the dataset
print("\\nDATASET SHAPE:", df.shape)
print("\\nCOLUMNS:", list(df.columns))
print("\\nDATA TYPES:\\n", df.dtypes)
print("\\nFIRST FIVE ROWS:\\n", df.head())
print("\\nMISSING VALUES:\\n", df.isna().sum())
print("\\nDUPLICATE ROWS:", df.duplicated().sum())

# 3. Clean and prepare data
df.columns = [column.strip().lower() for column in df.columns]
df["manufacturer"] = df["manufacturer"].astype(str).str.strip().str.lower()
df["model"] = df["model"].astype(str).str.strip()
df["transmission_type"] = df["trans"].astype(str).str.extract(r"^(auto|manual)", expand=False)
df["combined_mpg"] = (df["cty"] + df["hwy"]) / 2

rows_before = len(df)
df = df.drop_duplicates().copy()
required = ["manufacturer", "year", "displ", "cyl", "transmission_type",
            "cty", "hwy", "class"]
df = df.dropna(subset=required).copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
df["cyl"] = pd.to_numeric(df["cyl"], errors="coerce").astype(int)
df["displ"] = pd.to_numeric(df["displ"], errors="coerce")
df["cty"] = pd.to_numeric(df["cty"], errors="coerce")
df["hwy"] = pd.to_numeric(df["hwy"], errors="coerce")
df["combined_mpg"] = (df["cty"] + df["hwy"]) / 2
print("\\nROWS BEFORE CLEANING:", rows_before)
print("ROWS AFTER CLEANING:", len(df))
print("\\nNUMERICAL SUMMARY:\\n", df.select_dtypes(include="number").describe())
print("\\nCATEGORICAL SUMMARY:\\n", df.select_dtypes(exclude="number").describe())

# 4. Univariate, bivariate and multivariate analysis
print("\\nFREQUENCY: MANUFACTURER\\n", df["manufacturer"].value_counts())
print("\\nFREQUENCY: VEHICLE CLASS\\n", df["class"].value_counts())
print("\\nAVERAGE FUEL ECONOMY BY TRANSMISSION\\n",
      df.groupby("transmission_type")["combined_mpg"].mean().round(2))
print("\\nAVERAGE FUEL ECONOMY BY CLASS\\n",
      df.groupby("class")["combined_mpg"].mean().sort_values(ascending=False).round(2))
print("\\nCORRELATION MATRIX\\n",
      df[["displ", "cyl", "cty", "hwy", "combined_mpg", "year"]].corr().round(2))

# IQR-based outlier flags are reported, not automatically deleted.
for column in ["displ", "cty", "hwy", "combined_mpg"]:
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    count = ((df[column] < q1 - 1.5 * iqr) |
             (df[column] > q3 + 1.5 * iqr)).sum()
    print(f"IQR outlier flags for {column}: {count}")

# 5. Visualisations (7 charts, 5 chart types)
plt.figure(figsize=(8, 5))
plt.hist(df["combined_mpg"], bins=18, edgecolor="black")
plt.title("Distribution of Average Fuel Economy")
plt.xlabel("Average fuel economy (miles per gallon)")
plt.ylabel("Number of cars")
plt.tight_layout()
plt.savefig("chart1_fuel_economy_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
df["manufacturer"].value_counts().head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Manufacturers by Number of Records")
plt.xlabel("Number of car records")
plt.ylabel("Manufacturer")
plt.tight_layout()
plt.savefig("chart2_manufacturers.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 5))
df.boxplot(column="combined_mpg", by="transmission_type", grid=False)
plt.suptitle("")
plt.title("Fuel Economy by Transmission Type")
plt.xlabel("Transmission type")
plt.ylabel("Average fuel economy (mpg)")
plt.tight_layout()
plt.savefig("chart3_transmission_boxplot.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["displ"], df["combined_mpg"], alpha=0.65)
plt.title("Engine Displacement vs Average Fuel Economy")
plt.xlabel("Engine displacement (litres)")
plt.ylabel("Average fuel economy (mpg)")
plt.tight_layout()
plt.savefig("chart4_displacement_scatter.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["cty"], df["hwy"], alpha=0.65)
plt.title("City vs Highway Fuel Economy")
plt.xlabel("City fuel economy (mpg)")
plt.ylabel("Highway fuel economy (mpg)")
plt.tight_layout()
plt.savefig("chart5_city_highway_scatter.png", dpi=150)
plt.show()

corr = df[["displ", "cyl", "cty", "hwy", "combined_mpg", "year"]].corr()
plt.figure(figsize=(7, 5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("chart6_correlation_heatmap.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
df.groupby("class")["combined_mpg"].mean().sort_values().plot(kind="barh")
plt.title("Average Fuel Economy by Vehicle Class")
plt.xlabel("Average fuel economy (mpg)")
plt.ylabel("Vehicle class")
plt.tight_layout()
plt.savefig("chart7_class_fuel_economy.png", dpi=150)
plt.show()

print("\\nEDA completed. No machine-learning model was trained.")
