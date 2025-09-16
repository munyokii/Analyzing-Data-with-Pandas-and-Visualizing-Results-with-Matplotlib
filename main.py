"""
Analyzing Data with Pandas and Visualizing Results with Matplotlib
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# --------------------------
# Loading and Exploring Dataset
# --------------------------
try:
    # Loading the iris dataset
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame

    print('Dataset loaded successfully.\n')
except FileNotFoundError:
    print("Error: Dataset file not found.")
except ImportError:
    print("Error: Required module not found. Please install scikit-learn.")
except ValueError as e:
    print(f"Error loading dataset: {e}")

print("First five rows of the dataset:")
print(df.head(), '\n')

print("Dataset Information:")
print(df.info(), '\n')

print("Missing Values:")
print(df.isnull().sum(), '\n')

# Cleaning dataset
df = df.dropna()

# Data Analysis with Pandas
# Statistics
print("Descriptive Statistics:")
print(df.describe(), '\n')

# Grouping by species and computing mean
grouped_means = df.groupby("target").mean()
print("Mean values by species (target):")
print(grouped_means, "\n")

# Using pandas to extract one column of interest
avg_sepal_length = df.groupby("target")["sepal length (cm)"].mean()
print("Average Sepal Length by Species:\n", avg_sepal_length, "\n")

# Adding a derived column using pandas
df["sepal_petal_ratio"] = df["sepal length (cm)"] / df["petal length (cm)"]
print("New column (sepal/petal ratio) added:\n")
print(df[["sepal length (cm)", "petal length (cm)", "sepal_petal_ratio"]].head(), "\n")

# Styling and Visualization
sns.set_theme(style="whitegrid")

# 1. Line Chart
df["petal length (cm)"].plot(
    figsize=(8,5), title="Line Chart: Petal Length Trend Across Samples", label="Petal Length"
)
plt.xlabel("Sample Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart
avg_petal_length = df.groupby("target")["petal length (cm)"].mean()
avg_petal_length.plot(kind="bar", figsize=(8,5), color=["#440154","#31688e","#35b779"])
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species (0=Setosa, 1=Versicolor, 2=Virginica)")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram
df["sepal length (cm)"].plot(
    kind="hist", bins=15, color="skyblue", edgecolor="black", figsize=(8,5),
    title="Histogram: Sepal Length Distribution"
)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot
sns.scatterplot(
    x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="deep"
)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
