import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

print("All libraries loaded successfully!")

df = pd.read_csv("titanic.csv")
df.columns = df.columns.str.strip()
print(df.columns.tolist())
print(df.head())
# Load the Titanic dataset
df = sns.load_dataset("titanic")

# Display the first five rows
print("\nFirst 5 rows:")
print(df.head())

# Check dataset information
print("\nDataset information:")
print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Survival count
print("\nSurvival count:")
print(df["survived"].value_counts())

# Plot survival count
sns.countplot(data=df, x="survived")
plt.title("Titanic Survival Count")
plt.show()
# Survival by gender
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="sex", hue="survived")

plt.title("Titanic Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Passenger Count")
plt.legend(title="Survived (0 = No, 1 = Yes)")
plt.show()
# Survival by passenger class
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="pclass", hue="survived")

plt.title("Titanic Survival by Passenger Class")
plt.xlabel("Passenger Class (1 = First, 2 = Second, 3 = Third)")
plt.ylabel("Passenger Count")
plt.legend(title="Survived (0 = No, 1 = Yes)")

plt.tight_layout()
plt.show()
# Survival rate by gender
print("\nSurvival rate by gender:")
print(df.groupby("sex")["survived"].mean() * 100)

# Survival rate by passenger class
print("\nSurvival rate by passenger class:")
print(df.groupby("pclass")["survived"].mean() * 100)
# Survival by Passenger Class
# Check column names
print(df.columns.tolist())
print(df.head())

# Remove extra spaces from column names
df.columns = df.columns.str.strip()
# Survival by passenger class
sns.barplot(data=df, x="pclass", y="survived")

plt.title("Titanic Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()
plt.show()
# Survival rate by passenger class
survival_rate = df.groupby("pclass")["survived"].mean() * 100

print("\nSurvival Rate by Passenger Class:")
print(survival_rate)

survival_rate.plot(kind="bar")
plt.title("Titanic Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.tight_layout()
plt.show()
# Survival by age
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    hue="survived",
    bins=20,
    multiple="stack"
)

plt.title("Titanic Survival by Age")
plt.xlabel("Passenger Age")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.show()
# Survival by embarkation port
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="embarked", hue="survived")

plt.title("Titanic Survival by Embarkation Port")
plt.xlabel("Embarkation Port")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.show()