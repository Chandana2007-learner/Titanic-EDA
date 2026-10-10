import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

st.title("🚢 Titanic Exploratory Data Analysis")
st.write("Explore passenger survival patterns using six visualizations.")

Load the Titanic dataset

df = sns.load_dataset("titanic")

Dataset overview

st.header("1. Dataset Overview")
st.write("Number of rows:", df.shape[0])
st.write("Number of columns:", df.shape[1])
st.dataframe(df.head())

Survival count

st.header("2. Titanic Survival Count")
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x="survived", ax=ax)
ax.set_title("Titanic Survival Count")
ax.set_xlabel("Survived (0 = No, 1 = Yes)")
ax.set_ylabel("Passenger Count")
st.pyplot(fig)
plt.close(fig)

Survival by gender

st.header("3. Survival by Gender")
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x="sex", hue="survived", ax=ax)
ax.set_title("Titanic Survival by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Passenger Count")
ax.legend(title="Survived (0 = No, 1 = Yes)")
st.pyplot(fig)
plt.close(fig)

Survival by passenger class

st.header("4. Survival by Passenger Class")
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x="pclass", hue="survived", ax=ax)
ax.set_title("Titanic Survival by Passenger Class")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Passenger Count")
ax.legend(title="Survived (0 = No, 1 = Yes)")
st.pyplot(fig)
plt.close(fig)

Survival rate by passenger class

st.header("5. Survival Rate by Passenger Class")
survival_rate = df.groupby("pclass", observed=True)["survived"].mean() * 100
fig, ax = plt.subplots(figsize=(7, 4))
survival_rate.plot(kind="bar", ax=ax)
ax.set_title("Survival Rate by Passenger Class")
ax.set_xlabel("Passenger Class (1 = First, 2 = Second, 3 = Third)")
ax.set_ylabel("Survival Rate (%)")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

Survival by age

st.header("6. Survival by Age")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(
data=df,
x="age",
hue="survived",
bins=20,
multiple="stack",
ax=ax
)
ax.set_title("Titanic Survival by Age")
ax.set_xlabel("Passenger Age")
ax.set_ylabel("Passenger Count")
st.pyplot(fig)
plt.close(fig)

Survival by embarkation port

st.header("7. Survival by Embarkation Port")
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x="embarked", hue="survived", ax=ax)
ax.set_title("Titanic Survival by Embarkation Port")
ax.set_xlabel("Embarkation Port")
ax.set_ylabel("Passenger Count")
ax.legend(title="Survived (0 = No, 1 = Yes)")
st.pyplot(fig)
plt.close(fig)

Summary statistics

st.header("8. Survival Statistics")
st.write("Survival count")
st.write(df["survived"].value_counts().rename(index={0: "Did not survive", 1: "Survived"}))

st.write("Survival rate by gender (%)")
st.write((df.groupby("sex", observed=True)["survived"].mean() * 100).round(2))

st.write("Survival rate by passenger class (%)")
st.write((df.groupby("pclass", observed=True)["survived"].mean() * 100).round(2))
