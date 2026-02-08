import streamlit as st
import pickle
import numpy as np

def format_price(value):
    if value >= 100:
        return f"₹ {round(value/100,2)} Crores"
    elif value >= 1:
        return f"₹ {round(value,2)} Lakhs"
    else:
        return f"₹ {round(value*100,2)} Thousands"

# Load models
clf = pickle.load(open("clf.pkl", "rb"))
reg = pickle.load(open("reg.pkl", "rb"))

st.title("🏠 Real Estate Investment Advisor")

st.write("Enter property details below:")

# User inputs
size = st.number_input("Property Size (SqFt)")
bhk = st.number_input("BHK")
price = st.number_input("Current Price (Lakhs)")

if st.button("Predict"):
    data = np.array([[size, bhk, price]])

    invest = clf.predict(data)
    future = reg.predict(data)

    st.subheader("Results")

    if invest[0] == 1:
        st.success("✅ Good Investment")
    else:
        st.error("❌ Not a Good Investment")

    st.success(f"Estimated Price After 5 Years: {format_price(future[0])}")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data.csv")

st.subheader("Price Distribution")

fig, ax = plt.subplots()
df['Price_in_Lakhs'].hist(ax=ax, bins=40)
ax.set_xlabel("Price in Lakhs")
ax.set_ylabel("Number of Properties")
ax.set_title("Property Price Distribution")
st.pyplot(fig)


st.subheader("Size vs Price")

fig, ax = plt.subplots()
ax.scatter(df['Size_in_SqFt'], df['Price_in_Lakhs'], alpha=0.2)
st.pyplot(fig)

st.subheader("BHK Distribution")

fig, ax = plt.subplots()
df['BHK'].value_counts().plot(kind='bar', ax=ax)
st.pyplot(fig)

