import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

import streamlit as st
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import streamlit as st
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# تابع برای بارگذاری داده‌ها از فایل CSV
def load_chat_data_from_csv():
    chat_data = {}
    with open("C:/Users/Arghavan Computer/Desktop/teststrim/pythonProject/holosenstream/holosenex.csv", "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            chat_data[row[0]] = row[1]
    return chat_data


# آموزش مدل
def train_model(chat_data):
    X = list(chat_data.keys())
    y = list(chat_data.values())
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vectorized, y)

    with open("chat_model.pkl", "wb") as model_file:
        pickle.dump((model, vectorizer), model_file)


# پیش‌بینی پاسخ
def predict_response(user_input):
    with open("chat_model.pkl", "rb") as model_file:
        model, vectorizer = pickle.load(model_file)
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_response = model.predict(user_input_vectorized)
    return predicted_response[0]


# تنظیمات اولیه Streamlit
st.title("چت‌بات هوشمند - مکالمه پیوسته")

# آموزش مدل اگر اولین بار است که اجرا می‌شود
if "trained" not in st.session_state:
    chat_data = load_chat_data_from_csv()
    train_model(chat_data)
    st.session_state["trained"] = True
    st.session_state["chat_history"] = []

# دریافت ورودی کاربر
user_input = st.text_input("سوال یا پیام خود را وارد کنید:")

# اگر کاربر چیزی وارد کرد
if user_input:
    # پیش‌بینی پاسخ و اضافه کردن به تاریخچه
    response = predict_response(user_input)
    st.session_state.chat_history.append(("شما:", user_input))
    st.session_state.chat_history.append(("چت‌بات:", response))

# نمایش تاریخچه مکالمه
for sender, message in st.session_state.chat_history:
    st.write(f"**{sender}** {message}")

