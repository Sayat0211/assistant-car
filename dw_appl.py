# === Добавим в начало, сразу после импортов ===
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# === Page Config ===
st.set_page_config(
    page_title="🚘 Car Assistant",
    page_icon="🧠",
    layout="wide"
)

# === THEME SWITCHER ===
theme = st.radio("🌗 Choose Theme", ["Light mode", "Dark mode"], horizontal=True)

if theme == "Light mode":
    background_color = "#f5f0e6"
    text_color = "#1f1f1f"
    button_color = "#e0dbd1"
    hover_color = "#d1cfc7"
else:
    background_color = "#1e1e1e"
    text_color = "#f0f0f0"
    button_color = "#333333"
    hover_color = "#444444"

# === Dynamic CSS based on theme ===
st.markdown(f"""
    <style>
        html, body, [class*="css"], .main, .block-container {{
            background-color: {background_color} !important;
            color: {text_color} !important;
            font-size: 18px !important;        /* Larger base text */
            font-weight: 500 !important;       /* Slightly bold */
        }}

        /* General text styles */
        h1, h2, h3, h4, h5, h6, p, span, label, div {{
            color: {text_color} !important;
            font-weight: 600 !important;       /* Bold headers and labels */
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {button_color};
            color: {text_color};
            border: 1px solid #888;
            padding: 0.5rem 1rem;
            border-radius: 12px;
            transition: all 0.2s ease-in-out;
            font-size: 16px;
            font-weight: 600;
        }}
        .stButton > button:hover {{
            background-color: {hover_color};
            transform: scale(1.05);
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {hover_color};
            border-radius: 12px;
        }}
        .stTabs [data-baseweb="tab"] {{
            font-weight: bold;
            padding: 0.5rem 1.2rem;
            color: {text_color} !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: {button_color};
            border-radius: 10px;
        }}

        /* Inputs: text, select, number */
        input, textarea, select {{
            background-color: {button_color} !important;
            color: {text_color} !important;
            border: 1px solid #aaa !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: 500 !important;
        }}

        /* Placeholder text */
        ::placeholder {{
            color: {text_color}99 !important;
            font-size: 15px !important;
        }}
    </style>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    fuel_df = pd.read_csv("FuelType.csv")
    transmission_df = pd.read_csv("transmission_descriptions.csv")
    car_type_df = pd.read_csv("CarType.csv")
    car_df = pd.read_csv("22613data.csv")
    car_df = car_df.drop(['City','Volume'], axis=1)

    for df in [fuel_df, transmission_df, car_type_df]:
        df.rename(columns={"Описание": "Description", "Тип": "Type"}, inplace=True)

    return fuel_df, transmission_df, car_type_df, car_df

fuel_df, transmission_df, car_type_df, raw_data = load_data()

categorical_cols = ['Company', 'Mark', 'Fuel Type', 'Transmission', 'Car_type']

def remove_outliers(data, column):
    Q1, Q3 = data[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

@st.cache_data
def preprocess_data(data):
    df = data.drop_duplicates()
    df.fillna({'Mark': 'Unknown', 'Fuel Type': 'Unknown', 'Transmission': 'Unknown'}, inplace=True)
    df['Year'] = df['Year'].fillna(df['Year'].median()).astype(int)
    df['Mileage'] = df['Mileage'].fillna(df['Mileage'].median())

    df = remove_outliers(df, 'Price')
    df = remove_outliers(df, 'Mileage')

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).str.upper())
        encoders[col] = le

    return df, encoders

df, encoders = preprocess_data(raw_data)

@st.cache_resource
def train_model(df):
    X = df.drop('Price', axis=1)
    y = df['Price']
    model = RandomForestRegressor(n_estimators=350, random_state=4)
    model.fit(X, y)
    return model

model = train_model(df)

@st.cache_resource
def build_tfidf(df, column='Description'):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df[column])
    return tfidf, tfidf_matrix

fuel_tfidf, fuel_matrix = build_tfidf(fuel_df)
trans_tfidf, trans_matrix = build_tfidf(transmission_df)
body_tfidf, body_matrix = build_tfidf(car_type_df)

def get_best_match(query, df, tfidf, matrix):
    query_vec = tfidf.transform([query])
    similarities = cosine_similarity(query_vec, matrix).flatten()
    df['Similarity'] = similarities
    return df.loc[df['Similarity'].idxmax()]

# === Interface ===
st.title("🚘 Car Assistant")

tabs = st.tabs(["🔍 Match by Description", "💰 Estimate Price","📆 Credit Calc"])

# === Tab 1 ===
with tabs[0]:
    st.markdown("### 🧾 Describe your dream car and let us recommend fuel type, transmission, and body type:")
    #default_text = "I want a small city car"
    user_input = st.text_area("💬 Your request:")

    if st.button("✨ Find Best Match", key="desc_button"):
        if user_input.strip() == "":
            st.warning("🚨 Please enter a description.")
        else:
            corrected_query = correct_text(user_input)
            best_fuel = get_best_match(corrected_query, fuel_df, fuel_tfidf, fuel_matrix)
            best_transmission = get_best_match(corrected_query, transmission_df, trans_tfidf, trans_matrix)
            best_car_type = get_best_match(corrected_query, car_type_df, body_tfidf, body_matrix)

            st.markdown("### ✅ Suggested Specs:")
            st.markdown(f"**⛽ Fuel Type:** `{best_fuel['Type']}` — {best_fuel['Description']}")
            st.markdown(f"**⚙️ Transmission:** `{best_transmission['Type']}` — {best_transmission['Description']}")
            st.markdown(f"**🚗 Car Body:** `{best_car_type['Type']}` — {best_car_type['Description']}")

# === Tab 2 ===
with tabs[1]:
    st.markdown("### 📊 Enter your car’s features to get a price estimate:")

    company = st.selectbox("🏢 Manufacturer", sorted(raw_data['Company'].dropna().unique()))
    filtered_data = raw_data[raw_data['Company'] == company]
    mark = st.selectbox("🚘 Model", sorted(filtered_data['Mark'].dropna().unique()))
    year = st.number_input("📅 Year", 1990, 2025, 2015)
    fuel = st.selectbox("⛽ Fuel Type", sorted(raw_data['Fuel Type'].dropna().unique()))
    trans = st.selectbox("⚙️ Transmission", sorted(raw_data['Transmission'].dropna().unique()))
    mileage = st.number_input("🛣️ Mileage (km)", 0, 1_000_000, 100_000)
    car_type = st.selectbox("🚗 Body Type", sorted(raw_data['Car_type'].dropna().unique()))

    if st.button("📈 Estimate Price", key="price_button"):
        new_car = pd.DataFrame({
            'Company': [company],
            'Mark': [mark],
            'Year': [year],
            'Fuel Type': [fuel],
            'Transmission': [trans],
            'Mileage': [mileage],
            'Car_type': [car_type]
        })

        try:
            for col in categorical_cols:
                new_car[col] = new_car[col].astype(str).str.upper()
                if any(v not in encoders[col].classes_ for v in new_car[col]):
                    raise ValueError(f"❌ Unknown value in column '{col}'")
                new_car[col] = encoders[col].transform(new_car[col])

            pred = model.predict(new_car)[0]
            st.success(f"💵 Estimated Price: **{int(pred):,} ₸**")
        except ValueError as e:
            st.error(str(e))
# === Tab 3: Credit Calculator ===

with tabs[2]:
    st.markdown("### 💳 Credit Calculator")

    # Inputs
    car_price = st.number_input("Car Price (₸)", min_value=100000, value=1000000, step=10000)
    down_payment=st.number_input("Down Payment (₸)", min_value=0, max_value=car_price, value=int(car_price * 0.2), step=10000)
    term = st.slider("Term (months)", 6, 84, 36, step=6)
    rate = st.slider("Interest (%/yr)", 0.0, 100.0, 10.0, step=0.1)

    # Calculation
    if car_price > down_payment:
        loan = car_price - down_payment
        monthly_rate = (rate / 100) / 12

        if rate > 0:
            m = monthly_rate
            monthly = loan * (m * (1 + m)**term) / ((1 + m)**term - 1)
        else:
            monthly = loan / term

        st.success(f"Monthly: **{int(monthly):,} ₸**")
    else:
        st.warning("Down payment >= price")

