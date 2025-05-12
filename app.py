import streamlit as st
import pandas as pd
import re
import ast
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

st.set_page_config(layout="wide")

# Загрузка необходимых ресурсов
nltk.download("punkt")
nltk.download("stopwords")

# --- Словарь: номер аптеки → район ---
pharmacy_districts = {
    '34': 'Жетысуский', '40': 'Бостандыкский', '57': 'Жетысуский', '58': 'Алатауский',
    '59': 'Бостандыкский', '74': 'Медеуский', '75': 'Алмалинский', '81': 'Ауэзовский',
    '83': 'Алмалинский', '91': 'Медеуский', '94': 'Алмалинский', '132': 'Турксибский',
    '175': 'Жетысуский', '222': 'Алмалинский', '234': 'Жетысуский', '260': 'Ауэзовский',
    '263': 'Ауэзовский', '294': 'Медеуский', '309': 'Бостандыкский', '311': 'Алмалинский',
    '320': 'Турксибский', '323': 'Ауэзовский'
}

def get_districts(pharmacies_str):
    try:
        pharmacies = ast.literal_eval(pharmacies_str)
        districts = set()
        for p in pharmacies:
            numbers = re.findall(r"\u2116\s*(\d+)", p)
            for number in numbers:
                if number in pharmacy_districts:
                    districts.add(pharmacy_districts[number])
        return list(districts)
    except:
        return []

# --- Стемминг ---
stemmer = SnowballStemmer("russian")
russian_stopwords = stopwords.words("russian")

def stem_text(text):
    if not isinstance(text, str):
        return ""
    words = nltk.word_tokenize(text.lower())
    stemmed_words = [stemmer.stem(word) for word in words if word.isalpha()]
    return " ".join(stemmed_words)

@st.cache_data
def load_data():
    df = pd.read_csv("processed_products_data.csv")
    df["Категории_строка"] = df["Категории"].apply(
        lambda x: " ".join(eval(x)) if isinstance(x, str) and x.startswith("[") else ""
    )
    df["text"] = (
        df["Название"].fillna("") + " " +
        df["Описание"].fillna("") + " " +
        df["Категории_строка"].fillna("")
    )
    df["text_stemmed"] = df["text"].apply(stem_text)
    df["Районы"] = df["Аптеки"].apply(get_districts)
    return df

data = load_data()

@st.cache_resource
def fit_tfidf(texts):
    tfidf = TfidfVectorizer(stop_words=russian_stopwords, max_features=5000)
    matrix = tfidf.fit_transform(texts)
    return tfidf, matrix

tfidf_stemmed, tfidf_matrix_stemmed = fit_tfidf(data["text_stemmed"])

def search_products(query, top_n=5, selected_district=None):
    query_stemmed = stem_text(query)
    query_vec = tfidf_stemmed.transform([query_stemmed])
    similarity = cosine_similarity(query_vec, tfidf_matrix_stemmed).flatten()
    top_indices = similarity.argsort()[-top_n:][::-1]
    results = data.iloc[top_indices].copy()
    results["Score"] = similarity[top_indices]

    if selected_district:
        results = results[results["Районы"].apply(lambda r: selected_district in r if r else False)]

    return results[["Название", "Описание", "Цена", "Аптеки", "Районы", "Изображение"]]

# --- AI рекомендация ---
API_KEY = os.getenv("TOKEN")
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

def get_product_recommendations(product_name):
    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{
                "role": "user",
                "content": f"Опиши кратко товар {product_name} и дай основную рекомендацию в одном предложении на русском."
            }]
        )
        if completion.choices:
            return {"description": completion.choices[0].message.content.strip()}
        else:
            return {"error": "Информация не найдена для этого товара"}
    except Exception as e:
        return {"error": f"Ошибка при запросе к API: {str(e)}"}

# --- Интерфейс ---
st.title("💊 Поиск лекарств в аптеках Алматы")

query = st.text_input("Введите название лекарства или симптомы:")
districts = sorted(set(sum(data["Районы"], [])))
selected_district = st.selectbox("Выберите район:", [""] + districts)

if query:
    results = search_products(query, top_n=10, selected_district=selected_district)

    if results.empty:
        st.warning("Ничего не найдено для выбранного района.")
    else:
        for _, row in results.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 6])
                with col1:
                    if "Изображение" in row and isinstance(row["Изображение"], str):
                        st.image(row["Изображение"], width=200)
                with col2:
                    st.subheader(row["Название"])
                    st.write(f"Цена: {row['Цена']} тг")

                    with st.expander("Описание"):
                        st.write(row["Описание"])

                    if selected_district:
                        pharmacies = ast.literal_eval(row["Аптеки"])
                        filtered = []
                        for ph in pharmacies:
                            nums = re.findall(r"\u2116\s*(\d+)", ph)
                            for n in nums:
                                if pharmacy_districts.get(n) == selected_district:
                                    url = f"https://europharma.kz/map#{n}"
                                    filtered.append(f"[Аптека №{n}]({url})")
                        if filtered:
                            st.markdown(f"Аптеки в {selected_district}: " + ", ".join(filtered))

                    if st.button(f"Рекомендация от ИИ: {row['Название']}", key=row['Название']):
                        recommendation = get_product_recommendations(row['Название'])
                        st.info(recommendation.get("description", "Информация недоступна"))