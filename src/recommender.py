"""
Step 3: The recommendation logic used by the Streamlit app.

1. predict_disease()          -> symptoms to likely diseases (ML model)
2. recommend_medicines()      -> medicines for a disease, personalised by age and allergies
3. find_similar_medicines()   -> alternative medicines using text similarity (NLTK + TF-IDF)
"""

import joblib
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load saved files ----------
model = joblib.load("models/model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
symptom_list = joblib.load("models/symptoms.pkl")

# keep_default_na=False so the word "None" is read as text, not as a missing value
medicines = pd.read_csv("data/medicines.csv", keep_default_na=False)
disease_info = pd.read_csv("data/disease_info.csv")


# ---------- 1. Disease prediction ----------
def predict_disease(selected_symptoms, top_n=3):
    """Return the top N likely diseases with their probability."""
    # build one row: 1 if the symptom was selected, otherwise 0
    row = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
    row_df = pd.DataFrame([row], columns=symptom_list)

    probabilities = model.predict_proba(row_df)[0]

    # sort diseases from highest to lowest probability
    results = []
    for number, prob in enumerate(probabilities):
        disease_name = label_encoder.inverse_transform([number])[0]
        results.append((disease_name, float(prob)))
    results.sort(key=lambda item: item[1], reverse=True)

    return results[:top_n]


# ---------- 2. Personalised medicine recommendation ----------
def recommend_medicines(disease, age, allergies):
    """
    Return two lists:
      safe     -> medicines suitable for this person
      excluded -> medicines removed, with the reason (age or allergy)
    """
    safe = []
    excluded = []

    for _, row in medicines.iterrows():
        # does this medicine treat the disease?
        if disease not in row["used_for"].split(";"):
            continue

        # check age
        if age < row["min_age"]:
            excluded.append((row["medicine"], f"minimum age is {row['min_age']}"))
        # check allergy
        elif row["allergy_group"] in allergies:
            excluded.append((row["medicine"], f"you listed an allergy to {row['allergy_group']}"))
        else:
            safe.append(row)

    # keep the column names even when the list is empty
    safe_df = pd.DataFrame(safe, columns=medicines.columns)
    return safe_df, excluded


def get_disease_info(disease):
    """Return description, self-care tips and warning signs for a disease."""
    row = disease_info[disease_info["disease"] == disease].iloc[0]
    return row


# ---------- 3. Similar medicines (NLP) ----------
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r"[a-z]+")   # keeps only words


def clean_text(text):
    """Lowercase, split into words, and reduce each word to its root (stemming)."""
    words = tokenizer.tokenize(text.lower())
    return " ".join(stemmer.stem(word) for word in words)


# one text description per medicine
medicines["text"] = (
    medicines["drug_class"] + " " + medicines["used_for"] + " " + medicines["side_effects"]
)
medicines["clean_text"] = medicines["text"].apply(clean_text)

# turn the text into numbers with TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(medicines["clean_text"])


def find_similar_medicines(medicine_name, top_n=3):
    """Return medicines most similar to the given one."""
    position = medicines.index[medicines["medicine"] == medicine_name][0]

    # similarity of this medicine with all medicines
    scores = cosine_similarity(tfidf_matrix[position], tfidf_matrix)[0]

    # sort, skip the medicine itself (it is always the most similar to itself)
    ranked = scores.argsort()[::-1]
    results = []
    for index in ranked:
        if index != position:
            results.append((medicines.loc[index, "medicine"], float(scores[index])))
        if len(results) == top_n:
            break
    return results
