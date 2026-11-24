# 💊 Personalized Healthcare & Medicine Recommendation System

A beginner-friendly, end-to-end machine learning project. The user selects their
symptoms, age and allergies. The app predicts the most likely condition and suggests
medicines that are **safe for that person's profile**.

> ⚠️ **Disclaimer:** This is an educational project, **not medical advice**.
> The sample data is simplified. Always consult a doctor or pharmacist before taking any medicine.

## What it does

1. **Disease prediction**: ML models (Logistic Regression, Random Forest, XGBoost) predict the top 3 likely conditions from symptoms.
2. **Personalized medicine recommendation**: medicines are filtered by the user's **age** and **allergies**, and the app explains why a medicine was removed.
3. **Alternative medicines**: NLTK text cleaning (tokenizing and stemming) plus TF-IDF and cosine similarity find similar medicines.
4. **Streamlit web app** that ties it all together.

## Tech stack

Python · pandas · scikit-learn · XGBoost · NLTK · Streamlit · joblib · GitHub

## Project structure

```
healthcare-recommender/
├── app.py                      # Streamlit web app (run this)
├── requirements.txt
├── data/
│   ├── medicines.csv           # medicine info (class, uses, side effects, allergy group, min age)
│   ├── disease_info.csv        # description, self-care tips, warning signs
│   └── symptoms_diseases.csv   # training data (created by generate_dataset.py)
├── models/                     # saved model files (created by train_model.py)
└── src/
    ├── generate_dataset.py     # step 1: create the sample training data
    ├── train_model.py          # step 2: train and compare models, save the best
    └── recommender.py          # step 3: prediction + recommendation logic
```

## How to run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 2. install libraries
pip install -r requirements.txt

# 3. (optional) rebuild the data and the model, the ZIP already includes them
python src/generate_dataset.py
python src/train_model.py

# 4. start the app
streamlit run app.py
```

## How it works

| Step | File | Idea |
|------|------|------|
| Data | `generate_dataset.py` | Each patient is a row of 0/1 symptom columns plus the disease name |
| Training | `train_model.py` | 80/20 split, three models compared by accuracy and F1, best one saved |
| Prediction | `recommender.py` | Selected symptoms become a 0/1 row and the model gives probabilities |
| Personalization | `recommender.py` | Skip medicines where age is below the minimum or the allergy group matches |
| Alternatives | `recommender.py` | Clean text, TF-IDF vectors, cosine similarity |

## Model results (on the included sample data)

Run `python src/train_model.py` to see the scores. They are saved in `models/model_results.csv`.

**Important:** the included dataset is **synthetic** (generated from simple symptom
patterns), so the near-perfect scores do **not** reflect real-world performance.
Real clinical data is noisier and results would be lower.

## Using a real dataset (recommended upgrade)

Download a symptom–disease dataset from Kaggle (for example "Disease Symptom Prediction"),
convert it so that each symptom is a 0/1 column and there is one `disease` column, and save it as
`data/symptoms_diseases.csv`. Then update `data/disease_info.csv` and `data/medicines.csv`
for the new disease names and re-run `python src/train_model.py`. No code changes are needed.

## Ideas to extend

- Add more diseases, symptoms and medicines
- Add cross-validation and a confusion matrix plot
- Add drug-interaction checks
- Deploy on Streamlit Community Cloud

## Push to GitHub

```bash
git init
git add .
git commit -m "Healthcare and medicine recommendation system"
git branch -M main
git remote add origin https://github.com/<your-username>/healthcare-recommender.git
git push -u origin main
```

## Resume line

*Built an end-to-end symptom-based disease prediction and personalized medicine recommendation system
(Logistic Regression / Random Forest / XGBoost, NLTK + TF-IDF similarity) with age and allergy filtering;
deployed as a Streamlit web app.*
