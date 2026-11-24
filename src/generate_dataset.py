"""
Step 1: Create a sample symptoms -> disease dataset.

Why? Real medical datasets (for example from Kaggle) cannot be shipped with
this project, so this script builds a small practice dataset.
For each disease we list its common symptoms and how often (probability)
each symptom shows up. Then we create many random "patients".

Run:  python src/generate_dataset.py
"""

import random
import pandas as pd

random.seed(42)  # same result every time

# disease -> {symptom: chance of the symptom appearing (0 to 1)}
disease_symptoms = {
    "Common Cold": {"runny_nose": 0.9, "sneezing": 0.8, "sore_throat": 0.6,
                    "cough": 0.6, "blocked_nose": 0.7, "headache": 0.3, "fever": 0.2},
    "Influenza": {"fever": 0.9, "chills": 0.7, "body_ache": 0.9, "fatigue": 0.8,
                  "cough": 0.7, "headache": 0.6, "sore_throat": 0.4},
    "Allergic Rhinitis": {"sneezing": 0.9, "itchy_eyes": 0.8, "runny_nose": 0.8,
                          "blocked_nose": 0.6, "itching": 0.4},
    "Migraine": {"headache": 0.95, "sensitivity_to_light": 0.8, "nausea": 0.7,
                 "vomiting": 0.4, "fatigue": 0.4},
    "Acid Reflux": {"heartburn": 0.95, "acid_taste": 0.8, "bloating": 0.5,
                    "stomach_pain": 0.5, "nausea": 0.3, "difficulty_swallowing": 0.2},
    "Gastroenteritis": {"diarrhea": 0.9, "vomiting": 0.8, "nausea": 0.8,
                        "stomach_pain": 0.8, "fever": 0.4, "loss_of_appetite": 0.6,
                        "fatigue": 0.4},
    "Sore Throat": {"sore_throat": 0.95, "difficulty_swallowing": 0.7, "fever": 0.4,
                    "cough": 0.4, "headache": 0.2},
    "Muscle Strain": {"muscle_pain": 0.95, "joint_stiffness": 0.6, "body_ache": 0.4,
                      "fatigue": 0.2},
    "Fungal Skin Infection": {"skin_rash": 0.9, "itching": 0.95, "joint_stiffness": 0.02},
    "Constipation": {"constipation": 0.95, "bloating": 0.7, "stomach_pain": 0.6,
                     "loss_of_appetite": 0.3, "nausea": 0.2},
}

# all symptom names in one list
all_symptoms = sorted({s for d in disease_symptoms.values() for s in d})

rows = []
patients_per_disease = 300

for disease, symptom_chances in disease_symptoms.items():
    for _ in range(patients_per_disease):
        row = {symptom: 0 for symptom in all_symptoms}

        # add the disease's own symptoms based on their chance
        for symptom, chance in symptom_chances.items():
            if random.random() < chance:
                row[symptom] = 1

        # add one random "noise" symptom sometimes (real data is messy)
        if random.random() < 0.10:
            row[random.choice(all_symptoms)] = 1

        # a patient needs at least one symptom
        if sum(row.values()) == 0:
            row[random.choice(list(symptom_chances))] = 1

        row["disease"] = disease
        rows.append(row)

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows
df.to_csv("data/symptoms_diseases.csv", index=False)

print("Dataset created:", df.shape)
print(df["disease"].value_counts())
