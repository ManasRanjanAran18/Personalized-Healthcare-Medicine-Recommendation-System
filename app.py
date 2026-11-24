"""
Step 4: Streamlit web app.

Run:  streamlit run app.py
"""

import pandas as pd
import streamlit as st
from src.recommender import (
    symptom_list,
    medicines,
    predict_disease,
    recommend_medicines,
    get_disease_info,
    find_similar_medicines,
)

st.set_page_config(page_title="Healthcare & Medicine Recommender", page_icon="💊")

st.title("💊 Personalized Healthcare & Medicine Recommender")

# st.warning(
#     "This is an educational project, NOT medical advice. "
#     "Always talk to a doctor or pharmacist before taking any medicine."
# )

# ---------- Sidebar: patient profile (personalization) ----------
st.sidebar.header("Your profile")
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)

allergy_options = sorted(medicines["allergy_group"].unique())
allergy_options = [a for a in allergy_options if a != "None"]
allergies = st.sidebar.multiselect("Known allergies / sensitivities", allergy_options)

# ---------- Step 1: choose symptoms ----------
st.header("1. Select your symptoms")

# show nice names (runny_nose -> Runny nose)
nice_names = {s: s.replace("_", " ").capitalize() for s in symptom_list}
chosen = st.multiselect(
    "Pick all symptoms you have:",
    options=symptom_list,
    format_func=lambda s: nice_names[s],
)

if st.button("Get recommendations"):
    if len(chosen) < 2:
        st.error("Please select at least 2 symptoms for a better prediction.")
    else:
        # ---------- Step 2: disease prediction ----------
        st.header("2. Possible conditions")
        predictions = predict_disease(chosen, top_n=2)

        chart_data = pd.DataFrame(
            {"Probability (%)": [round(p * 100, 1) for _, p in predictions]},
            index=[d for d, _ in predictions],
        )
        st.bar_chart(chart_data)

        top_disease, top_prob = predictions[0]
        if top_prob < 0.5:
            st.info("The model is not very sure. Try adding more symptoms or see a doctor.")

        # ---------- Step 3: info about the top condition ----------
        info = get_disease_info(top_disease)
        st.subheader(f"Most likely: {top_disease}")
        st.write(info["description"])
        st.write("**Self-care tips:** " + info["self_care"])
        st.write("**See a doctor if:** " + info["see_doctor_if"])

        # ---------- Step 4: medicines ----------
        st.header("3. Medicines to discuss with your doctor")
        safe, excluded = recommend_medicines(top_disease, age, allergies)

        if len(safe) == 0:
            st.write("No suitable medicines found for your profile. Please consult a doctor.")
        else:
            table = safe[["medicine", "drug_class", "side_effects"]].copy()
            table.columns = ["Medicine", "Type", "Possible side effects"]
            st.table(table.reset_index(drop=True))

        if len(excluded) > 0:
            with st.expander("Medicines removed because of your profile"):
                for name, reason in excluded:
                    st.write(f"- **{name}**: {reason}")

        # ---------- Step 5: similar medicines ----------
        if len(safe) > 0:
            st.header("4. Alternatives")
            first_medicine = safe.iloc[0]["medicine"]
            st.write(f"Medicines similar to **{first_medicine}**:")
            for name, score in find_similar_medicines(first_medicine, top_n=3):
                st.write(f"- {name} (similarity {score:.2f})")
