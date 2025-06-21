import pandas as pd
import pickle
import cloudpickle
import streamlit as st
import lime.lime_tabular
import shap
import captum
from interpret.blackbox import LimeTabular
from interpret import show
from interpret.blackbox import ShapKernel
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from joblib import load

set_visualize_provider(InlineProvider())
from interpret import show
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from captum.attr import Saliency
import numpy as np
import matplotlib.pyplot as plt


# Define provided data
data = """
0= Empty
1= Exogenous intoxication
2= Coma
3= Severe traumatic brain injury
4= Post-thoracotomy
5= Post-laparotomy
6= Post-amputation
7= Post-neurology surgery
8= Recovered cardiac arrest
9= Metabolic encephalopathy
10= Hypoxic encephalopathy
11= Incomplete hanging
12= Decompensated heart failure
13= Severe obstetric condition
14= Decompensated COPD
15= ARDS
16= BNB-EH
17= BNB-IH
18= BNV
19= Myocarditis
20= Leptospirosis
21= Severe sepsis
22= DMO
23= Septic shock
24= Hypovolemic shock
25= Cardiogenic shock
26= Myocardial infarction
27= Polytrauma
28= Myasthenic crisis
29= Hypertensive emergency
30= Status asthmaticus
31= Status epilepticus
32= Pancreatitis 
33= Fat embolism
34= Stroke
35= Sleep apnea syndrome
36= Digestive bleeding
37= Chronic renal failure
38= Acute renal failure
39= Renal transplant
40= Guillain-Barré
41= AV block
42= Obstetric embolism
43= Aspiration pneumonia
44= Neuroleptic malignant syndrome
45= Diabetic ketoacidosis
46= Meningitis
47= Pulmonary edema
48= Others
"""

# Process data and create dictionaries
num_to_desc = {}
desc_to_num = {}

lines = [line.strip() for line in data.strip().split("\n")]

for line in lines:
    key_part, value = line.split("=", 1)  # Split at first '='
    key = int(key_part.strip())
    value = value.strip()  # Remove surrounding whitespace
    num_to_desc[key] = value
    desc_to_num[value] = key

st.set_page_config(layout="wide")

# Load fixed model
path = r"./Models/"
model = load("new_workflow.joblib")

# Load explanations
with open("Explainers/ig_explainer.pkl", "rb") as archivo:
    ig_exp = pickle.load(archivo)


# Function to get user input
def get_user_input():
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=20, step=1)
    diag_ing1 = desc_to_num[
        st.sidebar.selectbox(
            label="Admission Diagnosis 1", options=list(desc_to_num.keys())
        )
    ]
    diag_ing2 = desc_to_num[
        st.sidebar.selectbox(
            label="Admission Diagnosis 2", options=list(desc_to_num.keys())
        )
    ]
    diag_egr2 = desc_to_num[
        st.sidebar.selectbox(
            label="Discharge Diagnosis 2", options=list(desc_to_num.keys())
        )
    ]
    apache = st.sidebar.number_input(
        "APACHE II", min_value=0, max_value=40, value=18, step=1
    )
    tiempo_vam = st.sidebar.number_input(
        "Ventilator Time", min_value=1, max_value=200, value=5, step=1
    )

    user_data = {
        "Edad": age,
        "Diag.Ing1": diag_ing1,
        "Diag.Ing2": diag_ing2,
        "Diag.Egr2": diag_egr2,
        "APACHE": apache,
        "TiempoVAM": tiempo_vam,
    }
    features = pd.DataFrame(user_data, index=[0])
    return features


feature_names = [
    "Age",
    "Adm.Diag1",
    "Adm.Diag2",
    "Dis.Diag2",
    "APACHE",
    "VentilatorTime",
]


def plot_feature_importances(feature_names, importances):
    """
    Plot feature importance with differentiated colors
    for positive (orange) and negative (blue) contributions.
    """
    importances = np.array(importances).flatten()

    attrib_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

    attrib_df["AbsImportance"] = attrib_df["Importance"].abs()
    attrib_df = attrib_df.sort_values(by="AbsImportance", ascending=False)

    colors = attrib_df["Importance"].apply(lambda x: "orange" if x > 0 else "blue")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(attrib_df["Feature"], attrib_df["Importance"], color=colors)
    ax.set_xlabel("Relevance")
    ax.set_title("Feature Relevance (Colors: Orange=Positive, Blue=Negative)")
    ax.axvline(0, color="red", linestyle="--")
    ax.invert_yaxis()

    return fig


# Streamlit state variables
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.input_df = None

# Get user input
input_df = get_user_input()
input_df_original = input_df.copy()
input_df_original.rename(columns={
    "Edad": "Age",
    "Diag.Ing1": "Adm.Diag1",
    "Diag.Ing2": "Adm.Diag2",
    "Diag.Egr2": "Dis.Diag2",
    "APACHE": "APACHE",
    "TiempoVAM": "VentilatorTime",
}, inplace=True)


# Convert DataFrame to PyTorch tensor
input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
input_tensor = input_tensor.unsqueeze(0)

st.title("Prediction of Non-Survival at ICU Discharge")
st.write("This tool supports prediction of patient non-survival at ICU discharge.")

st.write("#### Patient Characteristics")
st.write(input_df_original)

# Prediction button
predict = st.sidebar.button("Predict")

# Explanation button
explain = st.sidebar.button("Explain")

if predict:
    # Calculate probability prediction
    prob = model.predict_proba(input_df)[:, 1][0]
    st.session_state.prediction = prob

if st.session_state.prediction is not None:
    st.write("### Probability of Non-Survival (Cut 50%)")
    st.write(f"##### {st.session_state.prediction:.2%}")

if explain and (st.session_state.prediction is None):
    st.warning("First, make a prediction.")
elif explain:
    st.write(f"### Explanation")
    attr = ig_exp.attribute(input_tensor, target=0)
    attributions_np = attr.numpy()
    fig = plot_feature_importances(feature_names, attributions_np)
    st.pyplot(fig, use_container_width=True)
