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

set_visualize_provider(InlineProvider())
from interpret import show
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from captum.attr import Saliency
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

# Cargar el modelo fijo
path = r"./Modelos/"
model_file_name = "modelo_mlp.pkl"
with open(fr"{path}{model_file_name}", 'rb') as file:
    model = pickle.load(file)

# Cargar las explicaciones
with open("Explainers/lime_explainer_iml.pkl", "rb") as archivo:
    lime_grafica = pickle.load(archivo)
with open("Explainers/shap_explainer_iml.pkl", "rb") as archivo:
    shap_grafica = pickle.load(archivo)


try:
    with open("Explainers/lime_explainer.pkl", "rb") as archivo:
        lime_exp = pickle.load(archivo)
except Exception as e:
    st.error(f"Error al cargar el explicador LIME: {e}")

with open("Explainers/shap_explainer.pkl", "rb") as archivo:
    shap_exp = pickle.load(archivo)
with open("Explainers/ig_explainer.pkl", "rb") as archivo:
    ig_exp = pickle.load(archivo)
with open("Explainers/saliency_explainer.pkl", "rb") as archivo:
    saliency_exp = pickle.load(archivo)


# Función para obtener los valores de entrada del usuario
def get_user_input():
    age = st.sidebar.number_input("Edad", min_value=0, max_value=120, value=20, step=1)
    diag_ing1 = st.sidebar.number_input("Diag.Ing1", min_value=0, max_value=20, value=0, step=1)
    diag_ing2 = st.sidebar.number_input("Diag.Ing2", min_value=0, max_value=30, value=0, step=1)
    diag_egr2 = st.sidebar.number_input("Diag.Egr2", min_value=0, max_value=20, value=0, step=1)
    apache = st.sidebar.number_input("APACHE II", min_value=0, max_value=40, value=18, step=1)
    tiempo_vam = st.sidebar.number_input("TiempoVAM", min_value=0, max_value=700, value=30, step=1)

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

feature_names = ['Edad', 'Diag.Ing1', 'Diag.Ing2', 'Diag.Egr2', 'APACHE', 'TiempoVAM']


def plot_feature_importances(feature_names, importances):
    """
    Graficar la importancia de las características con colores diferenciados
    para contribuciones positivas (naranja) y negativas (azul).
    """
    importances = np.array(importances).flatten()

    attrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    attrib_df['AbsImportance'] = attrib_df['Importance'].abs()
    attrib_df = attrib_df.sort_values(by='AbsImportance', ascending=False)

    colors = attrib_df['Importance'].apply(lambda x: 'orange' if x > 0 else 'blue')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(attrib_df['Feature'], attrib_df['Importance'], color=colors)
    ax.set_xlabel('Relevancia')
    ax.set_title('Relevancia de las Características (Colores: Naranja=Positiva, Azul=Negativa)')
    ax.axvline(0, color='red', linestyle='--')
    ax.invert_yaxis()

    return fig


# Funcion para preprocesar la salida de lime
def procesar_salida_lime(salida_lime, num_caracteristicas=6):
    # Crear dos arreglos de ceros del tamaño del número de características
    relevancias_clase_0 = [0] * num_caracteristicas
    relevancias_clase_1 = [0] * num_caracteristicas

    # Recorrer las relevancias de ambas clases simultáneamente
    for (indice_0, relevancia_0), (indice_1, relevancia_1) in zip(salida_lime[0], salida_lime[1]):
        relevancias_clase_0[indice_0] = relevancia_0
        relevancias_clase_1[indice_1] = relevancia_1

    # Devolver las relevancias para ambas clases
    return [relevancias_clase_0, relevancias_clase_1]


# Variables de estado de Streamlit
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.input_df = None
    st.session_state.explanation_method = None

# Obtener los valores de entrada del usuario
input_df = get_user_input()
input_df_original = input_df.copy()

# Convierte el DataFrame a un tensor de PyTorch
input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
# Añade una dimensión extra al tensor
input_tensor = input_tensor.unsqueeze(0)

st.title("Predicción de no supervivencia al egreso de UCI")
st.write("Esta herramienta es de apoyo en la predicción de no supervivencia de pacientes al egreso de UCI.")

st.write("#### Carcacterísticas del paciente")
st.write(input_df_original)

# Botón para realizar la predicción
predecir = st.sidebar.button("Predecir")

# Selección del método de explicabilidad
explanation_method = st.sidebar.selectbox(
    "Método de Explicabilidad",
    ["LIME", "SHAP", "Integrated Gradients", "Saliency Maps"],
    index=0
)
# Botón para generar la explicación
explicar = st.sidebar.button("Explicar")





if predecir:
    # Realizar la predicción de probabilidad
    prob = model.predict_proba(input_df)[:, 1][0]
    st.session_state.prediction = prob  # Guardar la predicción en el estado

if st.session_state.prediction is not None:
    st.write("### Probabilidad de No Supervivencia")
    st.write(f"##### {st.session_state.prediction:.2%}")


if explicar and (st.session_state.prediction is None):

    st.warning("Primero, realice una predicción.")

elif explicar:

    st.write(f"### Explicación de {explanation_method}")

    if explanation_method == "LIME":

        print("Metodo de explicabilidad LIME")


        def predict_fn(data):
            return model.predict_proba(data)


        @st.cache_resource
        def lime_explain(data_row, _predict_fn, top_labels=2 ):
            explanation = lime_exp.explain_instance(data_row,_predict_fn,top_labels)
            return explanation


        try:
            # # Tu código aquí
            # st.write("Antes del if")
            print("Intentando calcular la explicación con LIME...")
            explanation = lime_explain(data_row=input_df_original.iloc[0].values, _predict_fn=predict_fn , top_labels=2)
            print("Explicación generada exitosamente.")

        except Exception as e:
            st.write(f"Error encontrado: {e}")


        # with st.spinner('Generando explicación...'):
        #     explanation = lime_exp.explain_instance(
        #         data_row=input_df.iloc[0].values,
        #         predict_fn=model.predict_proba,
        #         top_labels=2
        #     )
        # #     st.write("#### ??")
        # st.success("Done!")
        # st.write(explanation.as_map())
        st.write("Después de la condición if")

        # exp = explanation.as_map()
        # resultado = procesar_salida_lime(exp)
        # attr = resultado[1]
        # attributions_np = attr.array()
        # fig = plot_feature_importances(feature_names, attributions_np)
        # st.pyplot(fig, use_container_width=True)

        # html_content = show(lime_grafica.explain_local(input_df.iloc[0].values), 0)
        #
        # # Mostrar HTML en Streamlit
        # st.markdown(html_content, unsafe_allow_html=True)


    elif explanation_method == "SHAP":
        # Generar explicación de SHAP
        # show(shap_grafica.explain_local(input_df), 0)
        st.write("### Explicación de Mapas de Silencia")

    elif explanation_method == "Integrated Gradients":
        attr = ig_exp.attribute(input_tensor, target=0)

        attributions_np = attr.numpy()
        fig = plot_feature_importances(feature_names, attributions_np)
        st.pyplot(fig, use_container_width=True)

    elif explanation_method == "Saliency Maps":
        attr = saliency_exp.attribute(input_tensor, target=0, abs=False)

        attributions_np = attr.numpy()
        fig = plot_feature_importances(feature_names, attributions_np)
        st.pyplot(fig, use_container_width=True)

