# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:16:13 2025

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import base64

# Configuración de la página
st.set_page_config(page_title="Predicción de Traspasos", layout="wide")

# Función para crear el botón de descarga del PDF existente
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">Descargar {file_label}</a>'
    return href

# Sidebar con información
with st.sidebar:
    st.title("ℹ️ Ayuda")
    st.markdown("""
    **Esta aplicación predice el riesgo de traspaso de trabajadores usando:**
    - 🚀 **Modelo:** Random Forest Classifier
    - 📊 **Métricas:** AUC Score, Matriz de Confusión
    - 🔍 **Filtros:** Por estado y nivel de interés
    """)
    
    # Botón para descargar el PDF existente
    if os.path.exists("manual.pdf"):
        st.markdown(get_binary_file_downloader_html("manual.pdf", "Manual PDF"), unsafe_allow_html=True)
    else:
        st.warning("Archivo manual.pdf no encontrado")
    
    st.markdown("---")
    st.markdown("Desarrollado por: Javier Horacio Pérez Ricárdez")

# Título principal con el modelo utilizado
st.title("🔍 Sistema Inteligente para Predecir Riesgo de Traspaso (Random Forest)")

# Cargar datos automáticamente si existe el archivo
file_path = "datos_trabajadores_1000.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    uploaded_file = True
    st.success("Archivo 'datos_trabajadores_1000.csv' cargado automáticamente")
else:
    # Cargar datos manualmente si no existe el archivo
    st.subheader("1. Cargar archivo de datos")
    uploaded_file = st.file_uploader("Sube el archivo CSV con los datos de trabajadores:", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Archivo cargado correctamente")

if 'df' in locals():
    st.subheader("2. Visualización de datos")
    with st.expander("🔍 Ver todos los registros", expanded=False):
        st.dataframe(df, height=500)
        st.write(f"📊 Total de registros: {len(df)}")
        st.write("📌 Resumen estadístico:")
        st.dataframe(df.describe())

    st.subheader("3. Filtros")
    col1, col2 = st.columns(2)
    estados = col1.multiselect("Selecciona Estado(s):", options=df["Estado"].unique(), default=df["Estado"].unique())
    interes = col2.multiselect("Selecciona Nivel de Interés:", options=df["Interes"].unique(), default=df["Interes"].unique())

    df_filt = df[df["Estado"].isin(estados) & df["Interes"].isin(interes)]
    st.write(f"Mostrando {len(df_filt)} registros filtrados")

    # Verificar si hay suficientes datos después del filtrado
    if len(df_filt) < 10:
        st.error("⚠️ Muy pocos registros después del filtrado. Amplíe sus criterios de filtro.")
        st.stop()

    # Preprocesamiento
    df_model = df_filt.copy()
    df_model["Dias_Ultimo_Contacto"] = (pd.to_datetime("2025-04-14") - pd.to_datetime(df_model["Fecha_Ultimo_Contacto"])).dt.days
    df_model.drop(columns=["ID_Trabajador", "Fecha_Ultimo_Contacto"], inplace=True)

    # Convertir categóricas
    cat_cols = ["Estado", "Delegacion", "Interes"]
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    # Variable objetivo binaria
    df_model["Traspaso_Alto"] = (df_filt["Prob_Traspaso"] >= 0.5).astype(int)
    y = df_model["Traspaso_Alto"]
    X = df_model.drop(columns=["Prob_Traspaso", "Traspaso_Alto"])

    # Mostrar distribución de clases
    st.write("Distribución de clases:", y.value_counts())

    # Entrenar modelo
    if st.button("🚀 Predecir riesgo de traspaso"):
        # Verificar si hay al menos dos clases
        if len(y.unique()) < 2:
            st.error("Error: Solo hay una clase en los datos. No se puede entrenar el modelo.")
            st.stop()
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Manejo seguro de probabilidades
        if y_proba.shape[1] == 1:
            y_proba = np.column_stack([1-y_pred, y_pred])
            y_proba_single = y_proba[:, 1]
        else:
            y_proba_single = y_proba[:, 1]

        st.subheader("4. Resultados del Modelo (Random Forest)")

        # Matriz de confusión
        st.markdown("**Matriz de Confusión:**")
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 1:
            st.warning("⚠️ Solo se encontró una clase en los datos de prueba")
            class_name = "Traspaso" if unique_classes[0] == 1 else "No Traspaso"
            cm = np.array([[len(y_test) if unique_classes[0] == i else 0 for i in [0, 1]]]).reshape(1, -1)
            cm_df = pd.DataFrame(cm, 
                               index=[class_name],
                               columns=["Pred. No Traspaso", "Pred. Traspaso"])
        else:
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, 
                               index=["No Traspaso", "Traspaso"], 
                               columns=["Pred. No Traspaso", "Pred. Traspaso"])
        
        st.dataframe(cm_df.style.background_gradient(cmap='Blues').format("{:,.0f}"))

        # Métricas de clasificación
        st.markdown("**Métricas de Clasificación:**")
        if len(unique_classes) == 1:
            st.warning("No se pueden calcular métricas completas con una sola clase")
            accuracy = accuracy_score(y_test, y_pred)
            st.metric(label="Accuracy", value=f"{accuracy:.2f}")
        else:
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))
            
            try:
                auc = roc_auc_score(y_test, y_proba_single)
                st.metric(label="AUC Score", value=round(auc, 2))
            except ValueError:
                st.warning("No se pudo calcular el AUC Score")

        # Predecir todo el dataset filtrado
        try:
            df_filt["Riesgo_Traspaso"] = model.predict_proba(X)[:, 1]
        except IndexError:
            df_filt["Riesgo_Traspaso"] = model.predict(X)

        # Mostrar empleados con mayor riesgo
        st.subheader("5. Recomendación de acción")
        top_riesgo = df_filt.sort_values("Riesgo_Traspaso", ascending=False).head(10)
        st.write("🔴 Estos trabajadores presentan mayor riesgo de traspaso:")
        st.dataframe(top_riesgo[["ID_Trabajador", "Estado", "Delegacion", "Edad", "Ingreso_Mensual", "Visitas_Promotor", "Riesgo_Traspaso"]])

        # Mapa
        st.subheader("6. Visualización por Estado")
        mapa = df_filt.groupby("Estado")["Riesgo_Traspaso"].mean().reset_index()
        fig = px.bar(mapa, x="Estado", y="Riesgo_Traspaso", title="Promedio de riesgo por Estado", 
                    color="Riesgo_Traspaso", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Por favor, sube un archivo CSV para comenzar o coloca el archivo 'datos_trabajadores_1000.csv' en el mismo directorio.")