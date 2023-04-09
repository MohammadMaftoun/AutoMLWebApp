import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import *

with st.sidebar:
    st.image("https://www.exasol.com/app/uploads/2022/02/AutoML-Exasol.png")
    st.title("AutoML_Project")
    choice = st.radio("Navigation",["Upload", "Visualization" ,"AutoML"])
    st.info("This web application can help us")

if os.path.exists("source.csv"):
    df = pd.read_csv("source.csv", index_col =None)

if choice == "Upload":
    st.title("Upload Your DataSet")
    file = st.file_uploader("Upload your dataset")
    if file:
        data_frame = pd.read_csv(file, index_col=None)
        data_frame.to_csv("source.csv", index=None)
        st.dataframe(data_frame)
if choice == "Visualization":
    st.title("EDA of Your DataSet")
    vis_report = df.profile_report()
    st_profile_report (vis_report)
    
if choice == "AutoML":
    st.title("AutoML of Your DataSet")
    #target = st.selectbox("Select Target Plz!",df.columns)
    setup(df, target = 'Outcome')
    best_model = compare_models()
    evaluate_model(best_model)
    compare_df = pull()
    st.success("This is Machine learning Model")
    st.dataframe(compare_df)
    best_model
    save_model(best_model, 'best_model')
