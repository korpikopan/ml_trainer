import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="ML Trainer v1", page_icon=":tada:", layout="wide")

def create_page(content, page_title=""):
    st.title(page_title)
    st.write(content)

# 1. as sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation Menu",
        options=["ML Trainer", "Contacts"],
        icons=["book", "envelope"],
        default_index=0,
        styles={
                "nav-link-selected": {"background-color": "red"},
            },
    )

with st.container():
    if selected == "ML Trainer":
        content = """This app is just a demo to familiarize yourself with basic 
        LinearRegression Model and simple ML regression predictor
        """
        create_page(content, page_title=selected)
        st.write("---")
        
        uploaded_file = st.file_uploader("1. Importe le fichier xlsx/csv (sous forme de tableau)", type=["csv", "xlsx"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            st.write("---")

            # selection de la target et de la data
            col_num = df.select_dtypes(include='number').columns
            target_select = st.selectbox("selectionne la colonne a  predire", options=col_num)
            remaining_cols = df.columns.drop(target_select)
            remaining_cols = [col for col in remaining_cols]
            data_select = st.multiselect("selectionne les colonnes a entrainé", remaining_cols, remaining_cols)

            target = df[target_select]
            data = df[data_select]

            st.write("---")
            sample_train_test_size = st.slider("echantillonage train/test", 0, 100, 80)

            xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=sample_train_test_size/100.0)
            st.write("##")

            train_button = st.button("Entrainer le modele")

            if train_button:
                lr = LinearRegression()
                lr.fit(xtrain, ytrain)

                score = lr.score(xtest, ytest)

                st.write("Score accuracy = {}".format(score))



    if selected == "Contacts":
        content = "vous pouvez me contacter sur [Learn more >](https://www.twitter.com/__korpikopan__)"
        create_page(content, page_title=selected)

