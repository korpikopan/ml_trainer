import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

st.set_page_config(page_title="ML Trainer v1", page_icon=":tada:", layout="wide")

MODEL = {
    'LinearRegression': LinearRegression, 
    'LogisticRegression': LogisticRegression,
    'Ridge': Ridge
}


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

    st.write("---")
    ml_model = st.radio("ML Model", ("LinearRegression", "LogisticRegression", "Ridge"), index=0)
    if ml_model == 'Ridge':
        model = MODEL.get(ml_model)(alpha=1.0)
    else:
        model = MODEL.get(ml_model)()

with st.container():
    if selected == "ML Trainer":
        content = """This app is just a demo to familiarize yourself with basic 
        Linear Models (LinearRegression, Ridge, LogisticRegression)
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
            target_select = st.selectbox("selectionne la colonne à prédire", options=col_num)
            remaining_cols = df.columns.drop(target_select)
            remaining_cols = [col for col in remaining_cols]
            data_select = st.multiselect("selectionne les colonnes à entrainer", remaining_cols, remaining_cols)

            target = df[target_select]
            data = df[data_select]

            st.write("---")
            sample_train_test_size = st.slider("choisissez la taille de l'échantillon d'apprentissage ( recommendation à 70%)", 0, 100, 70)

            xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=sample_train_test_size/100.0)
            st.write("##")

            train_button = st.button("Entrainer le modèle")

            if train_button:
                model.fit(xtrain, ytrain)

                score = model.score(xtest, ytest)
                if score >= 0.75:
                    st.write("Modele de prediction correct! Score de précision à {}".format(score*100.0))
                
                else:
                    st.write("Modele de prediction non performant! Score de précision à {}".format(score*100.0))



    if selected == "Contacts":
        content = "!!En Construction!!"
        create_page(content, page_title=selected)

