import streamlit as st
import pickle
import plotly.express as px
import plotly.graph_objects as go

px.defaults.template = "plotly_white"

def main():
    st.set_page_config(layout="wide", page_title="Antifraude",
                        page_icon=":chart_with_upwards_trend:")

    st.sidebar.header("Opções")
    st.sidebar.markdown("""O threshold é o ponto de corte para a decisão 
    entre uma transação normal e uma fraudulenta. A escolha influencia na quantidade de 
    falsos negativos e falsos positivos.""")

    threshold = st.sidebar.slider("Threshold:", 0.0, 1.0, 0.53, 0.01)

    st.title("Detecção de Fraude em Transações Bancárias")
    st.markdown("Análise e previsão de transações fraudulentas realizadas com cartão de crédito.")

    model = pickle.load(open('model/model.pkl', 'rb'))

    

    return None

if __name__ == "__main__":
    main()