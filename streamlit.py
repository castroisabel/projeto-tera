import streamlit as st
import pickle
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from src.data_collect import data_collect
from src.data_processing import preprocessing, feature_selection

px.defaults.template = "plotly_white"

@st.cache
def load_data():
    data = data_collect(train=False)
    data = preprocessing(data)  
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time']).dt.normalize()
    data.sort_values(by=['trans_date_trans_time'], ascending=True, inplace=True)
    X_test, y_test = feature_selection(data)

    return data, X_test, y_test

def evaluate_results(y_true, y_pred):
    tp = (y_true == 1) & y_pred
    fp = (y_true == 0) & y_pred
    tn = (y_true == 0) & ~y_pred
    fn = (y_true == 1) & ~y_pred

    return tp, fp, tn, fn


def evaluate_threshold(_df, t):
    (
        _df["Verdadeiro Positivo"],
        _df["Falso Positivo"],
        _df["Verdadeiro Negativo"],
        _df["Falso Negativo"],
    ) = evaluate_results(_df["is_fraud"], _df["predicted"])

    return _df

def main():
    st.set_page_config(layout="wide", page_title="Antifraude",
                        page_icon=":chart_with_upwards_trend:")

    st.sidebar.header("Opções")
    st.sidebar.markdown("""O threshold é o ponto de corte para a decisão 
    entre uma transação normal e uma fraudulenta. A escolha influencia na quantidade de 
    falsos negativos e falsos positivos.""")

    threshold = st.sidebar.slider("Threshold:", 0.0, 1.0, 0.5, 0.01)
    model = pickle.load(open('model/model.pkl', 'rb'))

    data, X_test, y_test = load_data()
    predicted_proba = model.predict_proba(X_test.values)
    predicted = (predicted_proba [:,1] >= threshold).astype('int')


    st.title("Detecção de Fraude em Transações Bancárias")
    st.markdown("Análise e previsão de transações fraudulentas realizadas com cartão de crédito.")


    #Fisrt line
    col1, col2 = st.columns([3,1])
    with col1:
        #Confusion Matrix
        cm = confusion_matrix(y_true=y_test.values, y_pred=predicted)
        classes = ['Não Fraude', 'Fraude']
        fig1 = px.imshow(
            pd.DataFrame(cm, index=classes, columns=classes),
            labels=dict(x='Previsto', y='Verdadeiro'),
            color_continuous_scale='RdYlBu',
            text_auto=True
            )
        fig1.update_layout(autosize=True, font_size=16)
        st.subheader("Matriz de Confusão")
        st.plotly_chart(fig1, use_container_width=False)

    with col2:
        #Summary
        st.subheader("Avaliação")
        summary = pd.DataFrame(precision_recall_fscore_support(y_true=y_test.values, y_pred=predicted, average='binary')).head(3)
        summary.index = ['Precision', 'Recall', 'F1-score']
        summary.columns = ['Métricas']
        st.write(summary.style.format('{:.2%}'))


    #Second line
    col3, col4 = st.columns([3,1])
    with col3:
        #Confusion Matrix
        df = data.copy()
        df['predicted'] = predicted
        df = evaluate_threshold(df, threshold)
        TP = df.query("predicted==1 & is_fraud==1").amt.sum()
        FN = df.query("predicted==0 & is_fraud==1").amt.sum()
        FP = df.query("predicted==1 & is_fraud==0").amt.sum()
        TN = df.query("predicted==0 & is_fraud==0").amt.sum()
        cm = [[TN, FP], [FN, TP]]
        classes = ['Não Fraude', 'Fraude']
        fig2 = px.imshow(
            pd.DataFrame(cm, index=classes, columns=classes),
            labels=dict(x='Previsto', y='Verdadeiro'),
            color_continuous_scale='RdYlBu',
            text_auto=True
            )
        fig2.update_layout(autosize=True, font_size=16)
        st.subheader("Matriz de Confusão Financeira")
        st.plotly_chart(fig2, use_container_width=False)


    errors_sum = df.groupby('trans_date_trans_time').agg(
        {"Falso Positivo": "sum", "Falso Negativo": "sum"}
    )
    fig3 =  px.line(errors_sum, markers=True)
    fig3.update_layout(
        autosize=False,
        font_size=16,
        height=550,
        width=1200,
        xaxis_title="Dia",
        yaxis_title="Contagem",
        plot_bgcolor="rgb(260, 260, 260)"
    )
    st.subheader("Casos diários de Falsos Positivos e Negativos")
    st.plotly_chart(fig3, use_container_width=False, width=100)

    st.markdown("<hr />", unsafe_allow_html=True)


    with st.form("predict_data"):
        st.header("Previsão de transação: ")
        category = st.selectbox("Categoria:",  ['misc_net', 'grocery_pos', 'entertainment', 'gas_transport',
                                                'misc_pos', 'grocery_net', 'shopping_net', 'shopping_pos',
                                                'food_dining', 'personal_care', 'health_fitness', 'travel',
                                                'kids_pets', 'home'])
        amt = st.number_input("Valor:", min_value=1, max_value=500000, step=1, value=250000)
        age = st.slider("Idade", 0, 100, value=50)
        city_pop = st.number_input("População da cidade:", min_value=1000, max_value=10000000, step=1000, value=2000000)
        date = st.date_input("Data:", dt.date.today())
        hour = st.time_input('Horário:', dt.datetime.now())
        merch_lat = st.number_input("Latitude: ", min_value=0, max_value=90, step=5, value=80)
        merch_long = st.number_input("Longitute: ", min_value=0, max_value=180, step=5, value =90)
        st.form_submit_button("Prever!")

        hour = hour.hour
        month = date.month
        day = date.weekday()


    new_data = pd.DataFrame({'category': category, 'amt': float(amt), 'city_pop': int(city_pop), 'merch_lat': merch_lat,
    'merch_long': -merch_long, 'age': age, 'hour': hour, 'day': day, 'month': month, 'is_fraud': 0}, index=[0])

    X_new, _ = feature_selection(new_data)
    X_new = X_new.reindex(columns = X_test.columns, fill_value=0)
    predicted_proba = model.predict_proba(X_new.values)
    
    st.subheader("Probabilidade de fraude: ")
    st.subheader(predicted_proba[:,1][0])

    return None
    

if __name__ == "__main__":
    main()