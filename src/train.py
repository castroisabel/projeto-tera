from data_collect import data_collect
from data_processing import preprocessing, feature_selection, data_balancing
from modeling import modeling
from evaluation import find_threshold, evaluate


if __name__ == "__main__":
    # Coleta de dados
    train_data = data_collect(train=True)
    test_data = data_collect(train=False)

    # Pré-processamento 
    # Treino
    train_data = preprocessing(train_data)
    X_train, y_train = feature_selection(train_data)
    X_resampled, y_resampled = data_balancing(X_train, y_train)
    # Teste
    test_data = preprocessing(test_data)  
    X_test, y_test = feature_selection(test_data) 

    # Modelagem
    model = modeling(X_resampled, y_resampled)

    # Avaliação
    threshold = find_threshold(model, X_resampled, y_resampled)
    evaluate(model, threshold, X_test, y_test)


