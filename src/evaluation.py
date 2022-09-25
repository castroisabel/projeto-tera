import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def argmin(d):
    if not d: return None
    min_val = min(d.values())
    return [k for k in d if d[k] == min_val][0]


def find_threshold(model, X_resampled, y_resampled):
    predicted_proba = model.predict_proba(X_resampled)

    threshold_search = {}
    threshold = np.linspace(0.3, 0.8, num=51).round(2)

    for t in threshold:
        predicted = (predicted_proba [:,1] >= t).astype('int')
        cm = confusion_matrix(y_true=y_resampled, y_pred=predicted)
        sum = cm[0][1] + cm[1][0]
        threshold_search[t] = []
        threshold_search[t].append(sum)

    return argmin(threshold_search)


def evaluate(model, threshold, X_test, y_test):
    predicted_proba = model.predict_proba(X_test)
    predicted = (predicted_proba [:,1] >= threshold).astype('int')
    print('Classification report:\n', classification_report(y_test, predicted))

    cm = confusion_matrix(y_true=y_test, y_pred=predicted)
    print('Confusion matrix:\n', cm)
    