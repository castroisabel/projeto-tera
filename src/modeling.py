import pickle 
from sklearn.ensemble import RandomForestClassifier 

def modeling(X_resampled, y_resampled):
    model = RandomForestClassifier(random_state=42, n_estimators=50)
    model.fit(X_resampled, y_resampled)

    pickle.dump(obj=model, file=open('./model/model.pkl', 'wb'))

    return model

