from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, Y_test):
    predictions = model.predict(X_test)
    return accuracy_score(Y_test, predictions)