from sklearn.linear_model import LogisticRegression

def train_model(X_train, Y_train):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model