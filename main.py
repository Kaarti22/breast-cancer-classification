from sklearn.model_selection import train_test_split
from src.data_processing import load_data
from src.model import train_model
from src.evaluation import evaluate_model
from src.predict import make_prediction

def main():
    data_frame = load_data()
    
    X = data_frame.drop(columns='label', axis=1)
    Y = data_frame['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    model = train_model(X_train, Y_train)

    training_data_accuracy = evaluate_model(model, X_train, Y_train)
    print(f"Training accuracy: {training_data_accuracy:.2f}")
    
    testing_data_accuracy = evaluate_model(model, X_test, Y_test)
    print(f"Testing accuracy: {testing_data_accuracy:.2f}")

    input_data = (17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189)
    prediction = make_prediction(model, input_data)
    if prediction[0] == 0:
        print("The breast cancer is Malignant")
    else:
        print("The breast cancer is Benign")

if __name__ == "__main__":
    main()
