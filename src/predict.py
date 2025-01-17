import numpy as np

def make_prediction(model, input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    return model.predict(input_data_as_numpy_array)