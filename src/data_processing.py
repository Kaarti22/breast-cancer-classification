import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():
    dataset = load_breast_cancer()
    data_frame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    data_frame['label'] = dataset.target
    return data_frame