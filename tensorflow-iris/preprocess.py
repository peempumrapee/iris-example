from sklearn.preprocessing import StandardScaler, LabelEncoder


def data_preprocessing(raw_data):
    data = raw_data[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
    target = raw_data[["class"]]

    # Data preprocessing
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Target preprocessing
    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    return data, target, scaler, encoder
