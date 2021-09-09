import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class PreProcessing:

  def formatting_data(dataframe, col_to_predict : str, how_far_prediction : int, test_size: float) -> list:
    """ creating feature column, processing, removing na´s, splitting training and testing set """

    dataframe = dataframe.fillna({'Close':dataframe.Close.mean()})

    # Adding an offset to the columns by "how_far_prediction" and adding to the set
    offset = dataframe[col_to_predict].shift((-1) * how_far_prediction)
    offset.dropna(inplace = True)

    # Feature data scaling
    X = np.array(dataframe[[col_to_predict]])
    X = preprocessing.scale(X) 

    # Using the offsetted input to use in the prediction
    X_used_to_predict = X[(-1) * how_far_prediction : ]

    # Using the unoffsetted to save data for training
    X = X[: (-1) * how_far_prediction]

    # Defining numpy object of the target column
    y = np.array(offset)

    # Using sklearn splitting method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)

    return [X_train, X_test, y_train, y_test, X_used_to_predict]