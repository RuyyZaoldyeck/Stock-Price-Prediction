from processing import PreProcessing
import pandas as pd
from sklearn.linear_model import LinearRegression

csv_data_df = pd.read_csv('BTC-USD.csv')

col_to_predict = 'Close'
test_size = 0.2
how_far_prediction = 5

def main() -> dict:
  """ Trains and display a dictionary with a linear regression score and the how_far_predictions values """

  X_train, X_test, y_train, y_test, X_predict = PreProcessing.formatting_data(csv_data_df, col_to_predict, how_far_prediction, test_size)

  # Using the linear regression model
  model = LinearRegression()

  # Training
  model.fit(X_train, y_train)

  # Scoring, and predeicting
  score = model.score(X_test, y_test)
  forecast = model.predict(X_predict)

  print({'linear_regression_score': score, 'forecast': forecast})

if __name__ == '__main__':
  main()