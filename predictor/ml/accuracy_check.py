import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, scaler, data, train_split=0.8):
    """
    Evaluates the LSTM model performance on unseen test data.

    Args:
        model: Trained LSTM model.
        scaler: MinMaxScaler used for preprocessing.
        data (pd.DataFrame): Original data with 'Close' prices.
        train_split (float): Fraction of data to use for training.

    Returns:
        None. Displays metrics and plots predictions.
    """
    # Scale 'Close' data
    scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))

    # Train-test split
    train_size = int(len(scaled_data) * train_split)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size-60:]  # Include the last 60 steps from training

    # Prepare test data
    x_test, y_test = [], []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        y_test.append(test_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict
    predicted = model.predict(x_test)
    predicted = scaler.inverse_transform(predicted)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)

    print("Model Performance:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(actual, color='blue', label='Actual Prices')
    plt.plot(predicted, color='red', label='Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
