# Import necessary libraries
# Import pandas for data manipulation and analysis
import pandas as pd
# Import numpy for numerical operations
import numpy as np
# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import ARIMA from statsmodels for time series forecasting
from statsmodels.tsa.arima.model import ARIMA
# Import mean_squared_error from sklearn for model evaluation
from sklearn.metrics import mean_squared_error

# Generate a sample time series dataset
def generate_sample_data():
    # Create a date range
    date_range = pd.date_range(start='2020-01-01', periods=100, freq='D')
    # Generate random data
    data = np.random.rand(100) * 100
    # Create a DataFrame
    df = pd.DataFrame(data, index=date_range, columns=['value'])
    # Return the DataFrame
    return df

# Function to plot the time series data
def plot_time_series(data, title='Time Series Data'):
    # Set the plot size
    plt.figure(figsize=(10, 6))
    # Plot the data
    plt.plot(data, label='Original Data')
    # Set the plot title
    plt.title(title)
    # Set the x-axis label
    plt.xlabel('Date')
    # Set the y-axis label
    plt.ylabel('Value')
    # Display the legend
    plt.legend()
    # Show the plot
    plt.show()

# Function to split the dataset into training and testing sets
def train_test_split(data, split_ratio=0.8):
    # Calculate the split point
    split_point = int(len(data) * split_ratio)
    # Split the data into training and testing sets
    train, test = data[:split_point], data[split_point:]
    # Return the training and testing sets
    return train, test

# Function to fit ARIMA model and make predictions
def arima_forecast(train, test, order=(5,1,0)):
    # Initialize the history with the training data
    history = [x for x in train]
    # Initialize the list of predictions
    predictions = []
    # Iterate over the testing data
    for t in range(len(test)):
        # Fit the ARIMA model
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        # Make a one-step forecast
        output = model_fit.forecast()
        # Get the forecast value
        yhat = output[0]
        # Append the forecast value to the predictions list
        predictions.append(yhat)
        # Append the actual value to the history
        history.append(test[t])
    # Return the list of predictions
    return predictions

# Function to evaluate the model
def evaluate_forecast(test, predictions):
    # Calculate the mean squared error
    error = mean_squared_error(test, predictions)
    # Print the mean squared error
    print(f'Test Mean Squared Error: {error:.3f}')
    # Set the plot size
    plt.figure(figsize=(10, 6))
    # Plot the actual data
    plt.plot(test.index, test, label='Actual Data')
    # Plot the predicted data
    plt.plot(test.index, predictions, color='red', label='Predicted Data')
    # Set the plot title
    plt.title('Actual vs Predicted')
    # Set the x-axis label
    plt.xlabel('Date')
    # Set the y-axis label
    plt.ylabel('Value')
    # Display the legend
    plt.legend()
    # Show the plot
    plt.show()

# Main function
def main():
    # Generate sample data
    df = generate_sample_data()
    # Plot the original time series data
    plot_time_series(df, title='Original Time Series Data')
    # Split the data into training and testing sets
    train, test = train_test_split(df['value'])
    # Fit ARIMA model and make predictions
    predictions = arima_forecast(train, test)
    # Evaluate the forecast
    evaluate_forecast(test, predictions)

# Run the main function
if __name__ == '__main__':
    main()
