import pandas as pd
import numpy as np
import sys

def normalize(mileages, prices):
    mileage_min = min(mileages)
    mileage_max = max(mileages)
    mileage_range = mileage_max - mileage_min
    price_min = min(prices)
    price_max = max(prices)
    price_range = price_max - price_min

    EPSILON = 1e-10  # Small value to prevent division by very small numbers
    if abs(mileage_range) < EPSILON or abs(price_range) < EPSILON:
        print("Error: Data range too small for normalization")
        sys.exit(1)

    mileages_n = [(m - mileage_min) / mileage_range for m in mileages]
    prices_n = [(p - price_min) / price_range for p in prices]
    return mileages_n, prices_n

def denormalize(mileages, prices, theta0_n, theta1_n):
    mileage_min = min(mileages)
    mileage_max = max(mileages)
    mileage_range = mileage_max - mileage_min
    price_min = min(prices)
    price_max = max(prices)
    price_range = price_max - price_min

    theta0 = price_min + price_range * theta0_n - (price_range * theta1_n * mileage_min) / mileage_range
    theta1 = (price_range / mileage_range) * theta1_n
    return theta0, theta1

def gradient_descent(x, y, learning_rate, num_iterations):
    theta0_n = 0
    theta1_n = 0

    m = len(x)

    x_numpy = np.array(x)
    y_numpy = np.array(y)

    for _ in range(num_iterations):
        estimated_y = theta0_n + theta1_n * x_numpy
        errors = estimated_y - y_numpy 
        theta0_n_gradient = np.sum(errors) / m
        theta1_n_gradient = np.sum(errors * x_numpy) / m
        theta0_n -= learning_rate * theta0_n_gradient
        theta1_n -= learning_rate * theta1_n_gradient

    return theta0_n, theta1_n
    
def load_data(file_path):
    data = pd.read_csv(file_path)
    mileage = data['km'].values
    price = data['price'].values
    return mileage, price


def save_parameters(theta0, theta1, file_path):
    with open(file_path, 'w') as f:
        f.write(f'{theta0},{theta1}')

def main():
    file_path = 'data.csv'  # The dataset file path
    learning_rate = 0.1  # Reduce learning rate significantly
    num_iterations = 2000      # Number of iterations for training

    mileages, prices = load_data(file_path)
    mileages_n, prices_n = normalize(mileages, prices)
    theta0_n, theta1_n = gradient_descent(mileages_n, prices_n, learning_rate, num_iterations)
    
    theta0, theta1 = denormalize(mileages, prices, theta0_n, theta1_n)
    save_parameters(theta0, theta1, 'parameters.txt')

if __name__ == '__main__':
    main()
