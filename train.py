import pandas as pd
import numpy as np
import sys

def normalize(mileages, prices):
    mileage_min, mileage_max = min(mileages), max(mileages)
    price_min, price_max = min(prices), max(prices)

    mileage_range = mileage_max - mileage_min
    price_range = price_max - price_min
    EPSILON = 1e-10  # Защита от деления на очень маленькие числа
    if abs(mileage_range) < EPSILON or abs(price_range) < EPSILON:
        print("Error: The data range is too small to normalize")
        sys.exit(1)

    mileages_n = [(m - mileage_min) / mileage_range for m in mileages]
    prices_n = [(p - price_min) / price_range for p in prices]
    
    return mileages_n, prices_n, mileage_min, mileage_range, price_min, price_range

def denormalize(theta0_n, theta1_n, mileage_min, mileage_range, price_min, price_range):
    theta0 = price_min + price_range * theta0_n - (price_range * theta1_n * mileage_min) / mileage_range
    theta1 = (price_range / mileage_range) * theta1_n
    return theta0, theta1

def gradient_descent(x, y, learning_rate, iterations):
    theta0, theta1 = 0.0, 0.0
    m = len(x)

    x = np.array(x)
    y = np.array(y)

    for _ in range(iterations):
        predictions = theta0 + theta1 * x
        errors = predictions - y
        theta0 -= learning_rate * np.sum(errors) / m
        theta1 -= learning_rate * np.sum(errors * x) / m

    return theta0, theta1

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data['km'].values, data['price'].values

def save_parameters(theta0, theta1, filepath):
    with open(filepath, 'w') as file:
        file.write(f"{theta0},{theta1}")

def main():
    data_file = 'data.csv'
    param_file = 'parameters.txt'
    learning_rate = 0.1
    iterations = 2000

    mileages, prices = load_data(data_file)
    mileages_n, prices_n, mileage_min, mileage_range, price_min, price_range = normalize(mileages, prices)
    theta0_n, theta1_n = gradient_descent(mileages_n, prices_n, learning_rate, iterations)
    theta0, theta1 = denormalize(theta0_n, theta1_n, mileage_min, mileage_range, price_min, price_range)
    save_parameters(theta0, theta1, param_file)

if __name__ == '__main__':
    main()
