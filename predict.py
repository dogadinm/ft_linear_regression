def load_parameters(file_path):
    with open(file_path, 'r') as f:
        theta0, theta1 = map(float, f.read().split(','))
    return theta0, theta1

def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

def main():
    theta0, theta1 = load_parameters('parameters.txt')
    
    mileage = float(input('Enter the mileage of the car: '))
    estimated_price = estimate_price(mileage, theta0, theta1)
    
    print(f'The estimated price for a car with {mileage} mileage is: ${estimated_price:.2f}')

if __name__ == '__main__':
    main()
