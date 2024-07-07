import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def exponential_func(x, a, b):
    return a * np.exp(b * x)


def gaussian_func(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


def analyse(data, exponential=True, gaussian=True):
    print(f"Analysing data: {data}")
    
    x_data = np.array([point[0] for point in data])
    y_data = np.array([point[1] for point in data])
    
    # Extend the x-axis by twice the maximum value for future prediction
    x_extended = np.linspace(np.min(x_data), 2 * np.max(x_data), 400)  # Increase the number of points for a smoother curve
    
    a_exp = b_exp = a_gauss = b_gauss = c_gauss = y_fit_exp = y_fit_gauss = 0
    
    if exponential:
        try:
            popt_exp, _ = curve_fit(exponential_func, x_data, y_data, p0=(1, 0.1))
        except:
            print("Exponential fit failed.")
            popt_exp = [1, 0.1]
        a_exp, b_exp = popt_exp
        y_fit_exp = exponential_func(x_extended, *popt_exp)  # Use extended x range for prediction
        
    
    if gaussian:
        try:
            popt_gauss, _ = curve_fit(gaussian_func, x_data, y_data, p0=(1, np.mean(x_data), np.std(x_data)))
        except:
            print("Gaussian fit failed.")
            popt_gauss = [1, np.mean(x_data), np.std(x_data)]
        a_gauss, b_gauss, c_gauss = popt_gauss
        y_fit_gauss = gaussian_func(x_extended, *popt_gauss)  # Use extended x range for prediction
    
    plt.figure(figsize=((12 if exponential and gaussian else 6), 6))
    
    if exponential:
        plt.subplot(1, 2 if gaussian else 1, 1)
        plt.scatter(x_data, y_data, label='Data')
        plt.plot(x_extended, y_fit_exp, label=f'Exponential Fit: a={a_exp:.2f}, b={b_exp:.2f}', color='red')
        plt.title('Exponential Fit')
        plt.legend()
    
    if gaussian:
        plt.subplot(1, 2 if exponential else 1, 2 if exponential else 1)
        plt.scatter(x_data, y_data, label='Data')
        plt.plot(x_extended, y_fit_gauss, label=f'Gaussian Fit: a={a_gauss:.2f}, b={b_gauss:.2f}, c={c_gauss:.2f}',
                 color='green')
        plt.title('Gaussian Fit')
        plt.legend()
    
    plt.show()
    
    print(f"Exponential Fit Parameters: a={a_exp:.2f}, b={b_exp:.2f}")
    print(f"Gaussian Fit Parameters: a={a_gauss:.2f}, b={b_gauss:.2f}, c={c_gauss:.2f}")


def main():
    # file path
    file_path = "taskB_3_1_new.csv"
    
    # read data - each line is a tuple of x and y values
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = line.strip().split(',')
            data.append((int(x), int(y)))
    
    # split data into subintervals 1-15, 16-30, 31-45, 46-51
    # intervals = [(1, 15), (16, 30), (31, 45), (46, 51)]
    intervals = [(1, 10000)]
    for _ in intervals:
        start, end = _
        sub_data = [point for point in data if start <= point[0] <= end]
        # analyse each subinterval
        analyse(sub_data, exponential=False, gaussian=True)


main()