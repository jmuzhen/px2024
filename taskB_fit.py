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
    x_extended = np.linspace(np.min(x_data), np.max(x_data), 400)  # Increase the number of points for a smoother curve
    y_data = np.array([point[1] for point in data])
    
    a_exp = b_exp = a_gauss = b_gauss = c_gauss = y_fit_exp = y_fit_gauss = 0
    
    if exponential:
        # Fit the data to the exponential function
        try:
            popt_exp, _ = curve_fit(exponential_func, x_data, y_data, p0=(1, 0.1))
        except:
            print("Exponential fit failed.")
            # we want to continue with the Gaussian fit even if the exponential fit fails
            popt_exp = [1, 0.1]
        a_exp, b_exp = popt_exp
        y_fit_exp = exponential_func(x_data, *popt_exp)
        y_fit_exp_extended = exponential_func(x_extended, *popt_exp)
    
    if gaussian:
        # Fit the data to the Gaussian function
        try:
            popt_gauss, _ = curve_fit(gaussian_func, x_data, y_data, p0=(1, np.mean(x_data), np.std(x_data)))
        except:
            print("Gaussian fit failed.")
            popt_gauss = [1, np.mean(x_data), np.std(x_data)]
        a_gauss, b_gauss, c_gauss = popt_gauss
        y_fit_gauss = gaussian_func(x_data, *popt_gauss)
        y_fit_gauss_extended = gaussian_func(x_extended, *popt_gauss)
    
    # Plot the data and the fits
    plt.figure(figsize=((12 if exponential and gaussian else 6), 6))
    
    if exponential:
        plt.subplot(1, 2 if gaussian else 1, 1)
        plt.scatter(x_data, y_data, label='Data')
        plt.plot(x_extended, y_fit_exp_extended, label=f'Exponential Fit: a={a_exp:.2f}, b={b_exp:.2f}', color='red')
        plt.title('Exponential Fit')
        plt.legend()
    
    if gaussian:
        plt.subplot(1, 2 if exponential else 1, 2 if exponential else 1)
        plt.scatter(x_data, y_data, label='Data')
        plt.plot(x_extended, y_fit_gauss_extended, label=f'Gaussian Fit: a={a_gauss:.2f}, b={b_gauss:.2f}, c={c_gauss:.2f}',
                 color='green')
        plt.title('Gaussian Fit')
        plt.legend()
    
    plt.show()
    
    # Calculate the sum of squared residuals (SSR) for both fits
    ssr_exp = np.sum((y_data - y_fit_exp) ** 2)
    ssr_gauss = np.sum((y_data - y_fit_gauss) ** 2)
    
    print(f"Exponential Fit Parameters: a={a_exp:.2f}, b={b_exp:.2f}")
    print(f"Gaussian Fit Parameters: a={a_gauss:.2f}, b={b_gauss:.2f}, c={c_gauss:.2f}")
    
    print(f"SSR for Exponential Fit: {ssr_exp:.2f}")
    print(f"SSR for Gaussian Fit: {ssr_gauss:.2f}")
    
    # Determine which fit is better
    if ssr_exp < ssr_gauss:
        print("The exponential function provides a better fit.")
    else:
        print("The Gaussian function provides a better fit.")


def main():
    # file path
    file_path = "taskB_data.csv"
    
    # read data - each line is a tuple of x and y values
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = line.strip().split(',')
            data.append((int(x), int(y)))
    
    # split data into subintervals
    intervals = [(1, 15), (16, 30), (31, 45), (46, 51)]
    # intervals = [(1, 10000)]
    for _ in intervals:
        start, end = _
        sub_data = [point for point in data if start <= point[0] <= end]
        # analyse each subinterval
        analyse(sub_data, exponential=True, gaussian=True)


main()