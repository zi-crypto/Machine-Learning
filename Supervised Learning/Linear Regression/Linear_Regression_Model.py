from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use("fivethirtyeight")

def Create_Dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation =='pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64) , np.array(ys, dtype=np.float64)

def Best_Fit_Slope_AND_Intercept(xs, ys):
    m = ( ( (mean(xs) * mean(ys)) - mean(xs * ys) ) / ( (mean(xs) ** 2) - mean(xs ** 2) ) )
    b = mean(ys) - (m * mean(xs))
    return m, b

def Squared_Error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)

def Coefficient_of_Determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    Squared_Error_Regr = Squared_Error(ys_orig, ys_line)
    Squared_Error_y_mean = Squared_Error(ys_orig, y_mean_line)
    return 1 - (Squared_Error_Regr / Squared_Error_y_mean)

xs, ys = Create_Dataset(40, 10, step=1, correlation='neg')
print(xs, ys)

m, b = Best_Fit_Slope_AND_Intercept(xs, ys)

Regression_Line = [(m*x)+b for x in xs]

predict_x = 7
predict_y = (m * predict_x) + b

r_squared = Coefficient_of_Determination(ys, Regression_Line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y)
plt.plot(xs, Regression_Line, color="r")

plt.show()

# PEMDAS Parenthesis, Exponents, Multiplication, Division, Addition, Subtraction

