##############################FUNCTIONS####################
def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1
    ###########################################


def occurrences(string, sub):
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count


############
def absSecond(val):
    return abs(int(val[1]) - int(val[2]))


#############
def fiveSecond(val):
    return float(val[5])


## /// Function of Weights
def binomial(val):
    x = abs(int(val))
    b = 0.03125 * (x * x) + (1.25 * x)
    return b

# /// Test and Calculate Binomial Functions of Weights
from numpy import arange
from scipy.optimize import curve_fit
from matplotlib import pyplot


# // define the true objective function
def objective(x, a, b, c):
    return a * x + b * x ** 2 + c


# load the dataset
x = [26, 23, 15, 7, 0]
y = [100, 98, 90, 60, 0]
# curve fit
popt, _ = curve_fit(objective, x, y)

# // summarize the parameter values
a, b, c = popt
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))

# // plot input vs output
pyplot.scatter(x, y)

# // define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)

# // calculate the output for the range
y_line = objective(x_line, a, b, c)

# // create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()

