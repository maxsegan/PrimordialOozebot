import csv
import statistics
import math
import matplotlib
import matplotlib.pyplot as plt

points = []
predictions = []
avgss = []
iteration = []
diversity = []
overfitting = []
equations = []

with open('ed5.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        vals = row[0].split(',')
        iteration.append(int(vals[0]))
        avgss.append(float(vals[1]))
        equations.append(vals[4])
        diversity.append(float(vals[2]))
        overfitting.append(float(vals[3]))
        prediction = [float(vals[5][1:])]
        predictions.append(prediction)
        for val in row[1:]:
            val = val[:-1]
            if val[-1] == ']':
                prediction.append(float(val[:-1]))
            else:
                prediction.append(float(val))

x = []
y = []
index = 0
with open('SR_div_1000.txt', newline='') as tspfile:
    for row in tspfile:
        if len(row) == 0:
            continue
        if index % 2 == 0:
            vals = row.split()
            x.append(float(vals[0]))
            y.append(float(vals[1]))
        index += 1
'''
plt.title('Symbolic Regression Solution')
plt.plot(x, y, 'o', markersize=2, color='blue')
plt.plot(x, predictions[0], 'o', markersize=2, color='red')
plt.savefig('btsp.png', dpi=500)
plt.show()


'''

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

# Comment out above and uncomment this to generate snapshots to make a video
for i in range(len(avgss)):
    plt.title('Symbolic Regression MSE: ' + str(truncate(avgss[i], 8)))
    plt.plot(x, y, 'o', markersize=2, color='blue')
    plt.plot(x, predictions[i], 'o', markersize=2, color='red')
    zeros = ''
    if len(avgss) - i < 10:
        zeros += '0'
    if len(avgss) - i < 100:
        zeros += '0'
    if len(avgss) - i < 1000:
        zeros += '0'
    plt.savefig("plts/plot" + zeros + str(len(avgss) - i) + ".png")
    plt.clf()

