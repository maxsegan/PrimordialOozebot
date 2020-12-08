import matplotlib.pyplot as plt
import csv
import statistics
import math

plt.title('Learning Curve')
plt.ylabel('Speed')
plt.xlabel('Iteration Number')

evo = []
emin = []
emax = []
evobars = []

numRuns = 5
numIterations = 20700
sqrtRuns = math.sqrt(numRuns)
iterationDataEvo = []

indicesToPlot = range(1, numIterations)

#xtiks = []
#for i in range(10):
#    xtiks.append(int(numIterations / 5 * i))
#plt.xticks(xtiks)

for i in range(1, numRuns + 1):
    iterationDataEvo.append({})
    with open('evo' + str(i) + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in list(reader):
            vals = row[0].split(',')
            iteration = int(vals[0])
            val = float(row[1]) / 5.0
            iterationDataEvo[-1][iteration] = val

print("Done reading data")

unifiedEvo = []

index = 0
for iteration in indicesToPlot:
    currentEvo = []
    unifiedEvo.append(currentEvo)
    for run in range(numRuns):
        valEvo = -1
        if iteration in iterationDataEvo[run]:
            valEvo = iterationDataEvo[run][iteration]
        else:
            #unchanged
            valEvo = unifiedEvo[-2][run]
        currentEvo.append(valEvo)

    evoAverage = statistics.mean(currentEvo)
    evoError = statistics.stdev(currentEvo) / sqrtRuns

    evo.append(evoAverage)
    evobars.append(evoError)

for i in range(len(evo)):
    emin.append(evo[i] - evobars[i])
    emax.append(evo[i] + evobars[i])

print("Done processing data")

#plt.xscale('log')
#plt.yscale('log')
plt.plot(indicesToPlot, evo, color='blue', linewidth=1)
plt.fill_between(indicesToPlot, emin, emax, facecolor='blue', lw=0, alpha=0.5)

plt.savefig('learningcurve.png', dpi=500)
plt.show()
