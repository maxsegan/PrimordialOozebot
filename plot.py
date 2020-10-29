import matplotlib.pyplot as plt
import csv
import statistics
import math

plt.title('Population Diversity')
plt.ylabel('Diversity Score')
plt.xlabel('Iteration Number')

random = []
randombars = []
rmin = []
rmax = []
hill = []
hillbars = []
hmin = []
hmax = []
evo = []
emin = []
emax = []
evobars = []
cross = []
crossbars = []
cmin = []
cmax = []
numRuns = 5
numIterations = 100000000
sqrtRuns = math.sqrt(numRuns)
iterationDataRandom = []
iterationDataHill = []
iterationDataEvo = []
iterationDataCross = []

indicesToPlot = [10, 15, 20, 25]
index = 60
while indicesToPlot[-1] < numIterations:
    indicesToPlot.append(index)
    index = int(index * 1.02)
indicesToPlot[-1] = numIterations - 1

#xtiks = []
#for i in range(10):
#    xtiks.append(int(numIterations / 5 * i))
#plt.xticks(xtiks)

for i in range(1, numRuns + 1):
    iterationDataRandom.append({})
    iterationDataHill.append({})
    iterationDataEvo.append({})
    iterationDataCross.append({})
    with open('rand' + str(i) + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = 0
        for row in reversed(list(reader)):
            vals = row[0].split(',')
            iteration = int(vals[0])
            val = float(vals[1])

            while index < len(indicesToPlot) - 1 and indicesToPlot[index + 1] < iteration:
                index += 1
            
            iterationDataRandom[-1][indicesToPlot[index]] = val
    with open('hill' + str(i) + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = 0
        for row in reversed(list(reader)):
            vals = row[0].split(',')
            iteration = int(vals[0])
            val = float(vals[2])

            while index < len(indicesToPlot) - 1 and indicesToPlot[index] < iteration:
                index += 1
            iterationDataHill[-1][indicesToPlot[index]] = val
    with open('evo' + str(i) + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = 0
        for row in reversed(list(reader)):
            vals = row[0].split(',')
            iteration = int(vals[0]) * 100
            val = float(vals[2])
            while index < len(indicesToPlot) - 1 and indicesToPlot[index] < iteration:
                index += 1
            
            iterationDataEvo[-1][indicesToPlot[index]] = val
    with open('ed' + str(i) + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = 0
        for row in reversed(list(reader)):
            vals = row[0].split(',')
            iteration = int(vals[0])
            val = float(vals[2])
            while index < len(indicesToPlot) - 1 and indicesToPlot[index] < iteration:
                index += 1

            iterationDataCross[-1][indicesToPlot[index]] = val
print("Done reading data")

unifiedRandom = []
unifiedHill = []
unifiedEvo = []
unifiedCross = []
index = 0
for iteration in indicesToPlot:
    currentRandom = []
    currentHill = []
    currentEvo = []
    currentCross = []
    unifiedRandom.append(currentRandom)
    unifiedHill.append(currentHill)
    unifiedEvo.append(currentEvo)
    unifiedCross.append(currentCross)
    for run in range(numRuns):
        valRandom = -1
        if iteration in iterationDataRandom[run]:
            valRandom = iterationDataRandom[run][iteration]
        else:
            # unchanged
            valRandom = unifiedRandom[-2][run]
        currentRandom.append(valRandom)

        valHill = -1
        if iteration in iterationDataHill[run]:
            valHill = iterationDataHill[run][iteration]
        else:
            # unchanged
            valHill = unifiedHill[-2][run]
        currentHill.append(valHill)

        valEvo = -1
        if iteration in iterationDataEvo[run]:
            valEvo = iterationDataEvo[run][iteration]
        else:
            #unchanged
            valEvo = unifiedEvo[-2][run]
        currentEvo.append(valEvo)

        valCross = -1
        if iteration in iterationDataCross[run]:
            valCross = iterationDataCross[run][iteration]
        else:
            #unchanged
            valCross = unifiedCross[-2][run]
        currentCross.append(valCross)

    randomAverage = statistics.mean(currentRandom)
    randomError = statistics.stdev(currentRandom) / sqrtRuns

    random.append(randomAverage)
    randombars.append(randomError)

    hillAverage = statistics.mean(currentHill)
    hillError = statistics.stdev(currentHill) / sqrtRuns

    hill.append(hillAverage)
    hillbars.append(hillError)

    evoAverage = statistics.mean(currentEvo)
    evoError = statistics.stdev(currentEvo) / sqrtRuns

    evo.append(evoAverage)
    evobars.append(evoError)

    crossAverage = statistics.mean(currentCross)
    crossError = statistics.stdev(currentCross) / sqrtRuns

    cross.append(crossAverage)
    crossbars.append(crossError)

for i in range(len(random)):
    rmin.append(random[i] - randombars[i])
    rmax.append(random[i] + randombars[i])

    hmin.append(hill[i] - hillbars[i])
    hmax.append(hill[i] + hillbars[i])

    emin.append(evo[i] - evobars[i])
    emax.append(evo[i] + evobars[i])

    cmin.append(cross[i] - crossbars[i])
    cmax.append(cross[i] + crossbars[i])

print("Done processing data")

plt.xscale('log')
#plt.yscale('log')
#plt.plot(indicesToPlot, random, color='blue', linewidth=1, label='Random Search')
plt.plot(indicesToPlot, hill, color='green', linewidth=1, label='Parallel Hill Climb')
plt.plot(indicesToPlot, evo, color='red', linewidth=1, label='Weighted Selection')
plt.plot(indicesToPlot, cross, color='blue', linewidth=1, label='Parental Replacement')
plt.fill_between(indicesToPlot, hmin, hmax, facecolor='green', lw=0, alpha=0.5)
plt.fill_between(indicesToPlot, emin, emax, facecolor='red', lw=0, alpha=0.5)
plt.fill_between(indicesToPlot, cmin, cmax, facecolor='blue', lw=0, alpha=0.5)
#plt.fill_between(indicesToPlot, rmin, rmax, facecolor='blue', lw=0, alpha=0.5)

plt.legend(loc='best')
plt.savefig('diversityp.png', dpi=500)
plt.show()
