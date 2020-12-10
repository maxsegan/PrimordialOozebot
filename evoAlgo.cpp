#include <math.h>
#include <vector>
#include <string>
#include <map>
#include <time.h>
#include <thread>
#include <chrono>
#include <thread>
#include <future>
#include "OozebotEncoding.h"
#include "ParetoSelector.h"

// Usage: nvcc -O2 evoAlgo.cpp -o evoAlgo -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64" cudaSim.cu OozebotEncoding.cpp ParetoSelector.cpp ParetoFront.cpp

// TODO command line args
// TODO air/water resistence

std::pair<OozebotEncoding, AsyncSimHandle> gen(int i) {
    OozebotEncoding encoding = OozebotEncoding::randomEncoding();
    AsyncSimHandle handle = OozebotEncoding::evaluate(encoding, i);
    return {encoding, handle};
}

ParetoSelector runGenerations(double mutationRate, int generationSize, int numEvaluations, std::vector<OozebotEncoding> &initialPop) {
    printf("initial pop size%d\n", (int) initialPop.size());
    ParetoSelector generation(generationSize, mutationRate);
    int i = 0;
    for (auto oozebot : initialPop) {
        printf("%d\n", i++);
        generation.insertOozebot(oozebot);
    }

    int evaluationNumber = 0;
    while (evaluationNumber < numEvaluations) {
        printf("trying to select and mate\n");
        evaluationNumber += generation.selectAndMate();
        printf("Finished run #%d\n", evaluationNumber);
    }

    return generation;
}

ParetoSelector runRandomSearch(int numEvaluations, int generationSize) {
    ParetoSelector generation(generationSize, 0);

    const int asyncThreads = 35;
    std::future<std::pair<OozebotEncoding, AsyncSimHandle>> threads[asyncThreads];
    for (int i = 0; i < asyncThreads; i++) {
        threads[i] = std::async(&gen, i + 1);
    }
    std::pair<OozebotEncoding, AsyncSimHandle> pair = gen(0);
    OozebotEncoding encoding = pair.first;
    AsyncSimHandle handle = pair.second;
    
    int j = 0;
    for (int i = 0; i < numEvaluations; i++) {
        auto res = OozebotEncoding::wait(handle);
        encoding.fitness = res.first;
        encoding.lengthAdj = res.second;
        generation.globalParetoFront.evaluateEncoding(encoding);
        generation.insertOozebot(encoding);

        if (i < numEvaluations - 1) {
            pair = threads[j].get();
            encoding = pair.first;
            handle = pair.second;
            if (i < numEvaluations - asyncThreads) {
                threads[j] = std::async(&gen, i + asyncThreads);
            }
            j = (j + 1) % asyncThreads;
        }
        if (i % generationSize == 0) {
            generation.sort();
        }
    }
    return generation;
}

ParetoSelector runRecursive(double mutationRate, int generationSize, int numEvaluations, int recursiveDepth) {
    if (recursiveDepth == 0) {
        printf("Kicking off random search\n");
        return runRandomSearch(numEvaluations / 2, generationSize);
    }
    ParetoSelector firstSelector = runRecursive(mutationRate, generationSize, numEvaluations, recursiveDepth - 1);
    ParetoSelector secondSelector = runRecursive(mutationRate, generationSize, numEvaluations, recursiveDepth - 1);
    firstSelector.sort();
    secondSelector.sort();
    std::vector<OozebotEncoding> initialPop;
    for (int i = 0; i < generationSize; i++) {
        if (i < generationSize / 2) {
            initialPop.push_back(firstSelector.generation[i].encoding);
        } else {
            initialPop.push_back(secondSelector.generation[i - generationSize / 2].encoding);
        }
        printf("%d\n", i);
    }
    printf("pop size %d\n", (int) initialPop.size());

    printf("Kicking generation of depth %d\n", recursiveDepth);
    return runGenerations(mutationRate, generationSize, numEvaluations, initialPop);
}

int main() {
    // TODO objectives - fitness, age (in log tenure groupings maybe?), weight?
    // Meta objectives to consider
    // – Simplicity
    // – Evolvability
    // – Novelty / Diversity
    // – Robustness / sensitivity

    srand((unsigned int) time(NULL));

    const int numEvaluationsPerGeneration = 30; // TODO take as a param
    const int generationSize = 15; // TODO take as a param
    double mutationRate = 0.05; // TODO take as a param

    ParetoSelector generation = runRecursive(mutationRate, generationSize, numEvaluationsPerGeneration, 3);

    // Now we hillclimb the best solution(s)
    double bestFitness = 0;
    double secondBestFitness = 0;
    int bestIndex = 0;
    int secondBestIndex = 0;
    for (int i = 0; i < generationSize; i++) {
        double fitness = generation.generation[i].encoding.fitness;
        if (fitness > bestFitness) {
            secondBestFitness = bestFitness;
            secondBestIndex = bestIndex;
            bestFitness = fitness;
            bestIndex = i;
        } else if (fitness > secondBestFitness) {
            secondBestFitness = fitness;
            secondBestIndex = i;
        }
    }
    OozebotEncoding encoding1 = generation.generation[bestIndex].encoding;
    OozebotEncoding encoding2 = generation.generation[secondBestIndex].encoding;
    int iterSinceImprovement = 0;
    unsigned long int nextID = OozebotEncoding::randomEncoding().id;
    while (iterSinceImprovement < 500) {
        OozebotEncoding newEncoding1 = mutate(encoding1);
        OozebotEncoding newEncoding2 = mutate(encoding2);
        AsyncSimHandle handle1 = OozebotEncoding::evaluate(newEncoding1, 0);
        AsyncSimHandle handle2 = OozebotEncoding::evaluate(newEncoding2, 1);
        auto res1 = OozebotEncoding::wait(handle1);
        auto res2 = OozebotEncoding::wait(handle2);
        newEncoding1.fitness = res1.first;
        newEncoding1.lengthAdj = res1.second;
        newEncoding1.id = ++nextID;
        newEncoding2.fitness = res2.first;
        newEncoding2.lengthAdj = res2.second;
        newEncoding2.id = ++nextID;
        iterSinceImprovement++;
        if (newEncoding1.fitness > encoding1.fitness) {
            encoding1 = newEncoding1;
            iterSinceImprovement = 0;
            printf("New high fitness of %f\n", encoding1.fitness);
        }
        if (newEncoding2.fitness > encoding2.fitness) {
            encoding2 = newEncoding2;
            iterSinceImprovement = 0;
            printf("New high fitness of %f\n", encoding2.fitness);
        }
    }
    logEncoding(encoding1);
    logEncoding(encoding2);

    return 0;
}
