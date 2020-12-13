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

OozebotEncoding gen() {
    OozebotEncoding encoding = OozebotEncoding::randomEncoding();
    OozebotEncoding::evaluate(encoding);
    return encoding;
}

OozebotEncoding hill(OozebotEncoding &encoding) {
    OozebotEncoding::evaluate(encoding);
    return encoding;
}

ParetoSelector runGenerations(double mutationRate, int generationSize, int numEvaluations, std::vector<OozebotEncoding> &initialPop, ParetoFront &globalFront) {
    ParetoSelector generation(generationSize, mutationRate);
    generation.globalParetoFront = &globalFront;
    for (auto oozebot : initialPop) {
        generation.insertOozebot(oozebot);
    }

    int evaluationNumber = 0;
    while (evaluationNumber < numEvaluations) {
        evaluationNumber += generation.selectAndMate();
        printf("Finished run #%d\n", evaluationNumber);
    }

    return generation;
}

ParetoSelector runRandomSearch(int numEvaluations, int generationSize, ParetoFront &globalFront) {
    ParetoSelector generation(generationSize, 0);
    generation.globalParetoFront = &globalFront;

    std::future<OozebotEncoding> threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = std::async(&gen);
    }
    
    int j = 0;
    for (int i = 0; i < numEvaluations; i++) {
        OozebotEncoding encoding = threads[j].get();
        globalFront.evaluateEncoding(encoding);
        generation.insertOozebot(encoding);

        if (i < numEvaluations - 1) {
            if (i < numEvaluations - NUM_THREADS) {
                threads[j] = std::async(&gen);
            }
            j = (j + 1) % NUM_THREADS;
        }
        if (i != 0 && i % generationSize == 0) {
            printf("Finished run #%d\n", i);
        }
    }
    generation.sort();
    return generation;
}

ParetoSelector runRecursive(double mutationRate, int generationSize, int numEvaluations, int recursiveDepth, ParetoFront &globalFront) {
    if (recursiveDepth == 0) {
        printf("Kicking off random search\n");
        // This is equivalent to doing one random search to seed except it's easier to code up
        return runRandomSearch(numEvaluations / 10, generationSize / 2, globalFront);
    }
    ParetoSelector firstSelector = runRecursive(mutationRate, generationSize, numEvaluations, recursiveDepth - 1, globalFront);
    ParetoSelector secondSelector = runRecursive(mutationRate, generationSize, numEvaluations, recursiveDepth - 1, globalFront);
    firstSelector.sort();
    secondSelector.sort();
    std::vector<OozebotEncoding> initialPop;
    for (int i = 0; i < generationSize; i++) {
        if (i < generationSize / 2) {
            initialPop.push_back(firstSelector.generation[i].encoding);
        } else {
            initialPop.push_back(secondSelector.generation[i - generationSize / 2].encoding);
        }
    }

    printf("Kicking off generation of depth %d\n", recursiveDepth);
    return runGenerations(mutationRate, generationSize, numEvaluations, initialPop, globalFront);
}

int main() {
    // Meta objectives to consider
    // – Simplicity
    // – Evolvability
    // – Novelty / Diversity
    // – Robustness / sensitivity

    srand((unsigned int) time(NULL));

    const int numEvaluationsPerGeneration = 20000; // TODO take as a param
    const int generationSize = 500; // TODO take as a param
    double mutationRate = 0.05; // TODO take as a param

    ParetoFront globalFront;
    ParetoSelector generation = runRecursive(mutationRate, generationSize, numEvaluationsPerGeneration, 4, globalFront);

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
    while (iterSinceImprovement < 5) {
        OozebotEncoding newEncoding1 = mutate(encoding1);
        OozebotEncoding newEncoding2 = mutate(encoding2);
        newEncoding1.id = ++nextID;
        newEncoding2.id = ++nextID;
        auto handle1 = std::async(&hill, newEncoding1);
        auto handle2 = std::async(&hill, newEncoding2);
        auto res1 = handle1.get();
        auto res2 = handle2.get();
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
