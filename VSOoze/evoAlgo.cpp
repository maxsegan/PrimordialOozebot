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

OozebotEncoding gen(double duration) {
    OozebotEncoding encoding = OozebotEncoding::randomEncoding();
    OozebotEncoding::evaluate(encoding, duration);
    return encoding;
}

std::pair<OozebotEncoding, int> hill(OozebotEncoding &encoding, double duration, int popIndex) {
    OozebotEncoding newEncoding = mutate(encoding);
    newEncoding.id = newGlobalID();
    OozebotEncoding::evaluate(newEncoding, duration);
    return { newEncoding, popIndex };
}

ParetoSelector runGenerations(double mutationRate, int generationSize, int numEvaluations, double duration, std::vector<OozebotEncoding> &initialPop, ParetoFront &globalFront) {
    ParetoSelector generation(generationSize, mutationRate);
    generation.globalParetoFront = &globalFront;
    for (auto oozebot : initialPop) {
        generation.insertOozebot(oozebot);
    }

    int evaluationNumber = 0;
    while (evaluationNumber < numEvaluations) {
        evaluationNumber += generation.selectAndMate(duration);
        printf("Finished run #%d\n", evaluationNumber);
    }

    return generation;
}

ParetoSelector hillClimb(int numEvaluations, double duration, ParetoSelector &selector, ParetoFront& globalFront) {
    int popSize = selector.generation.size() / 2;
    std::vector<OozebotEncoding> initialPop;
    for (auto wrapper : selector.generation) {
        initialPop.push_back(wrapper.encoding);
    }

    std::future<std::pair<OozebotEncoding, int>> threads[NUM_THREADS];

    int popIndex;
    for (popIndex = 0; popIndex < NUM_THREADS; popIndex++) {
        threads[popIndex] = std::async(&hill, initialPop[popIndex], duration, popIndex);
    }

    int j = 0;
    for (int i = 0; i < numEvaluations; i++) {
        auto pair = threads[j].get();
        globalFront.evaluateEncoding(pair.first);
        if (dominates(pair.first, initialPop[pair.second])) {
            initialPop[pair.second] = pair.first;
        }

        if (i < numEvaluations - 1) {
            if (i < numEvaluations - NUM_THREADS) {
                threads[j] = std::async(&hill, initialPop[popIndex], duration, popIndex);
            }
            j = (j + 1) % NUM_THREADS;
            popIndex = (popIndex + 1) % initialPop.size();
        }
        if (i != 0 && i % initialPop.size() == 0) {
            printf("Finished run #%d\n", i);
        }
    }

    ParetoSelector generation(initialPop.size(), 0);
    generation.globalParetoFront = &globalFront;
    for (auto oozebot : initialPop) {
        generation.insertOozebot(oozebot);
    }

    return generation;
}

ParetoSelector runRandomSearch(int numEvaluations, int generationSize, double duration, ParetoFront &globalFront) {
    ParetoSelector generation(generationSize, 0);
    generation.globalParetoFront = &globalFront;

    std::future<OozebotEncoding> threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = std::async(&gen, duration);
    }
    
    int j = 0;
    for (int i = 0; i < numEvaluations; i++) {
        OozebotEncoding encoding = threads[j].get();
        globalFront.evaluateEncoding(encoding);
        generation.insertOozebot(encoding);

        if (i < numEvaluations - 1) {
            if (i < numEvaluations - NUM_THREADS) {
                threads[j] = std::async(&gen, duration);
            }
            j = (j + 1) % NUM_THREADS;
        }
        if (i != 0 && i % generationSize == 0) {
            printf("Finished run #%d\n", i);
        }
    }
    return generation;
}

ParetoSelector runRecursive(double mutationRate, int generationSize, int numEvaluations, double duration, int recursiveDepth, ParetoFront &globalFront) {
    if (recursiveDepth == 0) {
        printf("Kicking off random search\n");
        // This is equivalent to doing one random search to seed except it's easier to code up
        return runRandomSearch(numEvaluations / 10, generationSize / 2, duration, globalFront);
    }
    
    ParetoSelector firstSelector = runRecursive(mutationRate / recursiveDepth, generationSize, numEvaluations, duration, recursiveDepth - 1, globalFront);
    ParetoSelector secondSelector = runRecursive(mutationRate / recursiveDepth, generationSize, numEvaluations, duration, recursiveDepth - 1, globalFront);
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
    double simDuration = duration * (1 + (double) recursiveDepth / 3.0);
    ParetoSelector selector = runGenerations(mutationRate, generationSize, numEvaluations, simDuration, initialPop, globalFront);
    return hillClimb(numEvaluations / 2, duration, selector, globalFront);
}

int main() {
    // Meta objectives to consider
    // – Simplicity
    // – Evolvability
    // – Novelty / Diversity
    // – Robustness / sensitivity

    srand((unsigned int) time(NULL));

    const int numEvaluationsPerGeneration = 10000; // TODO take as a param
    const int generationSize = 500; // TODO take as a param
    double mutationRate = 0.2; // TODO take as a param

    ParetoFront globalFront;
    ParetoSelector generation = runRecursive(mutationRate, generationSize, numEvaluationsPerGeneration, 4.5, 5, globalFront);

    return 0;
}
