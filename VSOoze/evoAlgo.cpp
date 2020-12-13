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

std::pair<OozebotEncoding, SimInputs> gen() {
    OozebotEncoding encoding = OozebotEncoding::randomEncoding();
    return {encoding, OozebotEncoding::inputsFromEncoding(encoding)};
}

ParetoSelector runGenerations(double mutationRate, int generationSize, int numEvaluations, std::vector<OozebotEncoding> &initialPop, ParetoFront &globalFront, std::vector<AsyncSimHandle> &handles) {
    ParetoSelector generation(generationSize, mutationRate);
    generation.globalParetoFront = &globalFront;
    for (auto oozebot : initialPop) {
        generation.insertOozebot(oozebot);
    }

    int evaluationNumber = 0;
    while (evaluationNumber < numEvaluations) {
        evaluationNumber += generation.selectAndMate(handles);
        printf("Finished run #%d\n", evaluationNumber);
    }

    return generation;
}

ParetoSelector runRandomSearch(int numEvaluations, int generationSize, ParetoFront &globalFront, std::vector<AsyncSimHandle> &handles) {
    ParetoSelector generation(generationSize, 0);
    generation.globalParetoFront = &globalFront;

    std::future<std::pair<OozebotEncoding, SimInputs>> threads[NUM_THREADS];
    std::pair<OozebotEncoding, SimInputs> pairs[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i] = std::async(&gen);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pairs[i] = threads[i].get();
        OozebotEncoding::evaluate(pairs[i].second, pairs[i].first, handles[i]);
        threads[i] = std::async(&gen);
    }
    
    int j = 0;
    for (int i = 0; i < numEvaluations; i++) {
        auto res = OozebotEncoding::wait(handles[j]);
        pairs[j].first.fitness = res.first;
        pairs[j].first.lengthAdj = res.second;
        globalFront.evaluateEncoding(pairs[j].first);
        generation.insertOozebot(pairs[j].first);
        printf("id: %lu fitness: %f\n", pairs[j].first.id, res.first);

        if (i < numEvaluations - 1) {
            if (i < numEvaluations - NUM_THREADS) {
                pairs[j] = threads[j].get();
                OozebotEncoding::evaluate(pairs[j].second, pairs[j].first, handles[j]);
            }
            if (i < numEvaluations - 2 * NUM_THREADS) {
                threads[j] = std::async(&gen);
            }
            j = (j + 1) % NUM_THREADS;
        }
        if (i != 0 && i % generationSize == 0) {
            printf("Finished run #%d\n\n", i);
        }
    }
    generation.sort();
    return generation;
}

ParetoSelector runRecursive(double mutationRate, int generationSize, int numEvaluations, int recursiveDepth, ParetoFront &globalFront, std::vector<AsyncSimHandle> &handles) {
    if (recursiveDepth == 0) {
        printf("Kicking off random search\n");
        return runRandomSearch(numEvaluations / 10, generationSize / 2, globalFront, handles);
    }
    ParetoSelector firstSelector = runRecursive(mutationRate, generationSize, numEvaluations, recursiveDepth - 1, globalFront, handles);
    ParetoSelector secondSelector = runRecursive(mutationRate, generationSize, numEvaluations, recursiveDepth - 1, globalFront, handles);
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
    return runGenerations(mutationRate, generationSize, numEvaluations, initialPop, globalFront, handles);
}

int main() {
    // Meta objectives to consider
    // – Simplicity
    // – Evolvability
    // – Novelty / Diversity
    // – Robustness / sensitivity

    srand((unsigned int) time(NULL));

    const int numEvaluationsPerGeneration = 20000; // TODO take as a param
    const int generationSize = 300; // TODO take as a param
    double mutationRate = 0.05; // TODO take as a param

    ParetoFront globalFront;
    std::vector<AsyncSimHandle> handles;
    for (int i = 0; i < NUM_THREADS; i++) {
        handles.push_back(createSimHandle(i));
    }

    ParetoSelector generation = runRecursive(mutationRate, generationSize, numEvaluationsPerGeneration, 4, globalFront, handles);

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
        SimInputs inputs1 = OozebotEncoding::inputsFromEncoding(newEncoding1);
        SimInputs inputs2 = OozebotEncoding::inputsFromEncoding(newEncoding2);
        OozebotEncoding::evaluate(inputs1, newEncoding1, handles[0]);
        OozebotEncoding::evaluate(inputs2, newEncoding2, handles[1]);
        auto res1 = OozebotEncoding::wait(handles[0]);
        auto res2 = OozebotEncoding::wait(handles[1]);
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
    logEncoding(encoding1, handles[0]);
    logEncoding(encoding2, handles[1]);

    for (auto handle : handles) {
        releaseSimHandle(handle);
    }

    return 0;
}
