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

ParetoSelector runGenerations(double mutationRate, int generationSize, int numEvaluations, std::vector<OozebotEncoding> initialPop, string logToken) {
    ParetoSelector generation(generationSize, mutationRate);
    for (auto oozebot : initialPop) {
        generation.insertOozebot(oozebot);
    }

    int evaluationNumber = 0;
    while (evaluationNumber < numEvaluations) {
        evaluationNumber += generation.selectAndMate();
        printf("Finished run #%d for eval %s\n", evaluationNumber, logToken);
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

int main() {
    // TODO objectives - fitness, age (in log tenure groupings maybe?), weight?
    // Meta objectives to consider
    // – Simplicity
    // – Evolvability
    // – Novelty / Diversity
    // – Robustness / sensitivity

    srand((unsigned int) time(NULL));

    const int maxEvaluations = 20000; // TODO take as a param
    const int minNumSolutions = 300; // TODO take as a param
    double mutationRate = 0.05; // TODO take as a param

    ParetoSelector generation(minNumSolutions, mutationRate);

    int numEvaluations = randomSeedNum;
    // In this stage do baseball leagues too, maybe 100k iterations, then create another one (recursive) as it's competitor
    // TODO baseball leagues
    // TODO hill climb at the end of each generation
    return 0;
}
