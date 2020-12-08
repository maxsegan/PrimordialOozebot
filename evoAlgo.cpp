#include <math.h>
#include <vector>
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

int main() {
    // TODO objectives - fitness, age (in log tenure groupings maybe?), weight?
    // Meta objectives to consider
    // – Simplicity
    // – Evolvability
    // – Novelty / Diversity
    // – Robustness / sensitivity

    srand((unsigned int) time(NULL));

    int maxEvaluations = 50000; // TODO take as a param
    const int minNumSolutions = 300; // TODO take as a param
    double mutationRate = 0.05; // TODO take as a param

    ParetoSelector generation(minNumSolutions, mutationRate);

    const int asyncThreads = 5;

    std::future<std::pair<OozebotEncoding, AsyncSimHandle>> threads[asyncThreads];
    for (int i = 0; i < asyncThreads; i++) {
        threads[i] = std::async(&gen, i + 1);
    }
    std::pair<OozebotEncoding, AsyncSimHandle> pair = gen(0);
    OozebotEncoding encoding = pair.first;
    AsyncSimHandle handle = pair.second;
    
    int j = 0;
    const int randomSeedNum = 2000;
    for (int i = 0; i < randomSeedNum; i++) {
        auto res = OozebotEncoding::wait(handle);
        encoding.fitness = res.first;
        encoding.lengthAdj = res.second;
        generation.globalParetoFront.evaluateEncoding(encoding);
        generation.insertOozebot(encoding);

        if (i < randomSeedNum - 1) {
            pair = threads[j].get();
            encoding = pair.first;
            handle = pair.second;
            if (i < randomSeedNum - asyncThreads) {
                threads[j] = std::async(&gen, i + asyncThreads);
            }
            j = (j + 1) % asyncThreads;
        }
    }

    int numEvaluations = randomSeedNum;
    // In this stage do baseball leagues too, maybe 100k iterations, then create another one (recursive) as it's competitor
    // TODO baseball leagues
    while (numEvaluations < maxEvaluations) {
        numEvaluations += generation.selectAndMate();
        printf("Finished run #%d\n", numEvaluations);
    }

    // Now we hillclimb the best solution
    generation.sort();
    OozebotEncoding encoding1 = generation.generation[0].encoding;
    OozebotEncoding encoding2 = generation.generation[0].encoding;
    int iterSinceImprovement = 0;
    unsigned long int nextID = numEvaluations;
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
    generation.globalParetoFront.evaluateEncoding(encoding1);
    generation.globalParetoFront.evaluateEncoding(encoding2);
    return 0;
}
