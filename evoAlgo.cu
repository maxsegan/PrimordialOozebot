#include <math.h>
#include <vector>
#include <map>
#include "OozebotEncoding.h"
#include "ParetoSelector.h"
//#include <chrono>

// Usage: nvcc -O2 evoAlgo.cu -o evoAlgo -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64" cudaSim.cu OozebotEncoding.cu ParetoSelector.cpp ParetoFront.cpp

// TODO command line args
// TODO air/water resistence

int main() {
    // TODO objectives - fitness, age (in log tenure groupings maybe?), weight?
    // Meta objectives to consider
    // – Simplicity
    // – Evolvability
    // – Novelty / Diversity
    // – Robustness / sensitivity

    int maxEvaluations = 1000000; // TODO take as a param
    int minNumSolutions = 300; // TODO take as a param
    double mutationRate = 0.25; // TODO take as a param

    ParetoSelector generation(minNumSolutions, mutationRate);


    for (int i = 0; i < minNumSolutions; i++) {
        OozebotEncoding encoding = OozebotEncoding::randomEncoding();
        AsyncSimHandle handle = OozebotEncoding::evaluate(encoding);
        auto res = OozebotEncoding::wait(handle);
        encoding.fitness = res.first;
        encoding.numTouchesRatio = res.second;
        generation.globalParetoFront.evaluateEncoding(encoding);
        generation.insertOozebot(encoding);
    }

    int numEvaluations = minNumSolutions;
    while (numEvaluations < maxEvaluations) {
        // Regularly inject new random solutions at regular intervals
        if (numEvaluations % 16 == 0) {
            OozebotEncoding encoding = OozebotEncoding::randomEncoding();
            AsyncSimHandle handle = OozebotEncoding::evaluate(encoding);
            auto res = OozebotEncoding::wait(handle);
            encoding.fitness = res.first;
            encoding.numTouchesRatio = res.second;
            generation.globalParetoFront.evaluateEncoding(encoding);
            generation.replaceLast(encoding);
        } else {
            generation.selectAndMate();
        }
        if (numEvaluations % 100 == 0) {
            printf("Finished run #%d\n", numEvaluations);
        }
        numEvaluations += 1;
    }
    return 0;
}
