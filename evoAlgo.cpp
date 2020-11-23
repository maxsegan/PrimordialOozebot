#include <math.h>
#include <vector>
#include <map>
#include "OozebotEncoding.h"
#include "ParetoSelector.h"
//#include <chrono>

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
    double mutationRate = 0.05; // TODO take as a param

    ParetoSelector generation(minNumSolutions, mutationRate);

    while (generation.size() < minNumSolutions) {
        OozebotEncoding encoding = randomEncoding();
        generation.insertOozebot(encoding);
    }

    int numEvaluations = minNumSolutions;
    while (numEvaluations < maxEvaluations) {
        // Regularly inject new random solutions at regular intervals
        if (numEvaluations % 16 == 0) {
            OozebotEncoding encoding = randomEncoding();
            generation.insertOozebot(encoding);
        } else {
            generation.selectAndMate();
        }
        numEvaluations += 1;
    }
    return 0;
}
