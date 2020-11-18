#include <math.h>
#include <vector>
#include <map>
#include "OozebotEncoding.h"
//#include <chrono>

// TODO command line args
// TODO air/water resistence

int main() {
    // TODO SQLite integration
    // TODO accept type of selection - tournament, weighted, random from the pareto front?
    // TODO specify the representation type
    // TODO objectives - fitness, age (in log tenure groupings maybe?), weight?

    // TODO specify mutation rate (10%? 5%? 25%? depending on age?)

    // Specify number of evaluations
    int maxEvaluations = 1000000;
    ParetoFront globalParetoFront = { {} };
    
    // Start with 100 random solutions
    int minNumSolutions = 100;

    std::vector<OozebotEncoding> generation = {};
    while (generation.size() < minNumSolutions) {
        OozebotEncoding encoding = randomEncoding();
        evaluate(encoding);
        generation.push_back(encoding);
        globalParetoFront.evaluateEncoding(encoding);
    }

    int numEvaluations = minNumSolutions;

    // O( M * N^2 ) if we're smart about it and keep domination count and domination indices
    // One full sort (M * N ^2) followed by a decrementing procedure that's worst N^2
    // OR we can can do this incrementally one node at a time while we sim - inserting seems (M * N)
    // Rank by pareto front -> within each front rank by crowding sort

    // children are compared to parent - if they dominate they replace
    // If neither dominates, child is compared to the global pareto front - if it's in it it replaces the parent
    // If not, we keep it if it's in a less crowded region than the parent

    // Crowding is maintained by dividing the entire
    // search space deterministically in subspaces, where is the
    // depth parameter and is the number of decision variables, and
    // by updating the subspaces dynamically

    // Regularly inject new random solutions at regular intervals

    // Meta objectives to consider
    // – Simplicity
    // – Evolvability
    // – Novelty / Diversity
    // – Robustness / sensitivity
    // – Modularity–Cost of manufacturing
    return 0;
}

// TODO move this to another file, maybe?
int selectWeightedRandom(std::vector<double> weights) {
    double r = double(rand()) / (double(RAND_MAX) + 1.0); // [0, 1.0)
    double accum = 0.0;
    for (auto iter = weights.begin(); iter != weights.end(); iter++) {
      accum += *iter;
      if (r < accum) {
        return i;
      }
    }
    return -1;
}
