#ifndef PARETO_FRONT_H
#define PARETO_FRONT_H

#include <utility>
#include <vector>
#include "OozebotEncoding.h"

class ParetoFront {
public:
    // Returns true if it is on the global pareto front, false otherwise
    // This functions will add the evaluated encoding and invalidate others appropriately
    bool evaluateEncoding(OozebotEncoding encoding);

    // 1 if very novel, asymptotes to 0 as it's less novel
    double noveltyDegreeForEncoding(OozebotEncoding encoding);

private:
    std::vector<OozebotEncoding> encodingFront;
    std::vector<std::pair<int, double>> allResults;
    std::vector<std::vector<int>> buckets;
    int ageBucketSize;
    int fitnessBucketSize;
    int maxAge;
    double maxFitness;
    int lastResize;

    void resize();
};

#endif