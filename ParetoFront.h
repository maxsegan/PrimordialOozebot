#ifndef PARETO_FRONT_H
#define PARETO_FRONT_H

#include <utility>
#include <vector>
#include "OozebotEncoding.h"

void logEncoding(OozebotEncoding encoding);

class ParetoFront {
public:
    // This functions will add the evaluated encoding and invalidate others appropriately
    bool evaluateEncoding(OozebotEncoding encoding);

    // 1 if very novel, asymptotes to 0 as it's less novel
    double noveltyDegreeForEncoding(OozebotEncoding encoding);

private:
    std::vector<OozebotEncoding> encodingFront;
    std::vector<std::pair<double, double>> allResults;
    std::vector<std::vector<int>> buckets;
    double lengthAdjBucketSize = 0.1;
    double fitnessBucketSize = 0.1;
    double maxLengthAdj = 0;
    double maxFitness = 0;
    int lastResize = 10;

    void resize();
};

#endif