#ifndef PARETO_SELECTOR_H
#define PARETO_SELECTOR_H

#include <map>

#include "OozebotEncoding.h"
#include "ParetoFront.h"

// Indices don't work bc we sort... need to ID these and do indices just at sort time - otherwise track IDs
struct OozebotSortWrapper {
    OozebotEncoding encoding;
    std::vector<signed long int> dominating; // Which ids do we dominate?
    std::vector<signed long int> dominated; // Which ids dominate us?
    int dominationDegree; // How many times are we dominated? Tmp variable for sorting
    double novelty;
};

class ParetoSelector {
public:
    ParetoFront globalParetoFront;
    const int generationSize;
    const double mutationProbability;

    ParetoSelector(int numGeneration, double mutationProbability):generationSize(numGeneration), mutationProbability(mutationProbability) {
        int divisor = numGeneration * (numGeneration - 1) / 2 + numGeneration * 2;
        for (int i = numGeneration - 1; i >= 0; i--) {
            this->indexToProbability.push_back((double) (i + 2) / divisor);
        }
        this->indexToProbability.push_back(1.0); // handle small rounding error
    }

    void insertOozebot(OozebotEncoding encoding);

    void selectAndMate();

    void replaceLast(OozebotEncoding encoding);

private:
    std::vector<OozebotSortWrapper> generation;
    std::vector<double> indexToProbability;
    std::map<signed long int, int> idToIndex;

    void sort();
    void removeOozebotAtIndex(int i);
    int selectionIndex();
};

#endif