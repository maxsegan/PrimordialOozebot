#ifndef PARETO_SELECTOR_H
#define PARETO_SELECTOR_H

#include "OozebotEncoding.h"
#include "ParetoFront.h"

struct OozebotSortWrapper {
    OozebotEncoding encoding;
    std::vector<int> dominatingIndices; // Which indices dominate us?
    int dominationDegree; // How many times are we dominated?
    int tmpDominationDegree; // Mutable for bookkeeping during sort
    double novelty;
};

class ParetoSelector {
public:
    ParetoFront globalParetoFront;
    const int generationSize;
    const double mutationProbability;

    ParetoSelector(int numGeneration, double mutationProbability) {
        generationSize = numGeneration;
        mutationProbability = mutationProbability;
        int divisor = numGeneration * (numGeneration - 1) / 2 + numGeneration * 2;
        for (int i = generationSize / 2 - 1; i < 0; i--) {
            this->indexToProbability.push_back((i+2) / divisor);
        }
    }

    void insertOozebot(OozebotEncoding encoding);

    void selectAndMate();

private:
    std::vector<OozebotSortWrapper> generation;
    std::vector<double> indexToProbability;

    void sort();
    void removeOozebotAtIndex(int i);
    int selectionIndex();
};

#endif