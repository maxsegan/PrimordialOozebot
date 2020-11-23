#include "ParetoSelector.h"
#include "OozebotEncoding.h"
#include "ParetoFront.h"
#include <vector>
#include <algorithm>

// M: # of objectives/dimensions in the pareto front
// N: Size of generation
// K: # of globally undominated solutions found to date
bool sortFunction(OozebotSortWrapper a, OozebotSortWrapper b) {
    return (a.novelty > b.novelty);
}

// Insertion is O(M * N) plus cost of sort - O(N^2) - plus cost of tracking globally O(K)
void ParetoSelector::insertOozebot(OozebotEncoding encoding) {
    evaluate(encoding);
    this->globalParetoFront.evaluateEncoding(encoding);
    auto generation = this->generation;
    OozebotSortWrapper wrapper = {encoding, {}, 0, 0, 1};

    for (std::vector<OozebotSortWrapper>::iterator iter = generation.begin(); iter != generation.end(); iter++) {
        if (dominates(encoding, (*iter).encoding)) {
            (*iter).dominatingIndices.push_back(generation.size());
            (*iter).dominationDegree += 1;
            (*iter).tmpDominationDegree += 1;
        } else if (dominates((*iter).encoding, encoding)) {
            int index = iter - generation.begin();
            wrapper.dominatingIndices.push_back(index);
            wrapper.dominationDegree += 1;
            wrapper.tmpDominationDegree += 1;
        }
    }
    this->generation.push_back(wrapper);
}

void ParetoSelector::removeOozebotAtIndex(int i) {
    OozebotSortWrapper wrapper = this->generation[i];
    for (auto it = wrapper.dominatingIndices.begin; it != wrapper.dominatingIndices.end(); ++it) {
        this->generation[*it].dominationDegree -= 1;
        this->generation[*it].tmpDominationDegree -= 1;
    }
    this->generationSize.erase(i);
}

// Crowding is maintained by dividing the entire
// search space deterministically in subspaces, where is the
// depth parameter and is the number of decision variables, and
// by updating the subspaces dynamically
void ParetoSelector::selectAndMate() {
    this->sort();
    int i = this->selectionIndex();
    int j = this->selectionIndex();
    while (j == i) {
        j = this->selectionIndex();
    }
    OozebotEncoding child = OozebotEncoding::mate(this->generation[i].encoding, this->generation[j].encoding);
    double r = (double) rand() / RAND_MAX;
    if (r < this->mutationProbability) {
        mutate(child);
    }

    // children are compared to parent - if they dominate they replace
    // If neither dominates, child is compared to the global pareto front - if it's in it it replaces the parent
    // If not, we keep it if it's in a less crowded region than the parent
    if (dominates(child, this->generation[i].encoding)) {
        this->removeOozebotAtIndex(i);
        this->insertOozebot(child);
    } else if (dominates(child, this->generation[j].encoding)) {
        this->removeOozebotAtIndex(j);
        this->insertOozebot(child);
    } else if (!dominates(this->generation[i].encoding, child)) {
        double childNovelty = this->globalParetoFront.noveltyDegreeForEncoding(child);
        double iNovelty = this->globalParetoFront.noveltyDegreeForEncoding(this->generation[i].encoding);
        if (childNovelty > iNovelty) {
            this->removeOozebotAtIndex(i);
            this->insertOozebot(child);
        }
    } else if (!dominates(this->generation[j].encoding, child)) {
        double childNovelty = this->globalParetoFront.noveltyDegreeForEncoding(child);
        double jNovelty = this->globalParetoFront.noveltyDegreeForEncoding(this->generation[j].encoding);
        if (childNovelty > iNovelty) {
            this->removeOozebotAtIndex(j);
            this->insertOozebot(child);
        }
    }
}

// Sort is O(N^2)
void ParetoSelector::sort() {
    auto generation = this->generation;
    std::vector<std::vector<OozebotSortWrapper>> workingVec;
    int numLeft = generation.size();
    while (numLeft > 0) {
        std::vector<OozebotSortWrapper> nextTier;
        for (std::vector<OozebotSortWrapper>::iterator iter = generation.begin(); iter != generation.end(); iter++) {
            if ((*iter).tmpDominationDegree == 0) {
                (*iter).tmpDominationDegree -= 1; // invalidates it for the rest of iterations
                (*iter).novelty = this->globalParetoFront.noveltyDegreeForEncoding((*iter).encoding); // These get stale so must recompute
                for (auto it = (*iter).dominatingIndices.begin(); it != (*iter).dominatingIndices.end(); ++it) {
                    generation[*it].tmpDominationDegree -= 1;
                }
                nextTier.push_back(*iter);
            }
        }
        std::sort(nextTier.begin(), nextTier.end(), sortFunction);
        workingVec.push_back(nextTier);
        numLeft -= nextTier.size();
    }
    std::vector<OozebotSortWrapper> nextGeneration;
    nextGeneration.reserve(generation.size());
    for (auto it = workingVec.begin(); it != workingVec.end(); ++it) {
        for (auto iter = (*it).begin(); iter != (*it).end(); ++iter) {
            (*iter).tmpDominationDegree = (*iter).dominationDegree;
            nextGeneration.push_back(*iter);
            if (nextGeneration.size() == this->generationSize) {
                break;
            }
        }
    }
    this->generation = nextGeneration;
}

int ParetoSelector::selectionIndex() {
    double r = (double) rand() / RAND_MAX;
    double accumulation = 0;
    int i = -1;
    int lastValidIndex = this->generationSize / 2;
    while (accumulation < r) {
        i += 1;
        accumulation += this->indexToProbability[i];
    }
    return i;
}
