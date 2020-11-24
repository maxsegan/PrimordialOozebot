#include "ParetoSelector.h"
#include "OozebotEncoding.h"
#include "ParetoFront.h"
#include <vector>
#include <algorithm>
#include <stdio.h>

// M: # of objectives/dimensions in the pareto front
// N: Size of generation
// K: # of globally undominated solutions found to date
bool sortFunction(OozebotSortWrapper a, OozebotSortWrapper b) {
    return (a.novelty > b.novelty);
}

// Insertion is O(M * N) plus cost of sort - O(N^2) - plus cost of tracking globally O(K)
void ParetoSelector::insertOozebot(OozebotEncoding encoding) {
    OozebotSortWrapper wrapper = {encoding, {}, {}, 0, 0};

    for (std::vector<OozebotSortWrapper>::iterator iter = this->generation.begin(); iter != this->generation.end(); iter++) {
        if (dominates(encoding, (*iter).encoding)) {
            (*iter).dominated.push_back(encoding.id);
            (*iter).dominationDegree += 1;
            wrapper.dominating.push_back((*iter).encoding.id);
        } else if (dominates((*iter).encoding, encoding)) {
            wrapper.dominated.push_back((*iter).encoding.id);
            wrapper.dominationDegree += 1;
            (*iter).dominating.push_back(encoding.id);
        }
    }
    this->idToIndex[encoding.id] = this->generation.size();
    this->generation.push_back(wrapper);
}

void ParetoSelector::removeOozebotAtIndex(int i) {
    OozebotSortWrapper wrapper = this->generation[i];
    unsigned long int removedId = wrapper.encoding.id;
    for (auto it = wrapper.dominating.begin(); it != wrapper.dominating.end(); ++it) {
        int index = this->idToIndex[(*it)];
        this->generation[index].dominationDegree -= 1;
        for (auto iter = this->generation[index].dominated.begin(); iter != this->generation[index].dominated.end(); iter++) {
            if ((*iter) == removedId) {
                this->generation[index].dominated.erase(iter);
                break;
            }
        }
    }
    for (auto it = wrapper.dominated.begin(); it != wrapper.dominated.end(); ++it) {
        int idx = this->idToIndex[(*it)];
        for (auto iter = this->generation[idx].dominating.begin(); iter != this->generation[idx].dominating.end(); iter++) {
            if ((*iter) == removedId) {
                this->generation[idx].dominating.erase(iter);
                break;
            }
        }
    }
    this->generation.erase(this->generation.begin() + i);
    while (i < generation.size()) {
        unsigned long int id = generation[i].encoding.id;
        this->idToIndex[id] -= 1;
        i++;
    }
}

void ParetoSelector::replaceLast(OozebotEncoding encoding) {
    this->removeOozebotAtIndex(this->generationSize - 1);
    this->insertOozebot(encoding);
}

// Crowding is maintained by dividing the entire
// search space deterministically in subspaces, where is the
// depth parameter and is the number of decision variables, and
// by updating the subspaces dynamically
void ParetoSelector::selectAndMate() {
    this->sort();
    // TODO go by indices so we can do multiple at the same time more easily
    int i = this->selectionIndex();
    int j = this->selectionIndex();
    while (j == i) {
        j = this->selectionIndex();
    }
    OozebotEncoding child = OozebotEncoding::mate(this->generation[i].encoding, this->generation[j].encoding);
    double r = (double) rand() / RAND_MAX;
    if (r < this->mutationProbability) {
        child = mutate(child);
    }
    child = globalParetoFront.evaluateEncoding(child);

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
        if (childNovelty > jNovelty) {
            this->removeOozebotAtIndex(j);
            this->insertOozebot(child);
        }
    }
}

// Sort is O(N^2)
void ParetoSelector::sort() {
    std::vector<std::vector<OozebotSortWrapper>> workingVec;
    int numLeft = this->generation.size();
    while (numLeft > 0) {
        std::vector<OozebotSortWrapper> nextTier;
        for (std::vector<OozebotSortWrapper>::iterator iter = this->generation.begin(); iter != this->generation.end(); iter++) {
            if ((*iter).dominationDegree == 0) {
                (*iter).dominationDegree -= 1; // invalidates it for the rest of iterations
                (*iter).novelty = this->globalParetoFront.noveltyDegreeForEncoding((*iter).encoding); // These get stale so must recompute
                nextTier.push_back(*iter);
            }
        }
        std::sort(nextTier.begin(), nextTier.end(), sortFunction);
        for (auto iter = nextTier.begin(); iter != nextTier.end(); iter++) {
            for (auto it = (*iter).dominating.begin(); it != (*iter).dominating.end(); ++it) {
                int index = this->idToIndex[(*it)];
                this->generation[index].dominationDegree -= 1;
            }
        }
        workingVec.push_back(nextTier);
        numLeft -= nextTier.size();
    }
    this->idToIndex.clear();
    std::vector<OozebotSortWrapper> nextGeneration;
    nextGeneration.reserve(this->generation.size());
    for (auto it = workingVec.begin(); it != workingVec.end(); ++it) {
        for (auto iter = (*it).begin(); iter != (*it).end(); ++iter) {
            (*iter).dominationDegree = (*iter).dominated.size();
            this->idToIndex[(*iter).encoding.id] = nextGeneration.size();
            nextGeneration.push_back(*iter);
            if (nextGeneration.size() == this->generationSize) {
                break;
            }
        }
    }
    this->generation = nextGeneration;
    for (auto it = this->generation.begin(); it != this->generation.end(); ++it) {
    }
}

int ParetoSelector::selectionIndex() {
    double r = (double) rand() / RAND_MAX;

    double accumulation = 0;
    int i = -1;
    while (accumulation < r) {
        i += 1;
        accumulation += this->indexToProbability[i];
    }
    return i;
}
