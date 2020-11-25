#include <vector>
#include <algorithm>
#include <random>
#include <stdio.h>

#include "ParetoFront.h"

bool ParetoFront::evaluateEncoding(OozebotEncoding encoding) {
    this->allResults.push_back({encoding.numTouchesRatio, encoding.fitness});
    int touchesBucket = round(encoding.numTouchesRatio / this->touchesBucketSize);
    if (encoding.numTouchesRatio > this->maxTouches) {
        this->maxTouches = encoding.numTouchesRatio;
        while (touchesBucket >= buckets.size()) {
            buckets.push_back({});
        }
    }
    if (encoding.fitness > this->maxFitness) {
        this->maxFitness = encoding.fitness;
        printf("New max fitness: %f, for ID: %ld\n", encoding.fitness, encoding.id);
    }
    
    int fitnessBucket = round(encoding.fitness / this->fitnessBucketSize);
    while (fitnessBucket >= this->buckets[touchesBucket].size()) {
        this->buckets[touchesBucket].push_back(0);
    }
    this->buckets[touchesBucket][fitnessBucket] += 1;
    
    if (lastResize < this->allResults.size() / 2) {
        this->resize();
    }
    auto iter = this->encodingFront.begin();
    while (iter != this->encodingFront.end()) {
        auto frontEncoding = *iter;

        if (dominates(frontEncoding, encoding)) {
            return false; // this is dominated by an existing one - by definition it can't dominate any others
        } else if (dominates(encoding, frontEncoding)) {
            iter = this->encodingFront.erase(iter); // this dominates one - it will certainly be added but also may dominate others
        } else {
            ++iter;
        }
    }
    this->encodingFront.push_back(encoding);
    return true;
}

void ParetoFront::resize() {
    this->lastResize = allResults.size();
    this->buckets = {};
    this->touchesBucketSize = this->maxTouches / 100;
    this->fitnessBucketSize = this->maxFitness / 100;

    for (auto it = this->allResults.begin(); it != this->allResults.end(); it++) {
        double touches = (*it).first;
        double fitness = (*it).second;

        int touchesIndex = round(touches / this->touchesBucketSize);
        int fitnessIndex = round(fitness / this->fitnessBucketSize);

        while (touchesIndex >= this->buckets.size()) {
            this->buckets.push_back({});
        }
        while (fitnessIndex >= this->buckets[touchesIndex].size()) {
            this->buckets[touchesIndex].push_back(0);
        }
        this->buckets[touchesIndex][fitnessIndex] = fitness;
    }
}

double ParetoFront::noveltyDegreeForEncoding(OozebotEncoding encoding) {
    int touchesBucket = round(encoding.numTouchesRatio / this->touchesBucketSize);
    if (this->buckets.size() < touchesBucket) {
        printf("ERROR bucket index out of bounds\n");
        return 1;
    }
    return 1 / this->buckets[touchesBucket].size();  
}
