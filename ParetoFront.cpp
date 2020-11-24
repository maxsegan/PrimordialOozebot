#include <vector>
#include <algorithm>
#include <random>
#include <stdio.h>

#include "ParetoFront.h"

OozebotEncoding ParetoFront::evaluateEncoding(OozebotEncoding encoding) {
    encoding = OozebotEncoding::evaluate(encoding);

    this->allResults.push_back({encoding.age, encoding.fitness});

    int ageBucket = encoding.age / this->ageBucketSize;
    if (encoding.age > this->maxAge) {
        this->maxAge = encoding.age;
        while (ageBucket >= buckets.size()) {
            buckets.push_back({});
        }
    }
    if (encoding.fitness > this->maxFitness) {
        this->maxFitness = encoding.fitness;
        printf("New max fitness: %f, for ID: %ld\n", encoding.fitness, encoding.id);
    }
    
    int fitnessBucket = round(encoding.fitness / this->fitnessBucketSize);
    while (fitnessBucket >= this->buckets[ageBucket].size()) {
        this->buckets[ageBucket].push_back(0);
    }
    this->buckets[ageBucket][fitnessBucket] += 1;
    
    if (lastResize < this->allResults.size() / 2) {
        this->resize();
    }
    auto iter = this->encodingFront.begin();
    while (iter != this->encodingFront.end()) {
        auto frontEncoding = *iter;

        if (dominates(frontEncoding, encoding)) {
            break; // this is dominated by an existing one - by definition it can't dominate any others
        } else if (dominates(encoding, frontEncoding)) {
            iter = this->encodingFront.erase(iter); // this dominates one - it will certainly be added but also may dominate others
        } else {
            ++iter;
        }
    }
    this->encodingFront.push_back(encoding);
    return encoding;
}

void ParetoFront::resize() {
    printf("Resizing\n");
    this->lastResize = allResults.size();
    this->buckets = {};
    this->ageBucketSize = this->maxAge / 100 + 1;
    this->fitnessBucketSize = this->maxFitness / 100;

    for (auto it = this->allResults.begin(); it != this->allResults.end(); it++) {
        int age = (*it).first;
        double fitness = (*it).second;

        int ageIndex = age / this->ageBucketSize;
        int fitnessIndex = round(fitness / this->fitnessBucketSize);

        while (ageIndex >= this->buckets.size()) {
            this->buckets.push_back({});
        }
        while (fitnessIndex >= this->buckets[ageIndex].size()) {
            this->buckets[ageIndex].push_back(0);
        }
        this->buckets[ageIndex][fitnessIndex] = fitness;
    }
}

double ParetoFront::noveltyDegreeForEncoding(OozebotEncoding encoding) {
    int ageBucket = encoding.age / this->ageBucketSize;
    if (this->buckets.size() < ageBucket) {
        printf("ERROR bucket index out of bounds\n");
        return 1;
    }
    return 1 / this->buckets[ageBucket].size();  
}
