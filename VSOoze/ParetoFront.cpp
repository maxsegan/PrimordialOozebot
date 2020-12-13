#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <future>
#include <thread>

#include "ParetoFront.h"
#include "cudaSim.h"

void logEncoding(OozebotEncoding &encoding, AsyncSimHandle &handle) {
    printf("New encoding on pareto front: %d with fitness: %f length adj: %f\n", encoding.id, encoding.fitness, encoding.lengthAdj);
    std::map<int, int> exteriorPoints = {};
    // We now log this to report - should this be here? Maybe not, but the async is annoying.
    // Prolly wanted a callback
    SimInputs inputs = OozebotEncoding::inputsFromEncoding(encoding);
    std::ofstream myfile;
    myfile.open("output/robo" + std::to_string(encoding.id) + "-" + std::to_string(encoding.fitness) + ".txt");
    // Janky JSON bc meh it's simple
    myfile << "{\n";
    myfile << "\"name\" : \"robo" + std::to_string(encoding.id) + "\",\n";
    myfile << "\"masses\" : [\n";
    int i = 0;
    bool first = true;
    for (auto it = inputs.points.begin(); it != inputs.points.end(); ++it) {
        if ((*it).numSprings == 26) { // If it has 26 it's on the interior so we ignore
            continue;
        }
        if (!first) {
            myfile << ",";
        }
        first = false;
        exteriorPoints[(int) (it - inputs.points.begin())] = i++;
        myfile << "[ " + std::to_string((*it).x) + ", " + std::to_string((*it).z) + ", " + std::to_string((*it).y) + "]";
    }
    myfile << "],\n";
    myfile << "\"springs\" : [\n";
    first = true;
    for (auto it = inputs.springs.begin(); it != inputs.springs.end(); ++it) {
        if (exteriorPoints.find((*it).p1) == exteriorPoints.end() || exteriorPoints.find((*it).p2) == exteriorPoints.end()) {
            continue;
        }
        if (!first) {
            myfile << ",";
        }
        first = false;
        myfile << "[ " + std::to_string(exteriorPoints[(*it).p1]) + ", " + std::to_string(exteriorPoints[(*it).p2]) + "]";
    }
    myfile << "],\n";
    myfile << "\"simulation\" : [\n";
    double t = 0;
    double dt = 1.0 / 24.0; // 24fps
    simulate(handle, inputs.points, inputs.springs, inputs.springPresets, 0, encoding.globalTimeInterval); // 0s to just capture initial conditions
    synchronize(handle);
    double simDuration = 30.0;
    while (t < simDuration) {
        myfile << "[\n";
        first = true;
        for (int i = 0; i < handle.numPoints; i++) {
            if (exteriorPoints.find(i) == exteriorPoints.end()) {
                continue;
            }
            if (!first) {
                myfile << ",";
            }
            Point p = handle.endPoints[i];
            first = false;
            myfile << "[ " + std::to_string(p.x) + ", " + std::to_string(p.z) + ", " + std::to_string(p.y) + "]";
        }
        if (t + dt >= simDuration) {
            myfile << "]\n";
        } else {
            myfile << "],\n";
            simulateAgain(handle, inputs.springPresets, t, t + dt, encoding.globalTimeInterval);
            synchronize(handle);
        }
        t += dt;
    }
    myfile << "]\n";
    myfile << "}";
    myfile.close();
}

bool ParetoFront::evaluateEncoding(OozebotEncoding encoding) {
    this->allResults.push_back({encoding.lengthAdj, encoding.fitness});
    int lengthAdjBucket = (int) round(encoding.lengthAdj / this->lengthAdjBucketSize);

    if (encoding.lengthAdj > this->maxLengthAdj) {
        this->maxLengthAdj = encoding.lengthAdj;
    }
    if (encoding.fitness > this->maxFitness) {
        this->maxFitness = encoding.fitness;
    }
    
    while (lengthAdjBucket >= buckets.size()) {
        buckets.push_back({});
    }
    int fitnessBucket = (int) round(encoding.fitness / this->fitnessBucketSize);
    while (fitnessBucket >= this->buckets[lengthAdjBucket].size()) {
        this->buckets[lengthAdjBucket].push_back(0);
    }
    this->buckets[lengthAdjBucket][fitnessBucket] += 1;
    
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
    std::thread(logEncoding, encoding, this->loggingHandle).detach();

    return true;
}

void ParetoFront::resize() {
    this->lastResize = (int) allResults.size();
    this->buckets = {};
    this->lengthAdjBucketSize = this->maxLengthAdj / 100;
    this->fitnessBucketSize = this->maxFitness / 100;

    for (auto it = this->allResults.begin(); it != this->allResults.end(); it++) {
        double lengthAdj = (*it).first;
        double fitness = (*it).second;

        int lengthAdjIndex = (int) round(lengthAdj / this->lengthAdjBucketSize);
        int fitnessIndex = (int) round(fitness / this->fitnessBucketSize);

        while (lengthAdjIndex >= this->buckets.size()) {
            this->buckets.push_back({});
        }
        while (fitnessIndex >= this->buckets[lengthAdjIndex].size()) {
            this->buckets[lengthAdjIndex].push_back(0);
        }
        this->buckets[lengthAdjIndex][fitnessIndex] += 1;
    }
}

double ParetoFront::noveltyDegreeForEncoding(OozebotEncoding encoding) {
    int lengthAdjBucket = (int) round(encoding.lengthAdj / this->lengthAdjBucketSize);
    int fitnessBucket = (int) round(encoding.fitness / this->fitnessBucketSize);
    return (double) 1 / this->buckets[lengthAdjBucket][fitnessBucket];  
}

ParetoFront::~ParetoFront() {
    releaseSimHandle(this->loggingHandle);
}
