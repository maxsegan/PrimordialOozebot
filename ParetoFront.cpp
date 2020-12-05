#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

#include "ParetoFront.h"
#include "cudaSim.h"

bool ParetoFront::evaluateEncoding(OozebotEncoding encoding) {
    this->allResults.push_back({encoding.numTouchesRatio, encoding.fitness});
    int touchesBucket = round(encoding.numTouchesRatio / this->touchesBucketSize);
    if (encoding.numTouchesRatio > this->maxTouches) {
        this->maxTouches = encoding.numTouchesRatio;
        while (touchesBucket >= buckets.size()) {
            buckets.push_back({});
        }
    }
    if (encoding.fitness > this->maxFitness && encoding.id > 20) {
        std::map<int, int> exteriorPoints = {};
        this->maxFitness = encoding.fitness;
        printf("New max fitness: %f, for ID: %ld\n", encoding.fitness, encoding.id);
        // We now log this to report - should this be here? Maybe not, but the async is annoying.
        // Prolly wanted a callback
        SimInputs inputs = OozebotEncoding::inputsFromEncoding(encoding);
        std::ofstream myfile;
        myfile.open("output/robo" + std::to_string(encoding.id) + ".txt");
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
            exteriorPoints[it - inputs.points.begin()] = i++;
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
        AsyncSimHandle handle = {inputs.points, NULL, NULL, NULL, 0, 0};
        double simDuration = 30.0;
        while (t < simDuration) {
            t += dt;
            myfile << "[\n";
            first = true;
            for (auto it = handle.points.begin(); it != handle.points.end(); ++it) {
                int i = it - handle.points.begin();
                if (exteriorPoints.find(i) == exteriorPoints.end()) {
                    continue;
                }
                if (!first) {
                    myfile << ",";
                }
                first = false;
                myfile << "[ " + std::to_string((*it).x) + ", " + std::to_string((*it).z) + ", " + std::to_string((*it).y) + "]";
            }
            if (t >= simDuration) {
                myfile << "]\n";
            } else {
                myfile << "],\n";
            }
            handle = simulate(handle.points, inputs.springs, inputs.springPresets, dt, encoding.globalTimeInterval, 0);
            resolveSim(handle);
        }
        myfile << "]\n";
        myfile << "}";
        myfile.close();
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
    return 1 / this->buckets[touchesBucket].size();  
}
