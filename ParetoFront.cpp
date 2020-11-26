#include <vector>
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
    if (encoding.fitness > this->maxFitness) {
        this->maxFitness = encoding.fitness;
        printf("New max fitness: %f, for ID: %ld\n", encoding.fitness, encoding.id);
        // We now log this to report - should this be here? Maybe not, but the async is annoying.
        // Prolly wanted a callback
        SimInputs inputs = OozebotEncoding::inputsFromEncoding(encoding);
        std::ofstream myfile;
        myfile.open("output/robo" + std::to_string(encoding.id) + ".txt");
        // Janky JSON bc meh it's simple
        myfile << "{\n";
        myfile << "\"name\": \"robo" + std::to_string(encoding.id) + "\",\n";
        myfile << "\"masses\" : [\n";
        for (auto it = inputs.points.begin(); it != inputs.points.end(); ++it) {
            myfile << "[ " + std::to_string((*it).x) + ", " + std::to_string((*it).z) + ", " + std::to_string((*it).y) + "]";
            if (it + 1 == inputs.points.end()) {
                myfile << "\n";
            } else {
                myfile << ",\n";
            }
        }
        myfile << "],\n";
        myfile << "\"springs\" : [\n";
        for (auto it = inputs.springs.begin(); it != inputs.springs.end(); ++it) {
            myfile << "[ " + std::to_string((*it).p1) + ", " + std::to_string((*it).p2) + "]";
            if (it + 1 == inputs.springs.end()) {
                myfile << "\n";
            } else {
                myfile << ",\n";
            }
        }
        myfile << "],\n";
        myfile << "\"simulation\" : [\n";
        double t = 0;
        double dt = 1.0 / 24.0; // 24fps
        auto points = inputs.points;
        while (t < 10) {
            AsyncSimHandle handle = simulate(points, inputs.springs, inputs.springPresets, dt, encoding.globalTimeInterval);
            resolveSim(handle);
            points = handle.points;
            t += dt;
            myfile << "[\n";
            for (auto it = handle.points.begin(); it != handle.points.end(); ++it) {
                myfile << "[ " + std::to_string((*it).x) + ", " + std::to_string((*it).z) + ", " + std::to_string((*it).y) + "]";
                if (it + 1 == handle.points.end()) {
                    myfile << "\n";
                } else {
                    myfile << ",\n";
                }
            }
            if (t >= 10) {
                myfile << "]\n";
            } else {
                myfile << "],\n";
            }
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
    if (this->buckets.size() < touchesBucket) {
        printf("ERROR bucket index out of bounds\n");
        return 1;
    }
    return 1 / this->buckets[touchesBucket].size();  
}
