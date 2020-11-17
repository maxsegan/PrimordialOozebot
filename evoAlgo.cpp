#include <math.h>
#include <vector>
#include <map>
#include "OozebotEncoding.h"
//#include <chrono>

// TODO command line args

int main() {
    // TODO SQLite integration
    // TODO accept type of selection - tournament, weighted, random from the pareto front?
    // TODO specify the representation type
    // TODO objectives - fitness, age (in log tenure groupings maybe?), weight?

    // Specify number of evaluations
    int maxEvaluations = 1000000;
    
    // Start with 100 random solutions
    int minNumSolutions = 100;

    std::vector<OozebotEncoding> generation = {};
    while (generation.size() < minNumSolutions) {
        OozebotEncoding encoding = randomEncoding();
        evaluate(encoding);
        generation.push_back(encoding);
    }
    int numEvaluations = minNumSolutions;
    // Select by Pareto-rank and Crowding distance
    // - The population is sorted into a hierarchy of sub-populations based on the ordering of Pareto dominance.
    // – Similarity between members of each sub-group is evaluated on the Pareto front
    // – The resulting groups and similarity measures are used to promote a diverse front of non-dominated solutions 

    // Also keep all time Pareto Front and inject some randomly each time?
    // Regularly inject new material?

    // Meta objectives to consider
    // – Simplicity
    // – Evolvability
    // – Novelty / Diversity
    // – Robustness / sensitivity
    // – Modularity–Cost of manufacturing


    while nextGeneration.count < generationSize {
        let index1 = sortedWeights[SR.selectWeightedRandom(weights: weights)].0
        let index2 = sortedWeights[SR.selectWeightedRandom(weights: weights)].0
        var genome = SRGenome.merge(mother: generation[index1], father: generation[index2])
        if Float.random(in: 0..<1) < 0.5 {
          genome = genome.mutate()
        }
        nextGeneration.append(genome)
      }
      generation = nextGeneration
    }
  } else {
    var iter = 0
    for genome in generation {
      let avgSumSquares = genome.avgSumSquares()
      if avgSumSquares < minAvgSumSquares {
        minAvgSumSquares = avgSumSquares
        outText = String(iter) + "," + String(minAvgSumSquares) + "," + String(SR.diversity(genomes: generation, xVals: diversityX)) + "," + String(SR.overfitting(genome: genome, xVals: xHoldout, yVals: yHoldout)) + "," + genome.description + "\n" + outText
      }
      iter += 1
    }
    
    while iter < n {
      let motherIndex = Int.random(in: 0..<generation.count)
      let mother = generation[motherIndex]
      var fatherIndex = Int.random(in: 0..<generation.count)
      if motherIndex == fatherIndex {
        fatherIndex = Int.random(in: 0..<generation.count)
      }
      let father = generation[fatherIndex]
      var c1 = SRGenome.merge(mother: mother, father: father)
      if Float.random(in: 0..<1) < 0.5 {
        c1 = c1.mutate()
      }
      var c2 = SRGenome.merge(mother: father, father: mother)
      if Float.random(in: 0..<1) < 0.5 {
        c2 = c2.mutate()
      }
      
      let motherC1D = mother.root.distance(other: c1.root)
      let fatherC1D = father.root.distance(other: c1.root)
      let motherC2D = mother.root.distance(other: c2.root)
      let fatherC2D = father.root.distance(other: c2.root)
      
      if motherC1D + fatherC2D < motherC2D + fatherC1D {
        if c1.mSS < mother.mSS {
          generation[motherIndex] = c1
        }
        if c2.mSS < father.mSS {
          generation[fatherIndex] = c2
        }
      } else {
        if c2.mSS < mother.mSS {
          generation[motherIndex] = c2
        }
        if c1.mSS < father.mSS {
          generation[fatherIndex] = c1
        }
      }
      
      if c1.mSS < minAvgSumSquares {
        minAvgSumSquares = c1.mSS
      }
      if c2.mSS < minAvgSumSquares {
        minAvgSumSquares = c2.mSS
      }
      
      iter += 2
    }
    return 0;
}

// TODO move this to another file, maybe?
int selectWeightedRandom(std::vector<double> weights) {
    double r = double(rand()) / (double(RAND_MAX) + 1.0); // [0, 1.0)
    double accum = 0.0;
    for (auto iter = weights.begin(); iter != weights.end(); iter++) {
      accum += *iter;
      if (r < accum) {
        return i;
      }
    }
    return -1;
}
