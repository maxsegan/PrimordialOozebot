#ifndef OOZEBOT_ENCODING_H
#define OOZEBOT_ENCODING_H

#include "cppSim.h"

class OozebotExpression;

struct OozebotEncoding {
    OozebotExpression[] *encodingDSL; // DSL that generates the SimInputs

    int weight; // Total mass
    double fitness; // Depends on objective - might be net displacement
    int age = 1; // Longest ancestral chain
};

struct SimInputs {
    std::vector<Point> points;
    std::vector<Spring> springs;
}

OozebotEncoding randomEncoding();

OozebotEncoding mutate(OozebotEncoding encoding);

// Evaluates and fills in fitness and related fields
void evaluate(OozebotEncoding encoding);

// Returns true if the first encoding dominates the second, false otherwise
inline bool dominates(OozebotEncoding firstEncoding, OozebotEncoding secondEncoding);

SimInputs inputsFromEncoding(OozebotEncoding encoding);

#endif
