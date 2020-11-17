#ifndef PARETO_FRONT_H
#define PARETO_FRONT_H

#include <vector>
#include "OozebotEncoding.h"

struct ParetoFront {
    std::vector<OozebotEncoding> encodingFront;
};

std::vector<OozebotEncoding> getRandomEncodings(ParetoFront front, size_t n);

// Returns true if it is on the global pareto front, false otherwise
// This functions will add the evaluated encoding and invalidate others appropriately
bool evaluateEncoding(ParetoFront front, OozebotEncoding encoding);

#endif