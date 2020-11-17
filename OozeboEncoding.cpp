#include "oozebotEncoding.h"

// TODO, depends on the encoding we want
OozebotEncoding randomEncoding() {
    return {0, 0, 0, 1};
}

// TODO translate from encoding to masses and springs and run the sim
// TODO make this async? Likely doable by returning an opaque handle
void evaluate(OozebotEncoding encoding) {
    encoding.fitness = 0.0;
}

// TODO consider a DSL to define variables, iteration, and control flow

inline bool dominates(OozebotEncoding firstEncoding, OozebotEncoding secondEncoding) {
    return firstEncoding.fitness > secondEncoding.fitness && firstEncoding.age > secondEncoding.age && firstEncoding.weight > secondEncoding.weight;
}