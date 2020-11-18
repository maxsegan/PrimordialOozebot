#include "oozebotEncoding.h"

class OozebotExpression {
    // TODO abstract class
}

// TODO define subclasses - define point properties, spring properties, boxes, growths, etc
// Each spring type includes a, b, c, and k
// TODO define type system to make it easier to generate valid expressions
// TODO define valid ranges for all types
// TODO a DSL to define variables, iteration, and control flow

// TODO, depends on the encoding we want
OozebotEncoding randomEncoding() {
    return {0, 0, 0, 1};
}

// TODO implement small mutations
OozebotEncoding mutate(OozebotEncoding encoding) {
    return encoding;
}

// TODO translate from encoding to masses and springs and run the sim
// TODO make this async? Likely doable by returning an opaque handle
void evaluate(OozebotEncoding encoding) {
    encoding.fitness = 0.0;
}

inline bool dominates(OozebotEncoding firstEncoding, OozebotEncoding secondEncoding) {
    return firstEncoding.fitness > secondEncoding.fitness && firstEncoding.age > secondEncoding.age && firstEncoding.weight > secondEncoding.weight;
}

SimInputs inputsFromEncoding(OozebotEncoding encoding) {
    return { {}, {} }; // TODO
}
