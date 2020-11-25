#ifndef OOZEBOT_ENCODING_H
#define OOZEBOT_ENCODING_H

#include <vector>
#include "cudaSim.h"

enum OozebotExpressionType {
    massDeclaration, // kg
    springDeclaration, // k, a, b, c all declared together
    boxDeclaration, // combination of springs and masses
    symmetryScope, // creation commands within this scope are duplicated flipped along the x/y/z asis
    fork, // same concept as symmetry scope but each split is independent TODO figure out the right way to encode this
    // If a block already exists at this index it noops
    layBlockAndMoveCursor, // Takes in a block idx and direction to move (up, down, left, right, forward, back)
    layAndMove,
    endScope,
};

enum OozebotDirection {
    up,
    down,
    left,
    right,
    forward,
    back,
};

enum OozebotAxis {
    x,
    y,
    z,
};

class OozebotExpression {
public:
    OozebotExpressionType expressionType;

    float kg; // 0.01 - 1 
    float k; // 500 - 10,000
    float a; // expressed as a ratio of l0's natural length 0.5-2
    float b; // -0.5 - 0.5, often 0
    float c; // 0 - 2pi
    OozebotDirection direction;
    OozebotAxis scopeAxis;
    int blockIdx; // which block to lay
    int layAndMoveIdx;
    std::vector<int> pointIdxs;
    std::vector<int> springIdxs;
};

struct SimInputs {
    std::vector<Point> points;
    std::vector<Spring> springs;
    std::vector<FlexPreset> springPresets;
};

class OozebotEncoding {
public:
    double fitness; // Depends on objective - might be net displacement
    double numTouchesRatio; // how many points ever touched the ground in the sim?
    double globalTimeInterval; // 0.1 - 1
    unsigned long int id;

    static OozebotEncoding mate(OozebotEncoding parent1, OozebotEncoding parent2);

    static SimInputs inputsFromEncoding(OozebotEncoding encoding);

    // Wait to get the fitness value - must call exactly once!
    static AsyncSimHandle evaluate(OozebotEncoding encoding);
    static std::pair<double, double> wait(AsyncSimHandle handle);

    static OozebotEncoding randomEncoding();

    // DSL that generates the SimInputs
    // TODO improve linkage
    std::vector<OozebotExpression> massCommands;
    std::vector<OozebotExpression> springCommands;
    std::vector<OozebotExpression> boxCommands;
    std::vector<std::vector<OozebotExpression>> layAndMoveSequences;
    std::vector<OozebotExpression> growthCommands;
};

OozebotEncoding mutate(OozebotEncoding encoding);

// Returns true if the first encoding dominates the second, false otherwise
inline bool dominates(OozebotEncoding firstEncoding, OozebotEncoding secondEncoding) {
    return firstEncoding.fitness >= secondEncoding.fitness && firstEncoding.numTouchesRatio <= secondEncoding.numTouchesRatio;
}

#endif
