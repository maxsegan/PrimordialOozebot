#ifndef OOZEBOT_ENCODING_H
#define OOZEBOT_ENCODING_H

#include <vector>
#include "cudaSim.h"

enum OozebotExpressionType {
    boxDeclaration, // combination of springs and masses - one size mass (kg), and spring config for all springs (k, a, b, c)
    layAndMove, // Building block commands to form creation instructions
    // Strings together a sequence of layAndMove commands with a "thickness" that's 1D, 2D, or 3D with radius provided
    layBlockAndMoveCursor, // Takes in a lay and move index to iterator as well as "anchor" direction to grow out of + thickness
    symmetryScope, // creation commands within this scope are duplicated flipped along the x/y/z asis
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
    xAxis,
    yAxis,
    zAxis,
    noAxis,
};

class OozebotExpression {
public:
    OozebotExpressionType expressionType;

    float kg; // 0.01 - 1 
    float k; // 500 - 10,000
    float a; // expressed as a ratio of l0's natural length 0.5-2
    float b; // -0.5 - 0.5, often 0
    float c; // 0 - 2pi
    int blockIdx; // which block to lay
    OozebotDirection direction;
    OozebotAxis scopeAxis;
    int layAndMoveIdx;
    int radius;
    OozebotAxis thicknessIgnoreAxis;
    // For extremities we grow out of the surface block reached from the center point moving in steps of these magnitude
    int anchorX;
    int anchorY;
    int anchorZ;
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
    static AsyncSimHandle evaluate(OozebotEncoding encoding, int streamNum);
    static std::pair<double, double> wait(AsyncSimHandle handle);

    static OozebotEncoding randomEncoding();

    // DSL that generates the SimInputs
    std::vector<OozebotExpression> boxCommands;
    std::vector<std::vector<OozebotExpression>> layAndMoveCommands;
    OozebotExpression bodyCommand;
    std::vector<OozebotExpression> growthCommands;
};

OozebotEncoding mutate(OozebotEncoding encoding);

// Returns true if the first encoding dominates the second, false otherwise
inline bool dominates(OozebotEncoding firstEncoding, OozebotEncoding secondEncoding) {
    return firstEncoding.fitness >= secondEncoding.fitness && firstEncoding.numTouchesRatio <= secondEncoding.numTouchesRatio;
}

#endif
