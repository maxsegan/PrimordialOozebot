#ifndef OOZEBOT_ENCODING_H
#define OOZEBOT_ENCODING_H

#include <vector>

class OozebotExpression;

class OozebotEncoding {
    double fitness; // Depends on objective - might be net displacement
    int age = 1; // Longest ancestral chain
    double globalTimeInterval; // 0.1 - 1

    static OozebotEncoding mate(OozebotEncoding parent1, OozebotEncoding parent2);

    static SimInputs inputsFromEncoding(OozebotEncoding encoding);

    // Evaluates and fills in fitness and related fields
    static void evaluate(OozebotEncoding encoding);

private:
    // DSL that generates the SimInputs
    // TODO improve linkage
    std::vector<OozebotExpression> massCommands;
    std::vector<OozebotExpression> springCommands;
    std::vector<OozebotExpression> boxCommands;
    std::vector<std::vector<OozebotExpression>> layAndMoveSequences;
    std::vector<OozebotExpression> growthCommands;
};

struct SimInputs {
    std::vector<Point> points;
    std::vector<Spring> springs;
    std::vector<FlexPreset> springPresets;
};

OozebotEncoding randomEncoding();

void mutate(OozebotEncoding encoding);

// Returns true if the first encoding dominates the second, false otherwise
inline bool dominates(OozebotEncoding firstEncoding, OozebotEncoding secondEncoding);

#endif
