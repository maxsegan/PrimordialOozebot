#ifndef OOZEBOT_ENCODING_H
#define OOZEBOT_ENCODING_H

struct OozebotEncoding {
    int x; // TODO
    int weight;
    double fitness;
    int age = 1;
};

OozebotEncoding randomEncoding();

// Evaluates and fills in fitness and related fields
void evaluate(OozebotEncoding encoding);

// Returns true if the first encoding dominates the second, false otherwise
inline bool dominates(OozebotEncoding firstEncoding, OozebotEncoding secondEncoding);

#endif
