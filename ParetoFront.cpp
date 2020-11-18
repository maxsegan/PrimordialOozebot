#include "ParetoFront.h"
#include <vector>
#include <algorithm>
#include <random>

class ParetoFront {
    std::vector<OozebotEncoding> encodingFront;
};

std::vector<OozebotEncoding> getRandomEncodings(ParetoFront front, size_t n) {
    std::vector<OozebotEncoding> out;
    std::sample(
        ParetoFront.encodingFront.begin(),
        ParetoFront.encodingFront.end(),
        std::back_inserter(out),
        n,
        std::mt19937{std::random_device{}()}
    );
    return out;
}

bool evaluateEncoding(ParetoFront front, OozebotEncoding encoding) {
    auto iter = front.encodingFront.begin();
    while (iter != front.encodingFront.end()) {
        auto frontEncoding = *iter;

        if (dominates(frontEncoding, encoding)) {
            return false; // this is dominated by an existing one - by definition it can't dominate any others
        } else if (dominates(encoding, frontEncoding)) {
            iter = iter.erase(iter); // this dominates one - it will certainly be added but also may dominate others
        } else {
            ++iter;
        }
    }

    return true;
}
