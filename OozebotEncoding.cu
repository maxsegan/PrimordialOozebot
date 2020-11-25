#define _USE_MATH_DEFINES
#include <cmath>

#include <math.h>
#include <algorithm>
#include <utility>
#include <map>

#include "oozebotEncoding.h"

const int kNumMasses = 6;
const int kNumSprings = 8;
const int kNumBoxes = 6;
const int kMaxLayAndMoveSequences = 10;
const int kMaxGrowthCommands = 10;
const int kMaxLayAndMoveLength = 100;

signed long int GlobalId = 1;

bool massSortFunction(OozebotExpression a, OozebotExpression b) {
    return (a.kg > b.kg);
}

bool springSortFunction(OozebotExpression a, OozebotExpression b) {
    return (a.b > b.b);
}

OozebotEncoding OozebotEncoding::randomEncoding() {
    // Create masses
    std::vector<OozebotExpression> massCommands;
    massCommands.reserve(kNumMasses);
    for (int i = 0; i < kNumMasses; i++) {
        OozebotExpression massExpression;
        massExpression.expressionType = massDeclaration;
        massExpression.kg = 0.1; // 0.01 + (rand() / RAND_MAX) * 0.99
        massCommands.push_back(massExpression);
    }
    std::sort(massCommands.begin(), massCommands.end(), massSortFunction);

    std::vector<OozebotExpression> springCommands;
    springCommands.reserve(kNumSprings);
    for (int i = 0; i < kNumSprings; i++) {
        OozebotExpression springExpression;
        springExpression.expressionType = springDeclaration;
        double r = (double) rand() / RAND_MAX; // 0 to 1
        springExpression.k = 500 + r * 9500;
        r = (double) rand() / RAND_MAX; // 0 to 1
        if (r < 0.5) { // half the time have it be 1
            springExpression.a = 1;
        } else {
            r = (double) rand() / RAND_MAX; // 0 to 1
            springExpression.a = 0.5 + r * 1.5;
        }
        r = (double) rand() / RAND_MAX; // 0 to 1
        if (r < 0.5) { // half the time have it not expand/contract
            springExpression.b = 0;
        } else {
            r = (double) rand() / RAND_MAX; // 0 to 1
            springExpression.b = -0.5 + r;
        }
        r = (double) rand() / RAND_MAX; // 0 to 1
        springExpression.c = r * 2 * M_PI;
        springCommands.push_back(springExpression);
    }
    std::sort(springCommands.begin(), springCommands.end(), springSortFunction);

    std::vector<OozebotExpression> boxCommands;
    boxCommands.reserve(kNumBoxes);
    for (int i = 0; i < kNumBoxes; i++) {
        OozebotExpression boxCreationExpression;
        boxCreationExpression.expressionType = boxDeclaration;
        for (int j = 0; j < 8; j++) {
            boxCreationExpression.pointIdxs.push_back(rand() % kNumMasses);
        }
        for (int j = 0; j < 28; j++) {
            boxCreationExpression.springIdxs.push_back(rand() % kNumSprings);
        }
        boxCommands.push_back(boxCreationExpression);
    }

    std::vector<std::vector<OozebotExpression>> layAndMoveSequences;
    // TMP Part B
    std::vector<OozebotExpression> sequence;

    OozebotExpression midExpression;
    midExpression.expressionType = layBlockAndMoveCursor;
    midExpression.blockIdx = 0;
    midExpression.direction = left;

    OozebotExpression sideExpression;
    sideExpression.expressionType = layBlockAndMoveCursor;
    sideExpression.blockIdx = 1;
    sideExpression.direction = forward;

    OozebotExpression legExpression;
    legExpression.expressionType = layBlockAndMoveCursor;
    legExpression.blockIdx = 2;
    legExpression.direction = down;

    sequence.push_back(midExpression);
    sequence.push_back(midExpression);
    sequence.push_back(midExpression);

    sequence.push_back(sideExpression);
    sequence.push_back(sideExpression);
    sequence.push_back(sideExpression);
    sequence.push_back(sideExpression);

    sequence.push_back(legExpression);
    sequence.push_back(legExpression);
    sequence.push_back(legExpression);
    sequence.push_back(legExpression);
    sequence.push_back(legExpression);
    sequence.push_back(legExpression);

    layAndMoveSequences.push_back(sequence);
    /*layAndMoveSequences.reserve(kMaxLayAndMoveSequences);
    for (int i = 0; i < kMaxLayAndMoveSequences; i++) {
        std::vector<OozebotExpression> sequence;
        sequence.reserve(kMaxLayAndMoveLength);
        for (int j = 0; j < kMaxLayAndMoveLength; j++) {
            OozebotExpression layAndMoveExpression;
            layAndMoveExpression.expressionType = layAndMoveExpression;
            // Add bias to duplicate direction and type
            if (j > 0) {
                double r = (double) rand() / RAND_MAX; // 0 to 1
                if (r < 0.5) {
                    layAndMoveExpression.direction = sequence[j - 1].direction;
                } else {
                    layAndMoveExpression.direction = rand() % 6;
                }
                r = (double) rand() / RAND_MAX; // 0 to 1
                if (r < 0.8) {
                    layAndMoveExpression.blockIdx = sequence[j - 1].blockIdx;
                } else {
                    layAndMoveExpression.blockIdx = rand() % kNumMasses;
                }
            } else {
                layAndMoveExpression.direction = rand() % 6;
                layAndMoveExpression.blockIdx = rand() % kNumMasses;
            }

            sequence.push_back(layAndMoveExpression);

            double r = (double) rand() / RAND_MAX; // 0 to 1
            if (r < 0.02) { // Don't always have to be full length, end early 2% of the time for each iteration
                break;
            }
        }
        layAndMoveSequences.push_back(sequence);
    }*/

    std::vector<OozebotExpression> growthCommands;

    OozebotExpression zExpression;
    OozebotExpression xExpression;
    OozebotExpression placeExpression;

    zExpression.expressionType = symmetryScope;
    zExpression.scopeAxis = z;

    xExpression.expressionType = symmetryScope;
    xExpression.scopeAxis = x;

    placeExpression.expressionType = layAndMove;
    placeExpression.layAndMoveIdx = 0;

    growthCommands.push_back(zExpression);
    growthCommands.push_back(xExpression);
    growthCommands.push_back(placeExpression);

    /*for (int i = 0; i < kMaxGrowthCommands; i++) {
        double r = (double) rand() / RAND_MAX; // 0 to 1
        OozebotExpression growthExpression;
        if (r < 0.05) {
            growthExpression.expressionType = fork;
            // todo figure out how to close
        } else if (r < 0.2) else {
            growthExpression.expressionType = symmetryScope;
            growthExpression.scopeAxis = rand() % 3;
        } else if (r < 0.4) {
            growthExpression.expressionType = endScope;
        } else {
            growthExpression.expressionType = layAndMove;
            growthExpression.layAndMoveIdx = rand() % kMaxLayAndMoveSequences;
        }
        growthCommands.push_back(growthExpression);
        r = (double) rand() / RAND_MAX;
        if (r < 0.1) { // keep sequences shorter than max length
            break;
        }
    }*/

    OozebotEncoding encoding;
    double r = (double) rand() / RAND_MAX; // 0 to 1
    encoding.globalTimeInterval = 0.1 + r * 0.9;
    encoding.numTouchesRatio = 0;
    encoding.id = GlobalId++;
    encoding.massCommands = massCommands;
    encoding.springCommands = springCommands;
    encoding.boxCommands = boxCommands;
    encoding.layAndMoveSequences = layAndMoveSequences;
    encoding.growthCommands = growthCommands;
    return encoding;
}

OozebotEncoding OozebotEncoding::mate(OozebotEncoding parent1, OozebotEncoding parent2) {
    OozebotEncoding child;
    child.massCommands = parent1.massCommands; // TODO others
    child.springCommands = {};
    for (int i = 0; i < kNumSprings / 2; i++) {
        child.springCommands.push_back(parent1.springCommands[i]);
    }
    for (int i = kNumSprings / 2; i < kNumSprings; i++) {
        child.springCommands.push_back(parent2.springCommands[i]);
    }
    std::sort(child.springCommands.begin(), child.springCommands.end(), springSortFunction);
    child.boxCommands = {};
    for (int i = 0; i < kNumBoxes / 2; i++) {
        child.boxCommands.push_back(parent1.boxCommands[i]);
    }
    for (int i = kNumBoxes / 2; i < kNumBoxes; i++) {
        child.boxCommands.push_back(parent2.boxCommands[i]);
    }
    child.layAndMoveSequences = parent1.layAndMoveSequences;
    child.growthCommands = parent1.growthCommands;
    //std::vector<OozebotExpression> springCommands;
    //std::vector<std::vector<OozebotExpression>> layAndMoveSequences;
    //std::vector<OozebotExpression> growthCommands;
    child.id = GlobalId++;
    child.globalTimeInterval = parent1.globalTimeInterval;
    return child;
}

// TODO update things other than springs
OozebotEncoding mutate(OozebotEncoding encoding) {
    int springIndex = rand() % encoding.springCommands.size();
    double seed = (double) rand() / RAND_MAX - 0.5; // -0.5 to 0.5
    int r = rand() % 2;
    if (r == 0) {
        // Update a spring
        r = rand() % 4;
        if (r == 0) {
            double k = encoding.springCommands[springIndex].k;
            k += std::min(std::max(seed * 100, 500.0), 10000.0);
            encoding.springCommands[springIndex].k = k;
        } else if (r == 1) {
            double a = encoding.springCommands[springIndex].a;
            a += std::min(std::max(seed * 0.1, 0.5), 2.0);
            encoding.springCommands[springIndex].a = a;
        } else if (r == 2) {
            double b = encoding.springCommands[springIndex].b;
            b += std::min(std::max(seed * 0.05, -0.5), 0.5);
            encoding.springCommands[springIndex].b = b;
        } else {
            double c = encoding.springCommands[springIndex].c;
            c += std::min(std::max(seed * 0.1, 0.0), 2 * M_PI);
            encoding.springCommands[springIndex].c = c;
        }
    } else {
        // Update a box
        int boxIndex = rand() % kNumBoxes;
        r = rand() % 2;
        if (r == 0) {
            int pointIndex = rand() % kNumMasses;
            encoding.boxCommands[boxIndex].pointIdxs[pointIndex] = rand() % kNumMasses;
        } else {
            int springIndex = rand() % kNumMasses;
            encoding.boxCommands[boxIndex].springIdxs[springIndex] = rand() % kNumSprings;
        }
    }
    return encoding;
}

AsyncSimHandle OozebotEncoding::evaluate(OozebotEncoding encoding) {
    SimInputs inputs = OozebotEncoding::inputsFromEncoding(encoding);
    if (inputs.points.size() == 0) {
        return { {}, NULL, NULL, NULL};
    }
    auto points = inputs.points;
    auto springs = inputs.springs;
    auto springPresets = inputs.springPresets;
   
    return simulate(points, springs, springPresets, 5.0, encoding.globalTimeInterval);
}

std::pair<double, double> OozebotEncoding::wait(AsyncSimHandle handle) {
    resolveSim(handle);
    if (handle.points.size() == 0) {
        return {0, 0};
    }
    double end = 0;
    bool hasNan = false;
    int numTouches = 0;
    for (auto iter = handle.points.begin(); iter != handle.points.end(); ++iter) {
        end += (*iter).x;
        if (isnan((*iter).x) || isinf((*iter).x)) {
            printf("Solution has NaN or inf\n");
            hasNan = true;
        }
        if ((*iter).timestampsContactGround > 0) {
            numTouches += 1;
        }
    }
    end = end / handle.points.size();
    double fitness = hasNan ? 0 : abs(end - handle.start);
    return {fitness, (double)numTouches / handle.points.size()};
}

void layBlockAtPosition(
    int x,
    int y,
    int z,
    std::vector<Point> &points,
    std::vector<Spring> &springs,
    std::map<int, std::map<int, std::map<int, int>>> &pointLocationToIndexMap,
    std::map<int, std::map<int, bool>> &pointIndexHasSpring,
    std::vector<OozebotExpression> massCommands,
    std::vector<OozebotExpression> springCommands,
    OozebotExpression boxCommand) {
    int i = 0;
    std::vector<int> pointIndices;
    // first make the points
    for (int xi = x; xi < x + 2; xi++) {
        for (int yi = y; yi < y + 2; yi++) {
            for (int zi = z; zi < z + 2; zi++) {
                if (pointLocationToIndexMap.find(xi) == pointLocationToIndexMap.end()) {
                    std::map<int, std::map<int, int>> innerMap;
                    pointLocationToIndexMap[xi] = innerMap;
                }
                if (pointLocationToIndexMap[xi].find(yi) == pointLocationToIndexMap[xi].end()) {
                    std::map<int, int> innermostMap;
                    pointLocationToIndexMap[xi][yi] = innermostMap;
                }
                if (pointLocationToIndexMap[xi][yi].find(zi) == pointLocationToIndexMap[xi][yi].end()) {
                    // It wasn't already there so we add it
                    pointLocationToIndexMap[xi][yi][zi] = points.size();
                    OozebotExpression pointExpression = massCommands[boxCommand.pointIdxs[i]];
                    Point p = {xi / 10.0f, yi / 10.0f, zi / 10.0f, 0, 0, 0, pointExpression.kg, 0, 0};
                    points.push_back(p);
                }
                pointIndices.push_back(pointLocationToIndexMap[xi][yi][zi]);
                i++;
            }
        }
    }
    // now make the springs
    i = 0;
    for (int ii = 0; ii < pointIndices.size(); ii++) {
        for (int jj = ii + 1; jj < pointIndices.size(); jj++) {
            int first = std::min(pointIndices[ii], pointIndices[jj]);
            int second = std::max(pointIndices[ii], pointIndices[jj]);
            // always index from smaller to bigger so we don't have to double bookkeep
            if (pointIndexHasSpring.find(first) == pointIndexHasSpring.end()) {
                std::map<int, bool> innerMap;
                pointIndexHasSpring[first] = innerMap;
            }
            if (pointIndexHasSpring[first].find(second) == pointIndexHasSpring[first].end()) {
                pointIndexHasSpring[first][second] = true;
                OozebotExpression springExpression = springCommands[boxCommand.springIdxs[i]];
                Point p1 = points[first];
                Point p2 = points[second];
                float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                Spring s = {springExpression.k, first, second, length, p1.numSprings, p2.numSprings, boxCommand.springIdxs[i]};
                springs.push_back(s);
                points[first].numSprings += 1;
                points[second].numSprings += 1;
            }
            i++;
        }
    }
}

int recursiveBuildABot(
    std::vector<Point> &points,
    std::vector<Spring> &springs,
    std::map<int, std::map<int, std::map<int, int>>> &pointLocationToIndexMap,
    std::map<int, std::map<int, bool>> &pointIndexHasSpring,
    std::vector<OozebotExpression> growthCommands,
    std::vector<OozebotExpression> massCommands,
    std::vector<OozebotExpression> springCommands,
    std::vector<OozebotExpression> boxCommands,
    std::vector<std::vector<OozebotExpression>> layAndMoveSequences,
    int x,
    int y,
    int z,
    bool invertX,
    bool invertY,
    bool invertZ,
    int commandIdx) {
    if (commandIdx >= growthCommands.size()) {
        return 100;
    }
    OozebotExpression growthCommand = growthCommands[commandIdx];
    int minY = 100;
    switch (growthCommand.expressionType) {
        case symmetryScope:
        {
            int firstMinY = recursiveBuildABot(
                points,
                springs,
                pointLocationToIndexMap,
                pointIndexHasSpring,
                growthCommands,
                massCommands,
                springCommands,
                boxCommands,
                layAndMoveSequences,
                x,
                y,
                z,
                invertX,
                invertY,
                invertZ,
                commandIdx + 1);

            int secondMinY = recursiveBuildABot(
                points,
                springs,
                pointLocationToIndexMap,
                pointIndexHasSpring,
                growthCommands,
                massCommands,
                springCommands,
                boxCommands,
                layAndMoveSequences,
                x,
                y,
                z,
                growthCommand.scopeAxis == x ? !invertX : invertX,
                growthCommand.scopeAxis == y ? !invertY : invertY,
                growthCommand.scopeAxis == z ? !invertZ : invertZ,
                commandIdx + 1);

            return std::min(firstMinY, secondMinY);
        }
        case layAndMove:
        {
            std::vector<OozebotExpression> layAndMoveSequence = layAndMoveSequences[growthCommand.layAndMoveIdx];
            for (auto iter = layAndMoveSequence.begin(); iter != layAndMoveSequence.end(); ++iter) {
                OozebotExpression cmd = *iter;
                // First we lay the current block
                layBlockAtPosition(x, y, z, points, springs, pointLocationToIndexMap, pointIndexHasSpring, massCommands, springCommands, boxCommands[cmd.blockIdx]);
                minY = std::min(minY, y); // only update minY when we actually lay a block - otherwise we could end at a new low without laying

                // Now we move
                OozebotDirection direction = cmd.direction;
                switch (direction) {
                    case up:
                        if (invertY == false) {
                            y += 1;
                        } else {
                            y -= 1;
                        }
                        break;
                    case down:
                        if (invertY == false) {
                            y -= 1;
                        } else {
                            y += 1;
                        }
                        break;
                    case left:
                        if (invertZ == false) {
                            z -= 1;
                        } else {
                            z += 1;
                        }
                        break;
                    case right:
                        if (invertZ == false) {
                            z += 1;
                        } else {
                            z -= 1;
                        }
                        break;
                    case forward:
                        if (invertX == false) {
                            x += 1;
                        } else {
                            x -= 1;
                        }
                        break;
                    case back:
                        if (invertX == false) {
                            x -= 1;
                        } else {
                            x += 1;
                        }
                        break;
                }
                x = std::max(std::min(x, 100), -100);
                y = std::max(std::min(y, 100), -100);
                z = std::max(std::min(z, 100), -100);
            }

            int otherMinY = recursiveBuildABot(
                points,
                springs,
                pointLocationToIndexMap,
                pointIndexHasSpring,
                growthCommands,
                massCommands,
                springCommands,
                boxCommands,
                layAndMoveSequences,
                x,
                y,
                z,
                invertX,
                invertY,
                invertZ,
                commandIdx + 1);
            return std::min(minY, otherMinY);
        }
        default:
        {
            printf("Unexpected expression\n");
            return -1;
        }
    }
}

SimInputs OozebotEncoding::inputsFromEncoding(OozebotEncoding encoding) {
    std::vector<Point> points;
    std::vector<Spring> springs;
    std::vector<FlexPreset> presets;

    for (auto it = encoding.springCommands.begin(); it != encoding.springCommands.end(); it++) {
        FlexPreset p = {(*it).a, (*it).b, (*it).c};
        presets.push_back(p);
    }

    // All indexes are points in 3 space times 10 (position on tenth of a meter, index by integer)
    // Largest value is 100, smallest is -100 on each axis
    std::map<int, std::map<int, std::map<int, int>>> pointLocationToIndexMap;
    std::map<int, std::map<int, bool>> pointIndexHasSpring;

    int minY = recursiveBuildABot(
        points,
        springs,
        pointLocationToIndexMap,
        pointIndexHasSpring,
        encoding.growthCommands,
        encoding.massCommands,
        encoding.springCommands,
        encoding.boxCommands,
        encoding.layAndMoveSequences,
        0,
        0,
        0,
        false,
        false,
        false,
        0);

    // ground robot on lowest point
    for (auto it = points.begin(); it != points.end(); ++it) {
        (*it).y -= (double(minY) / 10.0);
    }

    return { points, springs, presets };
}
