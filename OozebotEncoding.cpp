#include <math.h>
#include <algorithm>

#include "oozebotEncoding.h"
#include "cppSim.h"

const kNumMasses = 6;
const kNumSprings = 10;
const kNumBoxes = 6;
const kMaxLayAndMoveSequences = 10;
const kMaxGrowthCommands = 10;
const kMaxLayAndMoveLength = 100;

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
    OozebotExpressionType expressionType;

    double kg; // 0.01 - 1 
    double k; // 500 - 10,000
    double a; // expressed as a ratio of l0's natural length 0.5-2
    double b; // -0.5 - 0.5, often 0
    double c; // 0 - 2pi
    OozebotDirection direction;
    OozebotAxis scopeAxis;
    int blockIdx; // which block to lay
    int layAndMoveIdx;
    std::vector<int> pointIdxs;
    std::vector<int> springIdxs;
};

bool massSortFunction(OozebotExpression a, OozebotExpression b) {
    return (a.kg > b.kg);
}

bool springSortFunction(OozebotExpression a, OozebotExpression b) {
    return (a.b > b.b);
}

OozebotEncoding randomEncoding() {
    // Create masses
    std::vector<OozebotExpression> massCommands;
    massCommands.reserve(kNumMasses);
    for (int i = 0; i < kNumMasses; i++) {
        OozebotExpression massExpression;
        massExpression.expressionType = OozebotExpressionType.massDeclaration;
        massExpression.k = 0.1; // 0.01 + (rand() / RAND_MAX) * 0.99
        massCommands.push_back(massExpression);
    }
    std::sort(massCommands.begin(), massCommands.end(), massSortFunction);

    std::vector<OozebotExpression> springCommands;
    springCommands.reserve(kNumSprings);
    for (int i = 0; i < kNumSprings; i++) {
        OozebotExpression springExpression;
        springExpression.expressionType = OozebotExpressionType.springDeclaration;
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
        OozebotExpression boxExpression;
        boxExpression.expressionType = OozebotExpressionType.boxExpression;
        for (int j = 0; j < 8; j++) {
            boxExpression.pointIdxs.push_back(rand() % kNumPoints);
        }
        for (int j = 0; j < 28; j++) {
            boxExpression.springIdxs.push_back(rand() % kNumSprings);
        }
        boxCommands.push_back(boxExpression);
    }

    std::vector<std::vector<OozebotExpression>> layAndMoveSequences;
    // TMP Part B
    std::vector<OozebotExpression> sequence;

    OozebotExpression midExpression;
    midExpression.expressionType = OozebotExpressionType.layBlockAndMoveCursor;
    midExpression.blockIdx = 0;
    midExpression.direction = OozebotDirection.left;

    OozebotExpression sideExpression;
    sideExpression.expressionType = OozebotExpressionType.layBlockAndMoveCursor;
    sideExpression.blockIdx = 1;
    sideExpression.direction = OozebotDirection.forward;

    OozebotExpression legExpression;
    legExpression.expressionType = OozebotExpressionType.layBlockAndMoveCursor;
    legExpression.blockIdx = 2;
    legExpression.direction = OozebotDirection.down;

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
            layAndMoveExpression.expressionType = OozebotExpressionType.layAndMoveExpression;
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

    zExpression.expressionType = OozebotExpressionType.symmetryScope;
    zExpression.scopeAxis = OozebotAxis.z;

    xExpression.expressionType = OozebotExpressionType.symmetryScope;
    xExpression.scopeAxis = OozebotAxis.x;

    placeExpression.expressionType = OozebotExpressionType.layAndMove;
    placeExpression.layAndMoveIdx = 0;

    growthCommands.push_back(zExpression);
    growthCommands.push_back(xExpression);
    growthCommands.push_back(placeExpression);

    /*for (int i = 0; i < kMaxGrowthCommands; i++) {
        double r = (double) rand() / RAND_MAX; // 0 to 1
        OozebotExpression growthExpression;
        if (r < 0.05) {
            growthExpression.expressionType = OozebotExpressionType.fork;
            // todo figure out how to close
        } else if (r < 0.2) else {
            growthExpression.expressionType = OozebotExpressionType.symmetryScope;
            growthExpression.scopeAxis = rand() % 3;
        } else if (r < 0.4) {
            growthExpression.expressionType = OozebotExpressionType.endScope;
        } else {
            growthExpression.expressionType = OozebotExpressionType.layAndMove;
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
    encoding.age = 1;
    encoding.massCommands = massCommands;
    encoding.springCommands = springCommands;
    encoding.layAndMoveSequences = layAndMoveSequences;
    encoding.growthCommands = growthCommands;
    return encoding;
}

static OozebotEncoding OozebotEncoding::mate(OozebotEncoding parent1, OozebotEncoding parent2) {
    OozebotEncoding child;
    child.massCommands = parent1.massCommands; // TODO others
    child.springCommands = {};
    for (int i = 0; i < kNumMasses / 2; i++) {
        child.springCommands.push_back(parent1.springCommands[i]);
    }
    for (int i = kNumMasses / 2; i < kNumMasses; i++) {
        child.springCommands.push_back(parent2.springCommands[i]);
    }
    std::sort(child.springCommands.begin(), child.springCommands.end(), springSortFunction);
    child.layAndMoveSequences = parent1.layAndMoveSequences;
    child.growthCommands = parent1.growthCommands;
    std::vector<OozebotExpression> springCommands;
    std::vector<std::vector<OozebotExpression>> layAndMoveSequences;
    std::vector<OozebotExpression> growthCommands;
    child.age = max(parent1.age, parent2.age) + 1;
    child.globalTimeInterval = parent1.globalTimeInterval;
    return child;
}

// TODO update things other than springs
void mutate(OozebotEncoding encoding) {
    int springIndex = rand() % encoding.springCommands.size();
    double seed = (double) rand() / RAND_MAX - 0.5; // -0.5 to 0.5
    int r = rand() % 4;
    if (r == 0) {
        double k = encoding.springCommands[springIndex].k;
        k += min(max(seed * 100, 500), 10000);
        encoding.springCommands[springIndex].k = k;
    } else if (r == 1) {
        double a = encoding.springCommands[springIndex].a;
        a += min(max(seed * 0.1, 0.5), 2);
        encoding.springCommands[springIndex].a = a;
    } else if (r == 2) {
        double b = encoding.springCommands[springIndex].b;
        b += min(max(seed * 0.05, -0.5), 0.5);
        encoding.springCommands[springIndex].b = b;
    } else {
        double c = encoding.springCommands[springIndex].c;
        c += min(max(seed * 0.1, 0), 2 * M_PI);
        encoding.springCommands[springIndex].c = c;
    }
    return encoding;
}

// TODO translate from encoding to masses and springs and run the sim
// TODO make this async? Likely doable by returning an opaque handle
static void OozebotEncoding::evaluate(OozebotEncoding encoding) {
    SimInputs inputs = this.inputsFromEncoding(encoding);
    if (inputs.points.size() == 0) {
        encoding.fitness = 0;
        return encoding;
    }
    encoding.fitness = simulate(&inputs.points, &inputs.springs, inputs.springPresets, 5.0 /*s*/, encoding.globalTimeInterval);
    // TODO move to CUDA and make this async
}

inline bool dominates(OozebotEncoding firstEncoding, OozebotEncoding secondEncoding) {
    return firstEncoding.fitness > secondEncoding.fitness && firstEncoding.age > secondEncoding.age;
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
                std::map<int, std::map<int, int>> innerMap = pointLocationToIndexMap[xi];
                if (innerMap.find(yi) == innerMap.end()) {
                    std::map<int, int> innermostMap;
                    pointLocationToIndexMap[xi][yi] = innermostMap;
                }
                std::map<int, int> innermostMap = pointLocationToIndexMap[xi][yi];
                if (innermostMap.find(zi) != innermostMap.end()) {
                    // It wasn't already there so we add it
                    pointLocationToIndexMap[xi][yi][zi] = points.size();
                    OozebotExpression pointExpression = massCommands[boxCommand.pointIdxs[i]];
                    Point p = {xi / 10.0, yi / 10.0, zi / 10.0, 0, 0, 0, pointExpression.kg, 0, 0, 0};
                    points.push_back(p);
                }
                pointIndices.push_back(pointLocationToIndexMap[xi][yi][zi]);
                i++;
            }
        }
    }
    // now make the springs
    i = 0;
    for (auto it = pointIndices.begin(); it != pointIndices.end(); it++) {
        for (auto iter = it + 1; iter != pointIndices.end(); iter++) {
            // always index from smaller to bigger so we don't have to double bookkeep
            int first = min(*it, *iter);
            int second = max(*it, *iter);
            if (pointIndexHasSpring.find(first) == pointIndexHasSpring.end()) {
                std::map<int, bool> innerMap;
                pointIndexHasSpring[first] = innerMap;
            }
            if (pointIndexHasSpring[first].find(second) == pointIndexHasSpring[first].end()) {
                OozebotExpression springExpression = springCommands[boxCommand.springIdxs[i]];
                Point p1 = points[first];
                Point p2 = points[second];
                double length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                Spring s = {springExpression.k, first, second, length, boxCommand.springIdxs[i]};
                springs.push_back(s);
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
    if (commandIdx >= growthCommands.size) {
        return 100;
    }
    OozebotExpression growthCommand = growthCommands[commandIdx];
    int minY = 100;
    switch growthCommand.expressionType {
        case OozebotExpressionType.symmetryScope:
            int firstMinY = recursiveBuildABot(
                &points,
                &springs,
                &pointLocationToIndexMap,
                &pointIndexHasSpring,
                encoding.growthCommands,
                encoding.massCommands,
                encoding.springCommands,
                encoding.boxCommands,
                encoding.layAndMoveSequences,
                x,
                y,
                z,
                invertX,
                invertY,
                invertZ,
                commandIdx + 1);
            int secondMinY = recursiveBuildABot(
                &points,
                &springs,
                &pointLocationToIndexMap,
                &pointIndexHasSpring,
                encoding.growthCommands,
                encoding.massCommands,
                encoding.springCommands,
                encoding.boxCommands,
                encoding.layAndMoveSequences,
                x,
                y,
                z,
                growthCommand.scopeAxis == OozebotAxis.x ? !invertX : invertX,
                growthCommand.scopeAxis == OozebotAxis.y ? !invertY : invertY,
                growthCommand.scopeAxis == OozebotAxis.z ? !invertZ : invertZ,
                commandIdx + 1);
            return min(firstMinY, secondMinY);
        case OozebotExpressionType.layAndMove:
            std::vector<OozebotExpression> layAndMoveSequence = layAndMoveSequences[growthCommand.layAndMoveIdx];
            for (auto iter = layAndMoveSequence.begin(); iter != layAndMoveSequence.end(); ++iter) {
                OozebotExpression cmd = *it;
                // First we lay the current block
                layBlockAtPosition(x, y, z, &points, &springs, &pointLocationToIndexMap, &pointIndexHasSpring, massCommands, springCommands, boxCommands[cmd.blockIdx]);

                // Now we move
                OozebotDirection direction = cmd.direction;
                switch direction {
                    case OozebotDirection.up:
                        if (invertY == false) {
                            y += 1;
                        } else {
                            y -= 1;
                        }
                        break;
                    case OozebotDirection.down:
                        if (invertY == false) {
                            y -= 1;
                        } else {
                            y += 1;
                        }
                        break;
                    case OozebotDirection.left:
                        if (invertZ == false) {
                            z -= 1;
                        } else {
                            z += 1;
                        }
                        break;
                    case OozebotDirection.right:
                        if (invertZ == false) {
                            z += 1;
                        } else {
                            z -= 1;
                        }
                        break;
                    case OozebotDirection.forward:
                        if (invertX == false) {
                            x += 1;
                        } else {
                            x -= 1;
                        }
                        break;
                    case OozebotDirection.back:
                        if (invertX == false) {
                            x -= 1;
                        } else {
                            x += 1;
                        }
                        break;
                }
                x = max(min(x, 100), -100);
                y = max(min(y, 100), -100);
                z = max(min(z, 100), -100);
                minY = min(minY, y);
            }

            int otherMinY = recursiveBuildABot(
                &points,
                &springs,
                &pointLocationToIndexMap,
                &pointIndexHasSpring,
                encoding.growthCommands,
                encoding.massCommands,
                encoding.springCommands,
                encoding.boxCommands,
                encoding.layAndMoveSequences,
                x,
                y,
                z,
                invertX,
                invertY,
                invertZ,
                commandIdx + 1);
            return min(minY, otherMinY);
        default:
            printf("Unexpected expression\n");
            return -1;
    }
}

static SimInputs OozebotEncoding::inputsFromEncoding(OozebotEncoding encoding) {
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
        &points,
        &springs,
        &pointLocationToIndexMap,
        &pointIndexHasSpring,
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
        points.y -= minY;
    }

    return { points, springs, presets  };
}
