#include <algorithm>
#include <iostream>
#include <math.h>
#include <vector>
#include <map>
#include <chrono>
#include <thread>
#include <execution>

struct Point {
  double x; // meters
  double y; // meters
  double z; // meters
  double vx; // meters/second
  double vy; // meters/second
  double vz; // meters/second
  double mass; // kg
  int numSprings; // Int - hack for CUDA ease
};

struct Spring {
  double k; // N/m
  int p1; // Index of first point
  int p2; // Index of second point
  double l0; // meters
  double dx; // N - reset every iteration
  double dy; // N
  double dz; // N
};

const int kMaxSprings = 28;
const double kSpring = 500.0;
const double kGround = 100000.0;
const double kOscillationFrequency = 0;//10000;//100000
const double kDropHeight = 0.2;
const int kNumPerSide = 10;
const double staticFriction = 0.5;
const double kineticFriction = 0.3;
const double dt = 0.0001;
const double dampening = 1 - (dt * 5);
const double gravity = -9.81;

void updateSprings(std::vector<Point> &points, std::vector<Spring> &springs, double adjust, int start, int end) {
    for (int i = start; i < end; i++) {
        Spring s = springs[i];
        Point p1 = points[s.p1];
        Point p2 = points[s.p2];

        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        double dz = p1.z - p2.z;

        double dist = sqrt(dx * dx + dy * dy + dz * dz);

        // negative if repelling, positive if attracting
        double f = s.k * (dist - (s.l0 * adjust));

        double fd = f / dist;

        springs[i].dx = fd * dx;
        springs[i].dy = fd * dy;
        springs[i].dz = fd * dz;
    }
}

void updatePoints(std::vector<Point> &points, std::vector<Spring> &springs, std::vector<int> &pointsToSprings) {
    for (int i = 0; i < points.size(); i++) {
        Point p = points[i];
        int numSprings = p.numSprings;

        double mass = p.mass;
        double fx = 0;
        double fz = 0;
        double fy = gravity * mass;
        for (int j = 0; j < numSprings; j++) {
            int springIndex = pointsToSprings[i * kMaxSprings + j];
            Spring s = springs[springIndex];

            if (s.p1 == i) {
                fx -= s.dx;
                fy -= s.dy;
                fz -= s.dz;
            } else {
                fx += s.dx;
                fy += s.dy;
                fz += s.dz;
            }
        }
        double y = p.y;
        double vx = p.vx;
        double vy = p.vy;
        double vz = p.vz;

        if (y <= 0) {
            double fh = sqrt(fx * fx + fz * fz);
            double fyfric = abs(fy * staticFriction);
            if (fh < fyfric) {
                fx = 0;
                fz = 0;
            } else {
                double fykinetic = abs(fy * kineticFriction);
                fx = fx - fx / fh * fykinetic;
                fz = fz - fz / fh * fykinetic;
            }
            fy += -kGround * y;
        }
        double ax = fx / mass;
        double ay = fy / mass;
        double az = fz / mass;

        vx = (ax * dt + vx) * dampening;
        vy = (ay * dt + vy) * dampening;
        vz = (az * dt + vz) * dampening;
        p.vx = vx;
        p.vy = vy;
        p.vz = vz;
        p.x += vx * dt;
        p.y += vy * dt;
        p.z += vz * dt;
        points[i] = p;
    }
}

int main() {
    std::vector<Point> points;
    std::vector<Spring> springs;
    std::vector<int> pointSprings(kNumPerSide * kNumPerSide * kNumPerSide * kMaxSprings, 0);

    for (int x = 0; x < kNumPerSide; x++) {
        for (int y = 0; y < kNumPerSide; y++) {
            for (int z = 0; z < kNumPerSide; z++) {
                // (0,0,0) or (0.1,0.1,0.1) and all combinations
                Point p = {x / 10.0, kDropHeight + y / 10.0, z / 10.0, 0, 0, 0, 0.1, 0};
                points.push_back(p);
            }
        }
    }
    std::map<int, std::vector<int>> connected;
    connected[0] = {};
    // Create the springs
    for (int x = 0; x < kNumPerSide; x++) {
        for (int y = 0; y < kNumPerSide; y++) {
            for (int z = 0; z < kNumPerSide; z++) {
                int p1index = z + kNumPerSide * y + kNumPerSide * kNumPerSide * x;

                Point p1 = points[p1index];
                for (int x1 = x - 1; x1 < x + 2; x1++) {
                    if (x1 == kNumPerSide || x1 < 0) {
                        continue;
                    }
                    for (int y1 = y - 1; y1 < y + 2; y1++) {
                        if (y1 == kNumPerSide || y1 < 0) {
                            continue;
                        }
                        for (int z1 = z - 1; z1 < z + 2; z1++) {
                            if (z1 == kNumPerSide || z1 < 0 || (x1 == x && y1 == y && z1 == z)) {
                                continue;
                            }
                            int p2index = z1 + kNumPerSide * y1 + kNumPerSide * kNumPerSide * x1;
                            if (connected.find(p2index) == connected.end()) {
                                connected[p2index] = {};
                            }
                            if (std::find(connected[p1index].begin(), connected[p1index].end(), p2index) != connected[p1index].end()) {
                                continue;
                            }
                            connected[p1index].push_back(p2index);
                            connected[p2index].push_back(p1index);

                            Point p2 = points[p2index];
                            double length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                            Spring s = {kSpring, p1index, p2index, length, 0, 0, 0};
                            int springIndex = springs.size();
                            springs.push_back(s);
                            int ppsIndex1 = p1index * kMaxSprings + p1.numSprings;
                            int ppsIndex2 = p2index * kMaxSprings + p2.numSprings;
                            pointSprings[ppsIndex1] = springIndex;
                            pointSprings[ppsIndex2] = springIndex;
                            points[p1index].numSprings += 1;
                            points[p2index].numSprings += 1;
                            p1.numSprings += 1; // this is a reference so also increment here
                        }
                    }
                }
            }
        }
    }

    const size_t nthreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(nthreads);
  
    std::vector<std::pair<int, int>> springSplits = {};
    for (int i = 0; i < nthreads; i++) {
        springSplits.push_back(std::make_pair(i * springs.size() / nthreads, (i + 1) * springs.size() / nthreads));
    }

    std::vector<std::pair<int, int>> pointSplits = {};
    for (int i = 0; i < nthreads; i++) {
        pointSplits.push_back(std::make_pair(i * points.size() / nthreads, (i + 1) * points.size() / nthreads));
    }
  
    // 60 fps - 0.000166
    const double limit = 5;
    double t = 0;
    long long int numSprings = springs.size();
    long long int y = (long long int)(limit / dt * numSprings);
    printf("num springs evaluated: %lld\n", y);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    while (t < limit) {
        double adjust = 1 + sin(t * kOscillationFrequency) * 0.1;
        std::for_each(std::execution::par, std::begin(springSplits), std::end(springSplits), [&](auto pair) {
            updateSprings(points, springs, adjust, pair.first, pair.second);
        });
        updatePoints(points, springs, pointSprings);
        t += dt;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time difference = " << ms.count() / 1000.0 << "[s]" << std::endl;
    printf("p[0].y = %f, x = %f, z = %f\n", points[0].y, points[0].x, points[0].z);

    return 0;
}
