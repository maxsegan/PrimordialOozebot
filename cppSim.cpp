#include <iostream>
#include <math.h>
#include <vector>
#include <map>

struct Point {
  double x; // meters
  double y; // meters
  double z; // meters
  double vx; // meters/second
  double vy; // meters/second
  double vz; // meters/second
  double mass; // kg
  double fx; // N - reset every iteration
  double fy; // N
  double fz; // N
};

struct Spring {
  double k; // N/m
  double p1; // Index of first point
  double p2; // Index of second point
  double l0; // meters
  double currentl; // meters
};

int main() {
    double kSpring = 10000.0;
    double kGround = 100000.0;
    double kOscillationFrequency = 10000;//100000
    double kDropHeight = 0.2;

    std::vector<Point> *points;
    std::vector<Spring> *springs;
    std::map<int, std::map<int, std::map<int, Point>>> *cache;

    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 10; y++) {
            for (int z = 0; z < 10; z++) {
                // (0,0,0) or (0.1,0.1,0.1) and all combinations
                Point p = Point(x / 10.0, kDropHeight + y / 10.0, z / 10.0, 0, 0, 0, 0.1, 0, 0, 0)
                points.push_back(p)
                if (cache.count(x) == 0) {
                    cache[x] = [:]
                }
                if (cache[x].count(y) == 0) {
                    cache[x][y] = [:]
                }
                cache[x][y][z] = p
            }
        }
    }
    printf("Hello Sim\n");
    return 0;
}
