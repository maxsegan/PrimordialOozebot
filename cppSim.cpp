#include <algorithm>
#include <iostream>
#include <math.h>
#include <vector>
#include <map>
#include <chrono>

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
  int p1; // Index of first point
  int p2; // Index of second point
  double l0; // meters
};

const double kSpring = 500.0;
const double kGround = 100000.0;
const double kOscillationFrequency = 0;//10000;//100000
const double kDropHeight = 0;
const int kNumPerSide = 10;
const double staticFriction = 0.5;
const double kineticFriction = 0.3;
const double dt = 0.0001;
const double dampening = 1 - (dt * 5);
const double gravity = -9.81;

int main() {
    std::vector<Point> points;
    std::vector<Spring> springs;

    for (int x = 0; x < kNumPerSide; x++) {
        for (int y = 0; y < kNumPerSide; y++) {
            for (int z = 0; z < kNumPerSide; z++) {
                // (0,0,0) or (0.1,0.1,0.1) and all combinations
                Point p = {x / 10.0, kDropHeight + y / 10.0, z / 10.0, 0, 0, 0, 0.1, 0, 0, 0};
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
                            Spring s = {kSpring, p1index, p2index, length};
                            springs.push_back(s);
                        }
                    }
                }
            }
        }
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
        for (int i = 0; i < springs.size(); i++) {
            Spring l = springs[i];

            int p1index = l.p1;
            int p2index = l.p2;
            Point p1 = points[p1index];
            Point p2 = points[p2index];

            double xd = p1.x - p2.x;
            double yd = p1.y - p2.y;
            double zd = p1.z - p2.z;
            double dist = sqrt(xd * xd + yd * yd + zd * zd);

            // negative if repelling, positive if attracting
            double f = l.k * (dist - (l.l0 * adjust));
            double fd = f / dist;
            // distribute force across the axes
            double dx = xd * fd;
            double dy = yd * fd;
            double dz = zd * fd;

            points[p1index].fx -= dx;
            points[p2index].fx += dx;

            points[p1index].fy -= dy;
            points[p2index].fy += dy;

            points[p1index].fz -= dz;
            points[p2index].fz += dz;
        }
        for (int i = 0; i < points.size(); i++) {
            Point p = points[i];
        
            double mass = p.mass;
            double fy = p.fy + gravity * mass;
            double fx = p.fx;
            double fz = p.fz;
            double y = p.y;
            double vx = p.vx;
            double vy = p.vy;
            double vz = p.vz;

            if (y <= 0) {
                double fh = sqrt(pow(fx, 2) + pow(fz, 2));
                double fyfric = abs(fy * staticFriction);
                if (fh < fyfric) {
                    fx = 0;
                    fz = 0;
                } else {
                    double fykinetic = abs(fy * kineticFriction) * fh;
                    fx = fx - fx / fykinetic;
                    fz = fz - fz / fykinetic;
                }
                fy += -kGround * y;
            }
            double ax = fx / mass;
            double ay = fy / mass;
            double az = fz / mass;
            // reset the force cache
            p.fx = 0;
            p.fy = 0;
            p.fz = 0;
            vx = (ax * dt + vx) * dampening;
            p.vx = vx;
            vy = (ay * dt + vy) * dampening;
            p.vy = vy;
            vz = (az * dt + vz) * dampening;
            p.vz = vz;
            p.x += vx * dt;
            p.y += vy * dt;
            p.z += vz * dt;
            points[i] = p;
        }
        t += dt;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time difference = " << ms.count() / 1000.0 << "[s]" << std::endl;
    //for (int i = 0; i < points.size(); i++) {
    //    printf("p[%d].x = %f, y = %f, z = %f\n", i, points[i].x, points[i].y, points[i].z);
    //}

    return 0;
}
