#include<stdio.h>
#include<stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <map>
#include <chrono>

// Usage: nvcc -O2 cudaSim.cu -o cudaSim -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64"

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
  double dx; // caching for CUDA ease
  double dy; // caching for CUDA ease
  double dz; // caching for CUDA ease
};

void genPointsAndSprings(
	std::vector<Point> &points,
	std::vector<Spring> &springs,
	std::vector<int> &pointSprings);

#define maxSprings 28
#define staticFriction 0.5
#define kineticFriction 0.3
#define dt 0.0001
#define dampening 1 - (0.0005)
#define gravity -9.81
#define kSpring 500.0
#define kGround 100000.0
const double kOscillationFrequency = 0;
const double kDropHeight = 0.2;
const int pointsPerSide = 45;

__global__ void update_spring(Point *points, Spring *springs, double adjust, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
    	return;
    }

    Point p1 = points[springs[i].p1];
    Point p2 = points[springs[i].p2];

    double p1x = p1.x;
    double p1y = p1.y;
    double p1z = p1.z;
    double p2x = p2.x;
    double p2y = p2.y;
    double p2z = p2.z;
    double dist = sqrt(pow(p1x - p2x, 2) + pow(p1y - p2y, 2) + pow(p1z - p2z, 2));

    // negative if repelling, positive if attracting
    double f = springs[i].k * (dist - (springs[i].l0 * adjust));
    // distribute force across the axes
    double dx = f * (p1x - p2x) / dist;
    double dy = f * (p1y - p2y) / dist;
    double dz = f * (p1z - p2z) / dist;

    springs[i].dx = dx;
    springs[i].dy = dy;
    springs[i].dz = dz;
}

__global__ void update_point(Point *points, Spring *springs, int *pointsToSprings) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    Point p = points[i];
    double mass = p.mass;
    double y = p.y;
    double vx = p.vx;
    double vy = p.vy;
    double vz = p.vz;
    double fy = gravity * mass;
    double fx = 0;
    double fz = 0;
    for (int j = 0; j < p.numSprings; j++) {
    	int springIndex = pointsToSprings[i * maxSprings + j];
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

    if (y <= 0) {
        double fh = sqrt(pow(fx, 2) + pow(fz, 2));
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
    // reset the force cache
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

int main() {
    std::vector<Point> points;
    std::vector<Spring> springs;
    std::vector<int> pointSprings(pointsPerSide * pointsPerSide * pointsPerSide * maxSprings, 0);

    genPointsAndSprings(points, springs, pointSprings);

    Point *p_d;
    Spring *s_d;
    int *ps_d;
    cudaMalloc(&p_d, points.size() * sizeof(Point));
    cudaMemcpy(p_d, &points[0], points.size() * sizeof(Point), cudaMemcpyHostToDevice);

    cudaMalloc(&s_d, springs.size() * sizeof(Spring));
    cudaMemcpy(s_d, &springs[0], springs.size() * sizeof(Spring), cudaMemcpyHostToDevice);

    cudaMalloc(&ps_d, pointSprings.size() * sizeof(int));
    cudaMemcpy(ps_d, &pointSprings[0], pointSprings.size() * sizeof(int),  cudaMemcpyHostToDevice);

    double t = 0;
    // 60 fps - 0.000166
    double limit = 1;
    int ppsSquare = pointsPerSide * pointsPerSide;
  
  	int numSprings = (int)springs.size();

    int springThreads = 100;
    int springBlocks = (int)ceil(numSprings / 100.0);
    printf("num springs evaluated: %lld\n", long long int(limit / dt * numSprings));
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    while (t < limit) {
        double adjust = 1 + sin(t * kOscillationFrequency) * 0.1;
        
        update_spring<<<springBlocks, springThreads>>>(p_d, s_d, adjust, numSprings);
        update_point<<<ppsSquare, pointsPerSide>>>(p_d, s_d, ps_d);
        t += dt;
    }

    cudaDeviceSynchronize();
 

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time difference = " << ms.count() / 1000.0 << "[s]" << std::endl;

    Point *ps = (Point *)malloc(points.size() * sizeof(Point));
    cudaMemcpy(ps, p_d, points.size() * sizeof(Point), cudaMemcpyDeviceToHost);
    for (int i = 0; i < points.size(); i++) {
    	printf("x: %f, y: %f, z: %f, %d\n", ps[i].x, ps[i].y, ps[i].z, i);
    	break;
    }
    
    cudaFree(p_d);
    cudaFree(s_d);
    cudaFree(ps_d);
    free(ps);

    return 0;
}

void genPointsAndSprings(
	std::vector<Point> &points,
	std::vector<Spring> &springs,
	std::vector<int> &pointSprings) {
    std::map<int, std::map<int, std::map<int, Point>>> cache;

    for (int x = 0; x < pointsPerSide; x++) {
        for (int y = 0; y < pointsPerSide; y++) {
            for (int z = 0; z < pointsPerSide; z++) {
                // (0,0,0) or (0.1,0.1,0.1) and all combinations
                Point p = {x / 10.0, kDropHeight + y / 10.0, z / 10.0, 0, 0, 0, 0.1, 0};
                points.push_back(p);
                if (cache.count(x) == 0) {
                    cache[x] = {};
                }
                if (cache[x].count(y) == 0) {
                    cache[x][y] = {};
                }
                cache[x][y][z] = p;
            }
        }
    }
    std::map<int, std::vector<int>> connected;
    double ppsSquare = pointsPerSide * pointsPerSide;
    connected[0] = {};
    // Create the springs
    for (int x = 0; x < pointsPerSide; x++) {
        for (int y = 0; y < pointsPerSide; y++) {
            for (int z = 0; z < pointsPerSide; z++) {
                int p1index = z + pointsPerSide * y + ppsSquare * x;

                Point p1 = cache[x][y][z];
                for (int x1 = x - 1; x1 < x + 2; x1++) {
                    if (x1 == pointsPerSide || x1 < 0) {
                        continue;
                    }
                    for (int y1 = y - 1; y1 < y + 2; y1++) {
                        if (y1 == pointsPerSide || y1 < 0) {
                            continue;
                        }
                        for (int z1 = z - 1; z1 < z + 2; z1++) {
                            if (z1 == pointsPerSide || z1 < 0 || (x1 == x && y1 == y && z1 == z)) {
                                continue;
                            }
                            int p2index = z1 + pointsPerSide * y1 + ppsSquare * x1;
                            if (connected.find(p2index) == connected.end()) {
                                connected[p2index] = {};
                            }
                            if (std::find(connected[p1index].begin(), connected[p1index].end(), p2index) != connected[p1index].end()) {
                                continue;
                            }
                            connected[p1index].push_back(p2index);
                            connected[p2index].push_back(p1index);

                            Point p2 = cache[x1][y1][z1];
                            double length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                            Spring s = {kSpring, p1index, p2index, length, 0, 0, 0};
                            int springIndex = springs.size();
                            springs.push_back(s);
                            int ppsIndex1 = p1index * maxSprings + p1.numSprings;
                            int ppsIndex2 = p2index * maxSprings + p2.numSprings;
                            pointSprings[ppsIndex1] = springIndex;
                            pointSprings[ppsIndex2] = springIndex;
                            points[p1index].numSprings += 1;
                            points[p2index].numSprings += 1;
                        }
                    }
                }
            }
        }
    }
}