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
	std::vector<std::vector<Spring>> &pointSprings);

#define staticFriction 0.5
#define kineticFriction 0.3
#define dt 0.0000005
#define dampening 1 - (0.0000005 * 1000)
#define gravity -9.81
#define kSpring 10000.0
#define kGround 100000.0
const double kOscillationFrequency = 0;
const double kDropHeight = 0.2;
const int pointsPerSide = 20;

__global__ void update_spring(Point *points, Spring *springs, double adjust) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    Point p1 = points[s[i].p1];
    Point p2 = points[s[i].p2];

    double p1x = p1.x;
    double p1y = p1.y;
    double p1z = p1.z;
    double p2x = p2.x;
    double p2y = p2.y;
    double p2z = p2.z;
    double dist = sqrt(pow(p1x - p2x, 2) + pow(p1y - p2y, 2) + pow(p1z - p2z, 2));

    // negative if repelling, positive if attracting
    double f = l.k * (dist - (l.l0 * adjust));
    // distribute force across the axes
    double dx = f * (p1x - p2x) / dist;
    double dy = f * (p1y - p2y) / dist;
    double dz = f * (p1z - p2z) / dist;

    springs
    s[i].dx = dx;
    s[i].dy = dy;
    s[i].dz = dz;
}

__global__ void update_point(Point *points, Spring *springs, int **pointsToSprings) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d, %d, %d, %d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

    Point p = points[i];
    double fy = gravity * mass;
    double fx = 0;
    double fz = 0;
    int *pToS = pointSprings[i]

    for (int j = 0; j < p.numSprings; j++) {
        Spring s = pToS[j];
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
        
    double mass = p.mass;
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
    p.x += vx;
    p.y += vy;
    p.z += vz;
    points[i] = p;
}

int main() {
    std::vector<Point> points;
    std::vector<Spring> springs;
    std::vector<std::vector<Spring>> pointSprings;

    genPointsAndSprings(points, springs, pointSprings);
    Point *p_d;
    Spring *p_d;
    int **ps_d;
    std::vector<int *> psd;
    cudaMalloc(&p_d, points.size() * sizeof(Point));
    cudaMemcpy(p_d, &points[0], points.size() * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMalloc(&p_d, springs.size() * sizeof(Spring));
    cudaMemcpy(p_d, &springs[0], points.size() * sizeof(Spring), cudaMemcpyHostToDevice);
    cudaMalloc(&ps_d, points.size() * sizeof(int *));
    for (int i = 0; i < points.size(); i++) {
    	Spring *s_d;
    	psd.push_back(s_d);
    	cudaMalloc(&s_d, pointSprings[i].size() * sizeof(int));
    	cudaMemcpy(s_d, &pointSprings[i], pointSprings[i].size() * sizeof(int), cudaMemcpyHostToDevice);
    }

    double t = 0;
    // 60 fps - 0.000166
    double limit = 0.1;
    int ppsSquare = pointsPerSide * pointsPerSide;
  
  	int numSprings = (int)springs.size();

    if (numSprings % 1000 != 0) {
        pringf("Whoa, issue with num springs\n");
    }
    int springThreads = 1000;
    int springBlocks = numSprings / 1000;
    printf("num springs evaluated: %lld\n", long long int(limit / dt * numSprings));
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    while (t < limit) {
        double adjust = 1 + sin(t * kOscillationFrequency) * 0.1;
        
        update_spring<<<ppsSquare, pointsPerSide>>>(p_d, s_d, adjust);
        cudaDeviceSynchronize();
        update_point<<<springBlocks, springThreads>>>(p_d, p_d, ps_d);
        cudaDeviceSynchronize();
 
        t += dt;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time difference = " << ms.count() / 1000.0 << "[s]" << std::endl;

    Point *ps = (Point *)malloc(points.size() * sizeof(Point));
    cudaMemcpy(ps, p_d, points.size() * sizeof(Point), cudaMemcpyDeviceToHost);
    
    cudaFree(p_d);
    for (int i = 0; i < points.size(); i++) {
    	cudaFree(psd[i]);
    }
    cudaFree(ps_d);
    free(ps);

    return 0;
}

void genPointsAndSprings(
	std::vector<Point> &points,
	std::vector<Spring> &springs,
	std::vector<std::vector<Spring>> &pointSprings) {
    std::map<int, std::map<int, std::map<int, Point>>> cache;

    for (int x = 0; x < pointsPerSide; x++) {
        for (int y = 0; y < pointsPerSide; y++) {
            for (int z = 0; z < pointsPerSide; z++) {
                // (0,0,0) or (0.1,0.1,0.1) and all combinations
                Point p = {x / 10.0, kDropHeight + y / 10.0, z / 10.0, 0, 0, 0, 0.1, 0, 0, 0, 0};
                points.push_back(p);
                if (cache.count(x) == 0) {
                    cache[x] = {};
                }
                if (cache[x].count(y) == 0) {
                    cache[x][y] = {};
                }
                cache[x][y][z] = p;
                pointSprings.push_back({});
            }
        }
    }
    // Create the springs
    for (int x = 0; x < pointsPerSide; x++) {
        for (int y = 0; y < pointsPerSide; y++) {
            for (int z = 0; z < pointsPerSide; z++) {
                Point p1 = cache[x][y][z];
                int p1index = z + pointsPerSide * y + pointsPerSide * pointsPerSide * x;
                for (int x1 = x; x1 < x + 2; x1++) {
                    if (x1 == pointsPerSide) {
                        continue;
                    }
                    for (int y1 = y; y1 < y + 2; y1++) {
                        if (y1 == pointsPerSide) {
                            continue;
                        }
                        for (int z1 = z; z1 < z + 2; z1++) {
                            if (z1 == pointsPerSide || (x1 == x && y1 == y && z1 == z)) {
                                continue;
                            }
                            Point p2 = cache[x1][y1][z1];
                            int p2index = z1 + pointsPerSide * y1 + pointsPerSide * pointsPerSide * x1;
                            double length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                            Spring s = {kSpring, p1index, p2index, length};
                            springs.push_back(s);
                            pointSprings[p1index].push_back(s);
							pointSprings[p2index].push_back(s);
							p2.numSprings += 1;
							p1.numSprings += 1;
                        }
                    }
                }
            }
        }
    }
}
