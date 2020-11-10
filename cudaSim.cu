#include<stdio.h>
#include<stdlib.h>
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

void genPointsAndSprings(
	std::vector<Point> &points,
	std::vector<Spring> &springs,
	std::vector<std::vector<Spring>> &pointSprings);

const double staticFriction = 0.5;
const double kineticFriction = 0.3;
const double dt = 0.0000005;
const double dampening = 1 - (dt * 1000);
const double gravity = -9.81;
const double kSpring = 10000.0;
const double kGround = 100000.0;
const double kOscillationFrequency = 0;//10000;//100000
const double kDropHeight = 0.2;
const int pointsPerSide = 10;

__global__ void device_add(int *a, int *b, int *c) {
     c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

/*int main(void) {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);
 
    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size);
    fill_array(a);
    b = (int *)malloc(size);
    fill_array(b);
    c = (int *)malloc(size);
 
    // Alloc space for device copies of vector (a, b, c)
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));
 
    // Copy from host to device
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
 
    device_add<<<1,N>>>(d_a,d_b,d_c);
 
    // Copy result back to host
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
 
    print_output(a,b,c);
    free(a); free(b); free(c);
 
    //free gpu memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
 
    return 0;
}*/

int main() {
    std::vector<Point> points;
    std::vector<Spring> springs;
    std::vector<std::vector<Spring>> pointSprings;

    genPointsAndSprings(points, springs, pointSprings);
    double t = 0;
    // 60 fps - 0.000166
    double limit = 1.0;
  
  	int numSprings = (int)springs.size();
  	printf("%f, %f\n", t, points[0].mass);
    printf("num springs evaluated: %lld\n", long long int(limit / dt * numSprings));
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    while (t < limit) {
        double adjust = 1 + sin(t * kOscillationFrequency) * 0.1;
        for (int i = 0; i < springs.size(); i++) {
            Spring l = springs[i];

            int p1index = l.p1;
            int p2index = l.p2;
            Point p1 = points[p1index];
            Point p2 = points[p2index];

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
            points[p1index].fx -= dx;
            points[p2index].fx += dx;

            double dy = f * (p1y - p2y) / dist;
            points[p1index].fy -= dy;
            points[p2index].fy += dy;

            double dz = f * (p1z - p2z) / dist;
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
            p.fx = 0;
            p.fy = 0;
            p.fz = 0;
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
        t += dt;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time difference = " << ms.count() / 1000.0 << "[s]" << std::endl;
    printf("%f, %f\n", t, points[0].fy);

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
                Point p = {x / 10.0, kDropHeight + y / 10.0, z / 10.0, 0, 0, 0, 0.1, 0, 0, 0};
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
                        }
                    }
                }
            }
        }
    }
}
