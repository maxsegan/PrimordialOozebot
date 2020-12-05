#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <map>
#include <chrono>

#include "cudaSim.h"

// Usage: nvcc -O2 /std:c++17 cudaSim.cu -o cudaSim -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64"

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void genPointsAndSprings(
	std::vector<Point> &points,
	std::vector<Spring> &springs);

#define staticFriction 0.5
#define kineticFriction 0.3
#define dt 0.0001
#define dampening 0.9995
#define gravity -9.81
#define kSpring 500.0
#define kGround -100000.0
const float kDropHeight = 0.2;
const int pointsPerSide = 2;

__global__ void update_spring(
    Point *points,
    Spring *springs,
    SpringDelta *springDeltas,
    int n,
    double preset0,
    double preset1,
    double preset2,
    double preset3,
    double preset4,
    double preset5) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Spring s = springs[i];
    Point p1 = points[s.p1];
    Point p2 = points[s.p2];

    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;

    float dist = sqrt(dx * dx + dy * dy + dz * dz);

    // negative if repelling, positive if attracting
    float adjust;
    switch (s.flexIndex) {
        case 0:
            adjust = preset0;
            break;
        case 1:
            adjust = preset1;
            break;
        case 2:
            adjust = preset2;
            break;
        case 3:
            adjust = preset3;
            break;
        case 4:
            adjust = preset4;
            break;
        case 5:
            adjust = preset5;
            break;
        default:
            adjust = 1;
            break;
    }
    float f = s.k * (dist - (s.l0 * adjust));

    float fd = f / dist;

    float xd = fd * dx;
    float yd = fd * dy;
    float zd = fd * dz;

    springDeltas[p1.springDeltaIndex + s.p1SpringIndex] = {-xd, -yd, -zd};
    springDeltas[p2.springDeltaIndex + s.p2SpringIndex] = {xd, yd, zd};
}

__global__ void update_point(Point *points, SpringDelta *springDeltas, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Point p = points[i];

	float mass = p.mass;
    float fx = 0;
    float fz = 0;
    float fy = gravity * mass;
    int startIndex = p.springDeltaIndex;
    int done = p.numSprings + startIndex;
    for (int j = startIndex; j < done; j++) {
        SpringDelta sd = springDeltas[j];

		fx += sd.dx;
    	fy += sd.dy;
    	fz += sd.dz;
	}

    float y = p.y;
    float vx = p.vx;
    float vy = p.vy;
    float vz = p.vz;

    if (y <= 0) {
        float fh = sqrt(fx * fx + fz * fz);
        float fyfric = abs(fy * staticFriction);
        if (fh < fyfric) {
            fx = 0;
            fz = 0;
        } else {
            float fykinetic = abs(fy * kineticFriction) / fh;
            fx = fx - fx * fykinetic;
            fz = fz - fz * fykinetic;
        }
        fy += kGround * y;
        p.timestampsContactGround += 1;
    }
    float ax = fx / mass;
    float ay = fy / mass;
    float az = fz / mass;

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

AsyncSimHandle simulate(std::vector<Point> &points, std::vector<Spring> &springs, std::vector<FlexPreset> &presets, double n, double oscillationFrequency, int streamNum) {
    if (points.size() == 0) {
        return { {}, NULL, NULL, NULL};
    }
    std::vector<SpringDelta> pointSprings(springs.size() * 2, {0,0,0});
    int springDeltaIndex  = 0;
    double start = 0;
    for (int i = 0; i < points.size(); i++) {
        points[i].springDeltaIndex = springDeltaIndex;
        springDeltaIndex += points[i].numSprings;
        start += points[i].x;
    }
    start = start / points.size();

    int nDevices;
    int deviceNumber = 0;
    HANDLE_ERROR(cudaGetDeviceCount(&nDevices));
    if (nDevices > 1) {
        deviceNumber = streamNum % nDevices;
        HANDLE_ERROR(cudaSetDevice(deviceNumber));
    }

    Point *p_d;
    Spring *s_d;
    SpringDelta *ps_d;
    HANDLE_ERROR(cudaMalloc(&p_d, points.size() * sizeof(Point)));
    HANDLE_ERROR(cudaMemcpy(p_d, &points[0], points.size() * sizeof(Point), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc(&s_d, springs.size() * sizeof(Spring)));
    HANDLE_ERROR(cudaMemcpy(s_d, &springs[0], springs.size() * sizeof(Spring), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc(&ps_d, pointSprings.size() * sizeof(SpringDelta)));
    HANDLE_ERROR(cudaMemcpy(ps_d, &pointSprings[0], pointSprings.size() * sizeof(SpringDelta), cudaMemcpyHostToDevice));

    double t = 0;
    int numPoints = points.size();
    int numPointThreads = 12;
    int numPointBlocks = numPoints / numPointThreads + 1;
  
    int numSprings = springs.size();
    int numSpringThreads = 25;
    int numSpringBlocks = numSprings / numSpringThreads + 1;

    std::vector<float> pv;
    for (auto it = presets.begin(); it != presets.end(); it++) {
        pv.push_back(0.0);
    }

    //printf("num springs evaluated: %lld, %d\n", long long int(limit / dt * numSprings), numSprings);
    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    while (t < n) {
        for (int i = 0; i < pv.size(); i++) {
            const float a = presets[i].a;
            const float b = presets[i].b;
            const float c = presets[i].c; 
            pv[i] = a + b * sin(t * oscillationFrequency);
        }
        update_spring<<<numSpringBlocks, numSpringThreads>>>(p_d, s_d, ps_d, numSprings, pv[0], pv[1], pv[2], pv[3], pv[4], pv[5]);
        update_point<<<numPointBlocks, numPointThreads>>>(p_d, ps_d, numPoints);
        t += dt;
    }

    //std::vector<Point> newPoints(points.size(), {0, 0, 0, 0, 0, 0, 0, 0, 0});
    return {points, p_d, s_d, ps_d, start, deviceNumber};
 
    //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    //std::cout << "Time difference = " << ms.count() / 1000.0 << "[s]" << std::endl;
}

void resolveSim(AsyncSimHandle &handle) {
    if (handle.points.size() == 0) {
        return;
    }
    HANDLE_ERROR(cudaSetDevice(handle.device));
    HANDLE_ERROR(cudaMemcpy(&handle.points[0], handle.p_d, handle.points.size() * sizeof(Point), cudaMemcpyHostToHost));
    
    HANDLE_ERROR(cudaFree(handle.p_d));
    HANDLE_ERROR(cudaFree(handle.s_d));
    HANDLE_ERROR(cudaFree(handle.ps_d));
}

int mains() {
    std::vector<Point> points;
    std::vector<Spring> springs;
    std::vector<FlexPreset> presets = { {1, 0.0, 0.0} };

    genPointsAndSprings(points, springs);

    AsyncSimHandle handle = simulate(points, springs, presets, 5.0, 0.5, 0);
    resolveSim(handle);

    for (int i = 0; i < 8; i++) {
        printf("x: %f, y: %f, z: %f, %d\n", handle.points[i].x, handle.points[i].y, handle.points[i].z, i);
    }

    return 0;
}

void genPointsAndSprings(
	std::vector<Point> &points,
	std::vector<Spring> &springs) {

    for (int x = 0; x < pointsPerSide; x++) {
        for (int y = 0; y < pointsPerSide; y++) {
            for (int z = 0; z < pointsPerSide; z++) {
                // (0,0,0) or (0.1,0.1,0.1) and all combinations
                float px = x / 10.0;
                float py = y / 10.0 + kDropHeight;
                float pz = z / 10.0;
                Point p = {px, py, pz, 0, 0, 0, 0.1, 0, 0};
                points.push_back(p);
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

                Point p1 = points[p1index];
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

                            Point p2 = points[p2index];
                            float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                            Spring s = {kSpring, p1index, p2index, length, p1.numSprings, p2.numSprings};
                            springs.push_back(s);
                            points[p1index].numSprings += 1;
                            points[p2index].numSprings += 1;
                            p1.numSprings += 1;
                        }
                    }
                }
            }
        }
    }
}