#define _USE_MATH_DEFINES
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <map>
#include <chrono>
#include <limits>

#include "cudaSim.h"

// Usage: nvcc -O2 cudaSim.cu -o cudaSim -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64"

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

#define staticFriction 0.5
#define kineticFriction 0.3
#define dt 0.0001
#define dampening 0.9995
#define gravity -9.81
#define kGround -100000.0

__global__ void update_spring(
    Point *points,
    Spring *springs,
    SpringDelta *springDeltas,
    int n,
    int *invalid,
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
    if (dist > (s.l0 * 6)) {
        bool firstInvalidation = atomicCAS(invalid, 0, 1);
        return;
    }

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

AsyncSimHandle createSimHandle(int i) {
    Point *p_d;
    Spring *s_d;
    SpringDelta *ps_d;
    int *b_d;
    cudaStream_t stream;

    int nDevices;
    int deviceNumber = 0;
    
    HANDLE_ERROR(cudaGetDeviceCount(&nDevices));
    if (nDevices > 1) {
    //    deviceNumber = i % nDevices;
        HANDLE_ERROR(cudaSetDevice(deviceNumber));
    }

    HANDLE_ERROR(cudaStreamCreate(&stream));
    int *invalid_h;
    Point *start_p;
    Point *end_p;

    int numPoints = 50000;
    int numSprings = 1000000;
    int numSpringDeltas = 2 * numSprings;

    HANDLE_ERROR(cudaMalloc(&p_d, numPoints * sizeof(Point)));
    HANDLE_ERROR(cudaMalloc(&s_d, numSprings * sizeof(Spring)));
    HANDLE_ERROR(cudaMalloc(&ps_d, numSpringDeltas * sizeof(SpringDelta)));
    HANDLE_ERROR(cudaMalloc(&b_d, sizeof(int)));

    HANDLE_ERROR(cudaMallocHost(&invalid_h, sizeof(int)));
    HANDLE_ERROR(cudaMallocHost(&start_p, numPoints * sizeof(Point)));
    HANDLE_ERROR(cudaMallocHost(&end_p, numPoints * sizeof(Point)));

    return {end_p, start_p, 0, p_d, numPoints, s_d, numSprings, ps_d, numSpringDeltas, b_d, invalid_h, 0, 0, 0, deviceNumber, stream};
}

void releaseSimHandle(AsyncSimHandle &handle) {
    HANDLE_ERROR(cudaFree(handle.p_d));
    HANDLE_ERROR(cudaFree(handle.s_d));
    HANDLE_ERROR(cudaFree(handle.ps_d));
    HANDLE_ERROR(cudaFree(handle.b_d));
    HANDLE_ERROR(cudaFree(handle.invalid_h));
    HANDLE_ERROR(cudaStreamDestroy(handle.stream));
}

void simulate(AsyncSimHandle &handle, std::vector<Point> &points, std::vector<Spring> &springs, std::vector<FlexPreset> &presets, double n, double oscillationFrequency) {
    std::vector<SpringDelta> pointSprings(springs.size() * 2, {0, 0, 0});
    int springDeltaIndex  = 0;
    for (int i = 0; i < points.size(); i++) {
        points[i].springDeltaIndex = springDeltaIndex;
        springDeltaIndex += points[i].numSprings;
    }
    HANDLE_ERROR(cudaSetDevice(handle.device));

    if (points.size() > handle.pointsLength) {
        printf("Resizing points\n");
        handle.pointsLength = points.size() * 2;
        HANDLE_ERROR(cudaFree(handle.p_d));
        HANDLE_ERROR(cudaFree(handle.startPoints));
        HANDLE_ERROR(cudaFree(handle.endPoints));
        HANDLE_ERROR(cudaMalloc(&handle.p_d, 2 * points.size() * sizeof(Point)));
        HANDLE_ERROR(cudaMallocHost(&handle.startPoints, 2 * points.size() * sizeof(Point)));
        HANDLE_ERROR(cudaMallocHost(&handle.endPoints, 2 * points.size() * sizeof(Point)));
    }
    if (springs.size() > handle.springsLength) {
        printf("Resizing springs\n");
        handle.springsLength = springs.size() * 2;
        handle.springDeltaLength = pointSprings.size() * 2;
        HANDLE_ERROR(cudaFree(handle.s_d));
        HANDLE_ERROR(cudaFree(handle.ps_d));
        HANDLE_ERROR(cudaMalloc(&handle.s_d, 2 * springs.size() * sizeof(Spring)));
        HANDLE_ERROR(cudaMalloc(&handle.ps_d, 2 * pointSprings.size() * sizeof(SpringDelta)));
    }

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

    HANDLE_ERROR(cudaMemcpyAsync(handle.p_d, &points[0], numPoints * sizeof(Point), cudaMemcpyHostToDevice, handle.stream));
    HANDLE_ERROR(cudaMemcpyAsync(handle.s_d, &springs[0], numSprings * sizeof(Spring), cudaMemcpyHostToDevice, handle.stream));
    HANDLE_ERROR(cudaMemcpyAsync(handle.ps_d, &pointSprings[0], pointSprings.size() * sizeof(SpringDelta), cudaMemcpyHostToDevice, handle.stream));
    HANDLE_ERROR(cudaMemsetAsync(handle.b_d, 0, sizeof(int), handle.stream));

    while (t < n) {
        for (int i = 0; i < pv.size(); i++) {
            const float a = presets[i].a;
            const float b = presets[i].b;
            const float c = presets[i].c; 
            pv[i] = a * (1 + b * sin(t * oscillationFrequency));
        }
        update_spring<<<numSpringBlocks, numSpringThreads, 0, handle.stream>>>(handle.p_d, handle.s_d, handle.ps_d, numSprings, handle.b_d, pv[0], pv[1], pv[2], pv[3], pv[4], pv[5]);
        update_point<<<numPointBlocks, numPointThreads, 0, handle.stream>>>(handle.p_d, handle.ps_d, numPoints);
        if (t < 1.0 && t + dt >= 1.0) {
            HANDLE_ERROR(cudaMemcpyAsync(handle.startPoints, handle.p_d, numPoints * sizeof(Point), cudaMemcpyDeviceToHost, handle.stream));
            int numCycles = 1;
            double oscillationDuration = 2 * M_PI / oscillationFrequency;
            while ((oscillationDuration * numCycles + t) < n) {
                numCycles += 1;
            }
            n = (oscillationDuration * numCycles) + t;
        }
        t += dt;
    }
    handle.duration = t - 1.0;
    handle.numPoints = numPoints;
    HANDLE_ERROR(cudaMemcpyAsync(handle.endPoints, handle.p_d, numPoints * sizeof(Point), cudaMemcpyDeviceToHost, handle.stream));
    HANDLE_ERROR(cudaMemcpyAsync(handle.invalid_h, handle.b_d, sizeof(int), cudaMemcpyDeviceToHost, handle.stream));
}

void synchronize(AsyncSimHandle &handle) {
    HANDLE_ERROR(cudaSetDevice(handle.device));
    HANDLE_ERROR(cudaStreamSynchronize(handle.stream));
    if ((*(handle.invalid_h)) != 0) {
        printf("Invalidated sim\n");
        handle.duration = std::numeric_limits<double>::infinity();
    }
}

void simulateAgain(AsyncSimHandle &handle, std::vector<FlexPreset> &presets, double t, double n, double oscillationFrequency) {
    int numPoints = handle.numPoints;
    int numPointThreads = 12;
    int numPointBlocks = numPoints / numPointThreads + 1;
  
    int numSpringThreads = 25;
    int numSpringBlocks = handle.numSprings / numSpringThreads + 1;

    std::vector<float> pv;
    for (auto it = presets.begin(); it != presets.end(); it++) {
        pv.push_back(0.0);
    }

    HANDLE_ERROR(cudaSetDevice(handle.device));
    while (t < n) {
        for (int i = 0; i < pv.size(); i++) {
            const float a = presets[i].a;
            const float b = presets[i].b;
            const float c = presets[i].c; 
            pv[i] = a + b * sin(t * oscillationFrequency);
        }
        update_spring<<<numSpringBlocks, numSpringThreads, 0, handle.stream>>>(handle.p_d, handle.s_d, handle.ps_d, handle.numSprings, handle.b_d, pv[0], pv[1], pv[2], pv[3], pv[4], pv[5]);
        update_point<<<numPointBlocks, numPointThreads, 0, handle.stream>>>(handle.p_d, handle.ps_d, numPoints);
        t += dt;
    }
}
