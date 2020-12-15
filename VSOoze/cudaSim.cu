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
#include <cuda_runtime.h>

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

#define dt 0.0001
#define dampening 0.999
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
    double preset3) {
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
        float fyfric = abs(fy * p.us);
        if (fh < fyfric) {
            fx = 0;
            fz = 0;
        } else {
            float fykinetic = abs(fy * p.uk) / fh;
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

AsyncSimHandle createSimHandle(int i, int numPoints, int numSprings) {
    Point *p_d;
    Spring *s_d;
    SpringDelta *ps_d;
    int *b_d;

    int nDevices;
    int deviceNumber = 0;
    
    HANDLE_ERROR(cudaGetDeviceCount(&nDevices));
    if (nDevices > 1) {
        deviceNumber = i % nDevices;
        HANDLE_ERROR(cudaSetDevice(deviceNumber));
    }

    HANDLE_ERROR(cudaMalloc(&p_d, numPoints * sizeof(Point)));
    HANDLE_ERROR(cudaMalloc(&s_d, numSprings * sizeof(Spring)));
    HANDLE_ERROR(cudaMalloc(&ps_d, numSprings * 2 * sizeof(SpringDelta)));
    HANDLE_ERROR(cudaMalloc(&b_d, sizeof(int)));


    int *invalid_h = (int *) malloc(sizeof(int));
    Point *start_p = (Point *) malloc(numPoints * sizeof(Point));

    return {NULL, start_p, p_d, s_d, ps_d, b_d, numPoints, numSprings, invalid_h, 0, deviceNumber};
}

void releaseSimHandle(AsyncSimHandle &handle) {
    HANDLE_ERROR(cudaFree(handle.p_d));
    HANDLE_ERROR(cudaFree(handle.s_d));
    HANDLE_ERROR(cudaFree(handle.ps_d));
    HANDLE_ERROR(cudaFree(handle.b_d));
    free(handle.invalid_h);
    free(handle.startPoints);
}

void simulate(AsyncSimHandle &handle, std::vector<Point> &points, std::vector<Spring> &springs, std::vector<FlexPreset> &presets, double n, double oscillationFrequency) {
    int psSize = springs.size() * 2;
    int springDeltaIndex = 0;
    for (int i = 0; i < points.size(); i++) {
        points[i].springDeltaIndex = springDeltaIndex;
        springDeltaIndex += points[i].numSprings;
    }
    HANDLE_ERROR(cudaSetDevice(handle.device));

    double t = 0;
    int numPointThreads = 12;
    int numPointBlocks = handle.numPoints / numPointThreads + 1;
  
    int numSpringThreads = 25;
    int numSpringBlocks = handle.numSprings / numSpringThreads + 1;

    std::vector<float> pv;
    for (auto it = presets.begin(); it != presets.end(); it++) {
        pv.push_back(0.0);
    }
    HANDLE_ERROR(cudaMemcpyAsync(handle.p_d, &points[0], handle.numPoints * sizeof(Point), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyAsync(handle.s_d, &springs[0], handle.numSprings * sizeof(Spring), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemsetAsync(handle.b_d, 0, sizeof(int)));

    while (t < n) {
        for (int i = 0; i < pv.size(); i++) {
            const float a = presets[i].a;
            const float b = presets[i].b;
            const float c = presets[i].c; 
            pv[i] = (float) (a * (1 + b * sin(t * oscillationFrequency + c)));
        }
        update_spring<<<numSpringBlocks, numSpringThreads>>>(handle.p_d, handle.s_d, handle.ps_d, handle.numSprings, handle.b_d, pv[0], pv[1], pv[2], pv[3]);
        update_point<<<numPointBlocks, numPointThreads>>>(handle.p_d, handle.ps_d, handle.numPoints);
        if (t < 1.0 && t + dt >= 1.0) {
            HANDLE_ERROR(cudaMemcpy(handle.startPoints, handle.p_d, handle.numPoints * sizeof(Point), cudaMemcpyDeviceToHost));
        }
        t += dt;
    }
    handle.duration = t - 1.0;
    handle.endPoints = &points[0];
    HANDLE_ERROR(cudaMemcpy(handle.endPoints, handle.p_d, handle.numPoints * sizeof(Point), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(handle.invalid_h, handle.b_d, sizeof(int), cudaMemcpyDeviceToHost));
    if ((*(handle.invalid_h)) != 0) {
        printf("Invalidated sim\n");
        handle.duration = std::numeric_limits<double>::infinity();
    }
}

void simulateAgain(AsyncSimHandle &handle, std::vector<FlexPreset> &presets, double t, double n, double oscillationFrequency) {
    int numPoints = handle.numPoints;
    int numPointThreads = 120;
    int numPointBlocks = numPoints / numPointThreads + 1;
  
    int numSpringThreads = 250;
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
            pv[i] = (float) (a * (1 + b * sin(t * oscillationFrequency + c)));
        }
        update_spring<<<numSpringBlocks, numSpringThreads>>>(handle.p_d, handle.s_d, handle.ps_d, handle.numSprings, handle.b_d, pv[0], pv[1], pv[2], pv[3]);
        update_point<<<numPointBlocks, numPointThreads>>>(handle.p_d, handle.ps_d, numPoints);
        t += dt;
    }
    HANDLE_ERROR(cudaMemcpy(handle.endPoints, handle.p_d, handle.numPoints * sizeof(Point), cudaMemcpyDeviceToHost));
}
