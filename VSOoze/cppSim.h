#ifndef CPP_SIM_H
#define CPP_SIM_H

#include <vector>
#include "cudaSim.h"
/*
struct Point {
    float x; // meters
    float y; // meters
    float z; // meters
    float vx; // meters/second
    float vy; // meters/second
    float vz; // meters/second
    float mass; // kg
  float uk; // kinetic friction coefficient
  float us; // static friction coefficient
  float fx; // N - internal bookkeeping
  float fy; // N
  float fz; // N
  int numSprings;
};

struct Spring {
  const float k; // N/m
  const int p1; // Index of first point
  const int p2; // Index of second point
  const float l0; // meters
  const int flexIndex;
};

struct FlexPreset {
    const float a;
    const float b;
    const float c;
};*/

// Updates the x, y, and z values of the points after running a simulation for n seconds
bool simulateCPP(std::vector<Point> &points, std::vector<Spring> &springs, std::vector<FlexPreset> presets, double n, float oscillationFrequency);

bool simulateAgainCPP(std::vector<Point>& points, std::vector<Spring>& springs, std::vector<FlexPreset> presets, double n, double t, float oscillationFrequency);

#endif