#ifndef CUDA_SIM
#define CUDA_SIM

#include <vector>

struct Point {
  float x; // meters
  float y; // meters
  float z; // meters
  float vx; // meters/second
  float vy; // meters/second
  float vz; // meters/second
  float mass; // kg
  int numSprings; // Int - hack for CUDA ease
  int springDeltaIndex;
};

struct Spring {
  const float k; // N/m
  const int p1; // Index of first point
  const int p2; // Index of second point
  const float l0; // meters
  const int p1SpringIndex;
  const int p2SpringIndex;
  const int flexIndex;
};

struct FlexPreset {
  const float a;
  const float b;
  const float c;
};

struct SpringDelta {
  float dx;
  float dy;
  float dz;
};

// Updates the x, y, and z values of the points after running a simulation for n seconds
void simulate(std::vector<Point> &points, std::vector<Spring> &springs, std::vector<FlexPreset> presets, double n, double oscillationFrequency);

#endif