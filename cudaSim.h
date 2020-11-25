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
  int numSprings; // Int - hack for CUDA ease - must be filled in externally, though
  int springDeltaIndex; // Filled in internally, ignore
  float timestampsContactGround = 0; // tracked internally
};

struct Spring {
  const float k; // N/m
  const int p1; // Index of first point
  const int p2; // Index of second point
  const float l0; // meters
  int p1SpringIndex; // Filled in externally - these just must be incremental and different for each point
  int p2SpringIndex;
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

struct AsyncSimHandle {
  std::vector<Point> points;
  Point *p_d;
  Spring *s_d;
  SpringDelta *ps_d;
  double start;
};

// Updates the x, y, and z values of the points after running a simulation for n seconds
AsyncSimHandle simulate(std::vector<Point> &points, std::vector<Spring> &springs, std::vector<FlexPreset> &presets, double n, double oscillationFrequency);

void resolveSim(AsyncSimHandle &handle);

#endif