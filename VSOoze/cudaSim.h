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
  float uk; // kinetic friction coefficient
  float us; // static friction coefficient
  int numSprings; // Int - hack for CUDA ease - must be filled in externally, though
  int springDeltaIndex; // Filled in internally, ignore
  float fx; // N - internal bookkeeping for the cpp sim
  float fy; // N
  float fz; // N
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
  Point *endPoints;
  Point *startPoints;
  Point *p_d;
  Spring *s_d;
  SpringDelta* ps_d;
  int* b_d;
  int numPoints;
  int numSprings;
  int *invalid_h;
  double duration; // It will run for slightly longer than requested to align to the same point in the frequency
  int device;
};

AsyncSimHandle createSimHandle(int i, int numPoints, int numSprings);

void releaseSimHandle(AsyncSimHandle &handle);

// Updates the x, y, and z values of the points after running a simulation for n seconds
void simulate(AsyncSimHandle &handle, std::vector<Point> &points, std::vector<Spring> &springs, std::vector<FlexPreset> &presets, double n, double oscillationFrequency);

// Continue it's current simulation
void simulateAgain(AsyncSimHandle &handle, std::vector<FlexPreset> &presets, double t, double n, double oscillationFrequency);

#endif
