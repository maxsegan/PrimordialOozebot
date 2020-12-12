#ifndef CUDA_SIM
#define CUDA_SIM

#include <vector>
#include <cuda_runtime.h>

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
  int numPoints;
  Point *p_d;
  int pointsLength;
  Spring *s_d;
  int springsLength;
  SpringDelta *ps_d;
  int springDeltaLength;
  int *b_d;
  int *invalid_h;
  int numSprings;
  double length;
  double duration; // It will run for slightly longer than requested to align to the same point in the frequency
  int device;
  cudaStream_t stream;
};

AsyncSimHandle createSimHandle(int i);

void releaseSimHandle(AsyncSimHandle &handle);

// Updates the x, y, and z values of the points after running a simulation for n seconds
void simulate(AsyncSimHandle &handle, std::vector<Point> &points, std::vector<Spring> &springs, std::vector<FlexPreset> &presets, double n, double oscillationFrequency);

// wait on handle to finish current sim
void synchronize(AsyncSimHandle &handle);

// Continue it's current simulation
void simulateAgain(AsyncSimHandle &handle, std::vector<FlexPreset> &presets, double t, double n, double oscillationFrequency);

#endif
