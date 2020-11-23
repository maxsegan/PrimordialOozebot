#ifndef CPP_SIM_H
#define CPP_SIM_H

#include <vector>

struct Point {
  double x; // meters
  double y; // meters
  double z; // meters
  double vx; // meters/second
  double vy; // meters/second
  double vz; // meters/second
  const double mass; // kg
  double fx; // N - internal bookkeeping
  double fy; // N
  double fz; // N
};

struct Spring {
  const double k; // N/m
  const int p1; // Index of first point
  const int p2; // Index of second point
  const double l0; // meters
  const int flexIndex;
};

struct FlexPreset {
    const double a;
    const double b;
    const double c;
};

// Updates the x, y, and z values of the points after running a simulation for n seconds
void simulate(std::vector<Point> &points, std::vector<Spring> &springs, std::vector<FlexPreset> presets, double n, double oscillationFrequency);

#endif