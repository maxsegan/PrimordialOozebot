#include "cppSim.h"
#include <algorithm>
#include <iostream>
#include <math.h>
#include <map>
#include <chrono>

const float kGround = -100000.0;
const float dt = 0.0001;
const float dampening = 0.999;
const float gravity = -9.81;

bool simulateCPP(std::vector<Point>& points, std::vector<Spring>& springs, std::vector<FlexPreset> presets, double n, float oscillationFrequency) {
    return simulateAgainCPP(points, springs, presets, n, 0, oscillationFrequency);
}

bool simulateAgainCPP(std::vector<Point>&points, std::vector<Spring>&springs, std::vector<FlexPreset> presets, double n, double t, float oscillationFrequency) {
    std::vector<float> presetValues;
    for (auto it = presets.begin(); it != presets.end(); it++) {
        presetValues.push_back(0.0);
    }
    while (t < n) {
        for (int i = 0; i < presetValues.size(); i++) {
            const float a = presets[i].a;
            const float b = presets[i].b;
            const float c = presets[i].c;
            presetValues[i] = a * (1 + b * sin(t * oscillationFrequency + c));
        }
        for (std::vector<Spring>::iterator i = springs.begin(); i != springs.end(); ++i) {
            Spring l = *i;

            const int p1index = l.p1;
            const int p2index = l.p2;
            Point p1 = points[p1index];
            Point p2 = points[p2index];

            const float xd = p1.x - p2.x;
            const float yd = p1.y - p2.y;
            const float zd = p1.z - p2.z;
            const float dist = sqrt(xd * xd + yd * yd + zd * zd);

            if (dist > (l.l0 * 6)) {
                return false;
            }

            // negative if repelling, positive if attracting
            const float f = l.k * (dist - (l.l0 * presetValues[l.flexIndex]));
            const float fd = f / dist;
            // distribute force across the axes
            const float dx = xd * fd;
            const float dy = yd * fd;
            const float dz = zd * fd;

            points[p1index].fx -= dx;
            points[p2index].fx += dx;

            points[p1index].fy -= dy;
            points[p2index].fy += dy;

            points[p1index].fz -= dz;
            points[p2index].fz += dz;
        }
        for (std::vector<Point>::iterator i = points.begin(); i != points.end(); ++i) {
            Point p = *i;
        
            const float mass = p.mass;
            const float y = p.y;
            float fy = p.fy + gravity * mass;
            float fx = p.fx;
            float fz = p.fz;
            float vx = p.vx;
            float vy = p.vy;
            float vz = p.vz;

            if (y <= 0) {
                double fh = sqrt(fx * fx + fz * fz);
                const float fyfric = abs(fy * p.us);
                if (fh < fyfric) {
                    fx = 0;
                    fz = 0;
                } else {
                    const float fykinetic = abs(fy * p.uk) / fh;
                    fx = fx - fx * fykinetic;
                    fz = fz - fz * fykinetic;
                }
                fy += kGround * y;
            }
            const float ax = fx / mass;
            const float ay = fy / mass;
            const float az = fz / mass;
            // reset the force cache
            p.fx = 0;
            p.fy = 0;
            p.fz = 0;
            vx = (ax * dt + vx) * dampening;
            vy = (ay * dt + vy) * dampening;
            vz = (az * dt + vz) * dampening;
            p.vx = vx;
            p.vy = vy;
            p.vz = vz;
            p.x += vx * dt;
            p.y += vy * dt;
            p.z += vz * dt;
            *i = p;
        }
        t += dt;
    }
}
