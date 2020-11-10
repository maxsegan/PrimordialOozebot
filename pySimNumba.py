
import time
import math
import numpy
from numba import jit

kSpring = 10000.0
kGround = 100000.0
kOscillationFrequency = 0#10000#100000
kDropHeight = 0.2

class Point:
    def __init__(self, x, y, z, vx, vy, vz, mass, fx, fy, fz):
        self.x = x #0
        self.y = y #1
        self.z = z #2
        self.vx = vx #3
        self.vy = vy #4
        self.vz = vz #5
        self.mass = mass #6
        self.fx = fx #7
        self.fy = fy #8
        self.fz = fz #9

class Spring:
    def __init__(self, k, p1, p2, l0, currentl):
        self.k = k
        self.p1 = p1
        self.p2 = p2
        self.l0 = l0
        self.currentl = currentl

def main():
    points, springs = genPointsAndSprings()

    staticFriction = 0.5
    kineticFriction = 0.3
    dt = 0.0000005
    dampening = 1 - (dt * 1000)
    gravity = -9.81

    limit = 0.001
    print("num springs evaluated: ", len(springs))
    print("time multiplier: ",  limit / dt)

    start_time = time.time()
    sim(limit, staticFriction, kineticFriction, dt, dampening, gravity, points, springs)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    limit = 0.01
    print("time multiplier: ",  limit / dt)
    start_time = time.time()
    sim(limit, staticFriction, kineticFriction, dt, dampening, gravity, points, springs)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    sim(limit, staticFriction, kineticFriction, dt, dampening, gravity, points, springs)
    print("--- %s seconds ---" % (time.time() - start_time))

@jit(nopython=True)
def sim(limit, staticFriction, kineticFriction, dt, dampening, gravity, points, springs):
    t = 0.0
    while t < limit:
        adjust = 1 + math.sin(t * kOscillationFrequency) * 0.1
        for l in springs:
            p1 = points[int(l[1])]
            p2 = points[int(l[2])]
            p1x = p1[0]
            p1y = p1[1]
            p1z = p1[2]
            p2x = p2[0]
            p2y = p2[1]
            p2z = p2[2]
            dist = math.sqrt((p1x - p2x)**2 + (p1y - p2y)**2 + (p1z - p2z)**2)

            # negative if repelling, positive if attracting
            f = l[0] * (dist - (l[3] * adjust))
            # distribute force across the axes
            dx = f * (p1x - p2x) / dist
            dy = f * (p1y - p2y) / dist
            dz = f * (p1z - p2z) / dist

            p1[7] -= dx
            p2[7] += dx

            p1[8] -= dy
            p2[8] += dy

            p1[9] -= dz
            p2[9] += dz

        for p in points:
            fx = p[7]
            fy = p[8]
            fz = p[9]
            mass = p[6]

            if p[1] < 0:
                fy += -kGround * p[1]
                fh = math.sqrt(fx**2 + fz**2)
                if fh < abs(fy * staticFriction):
                    fx = 0
                    p[3] = 0
                    fz = 0
                    p[5] = 0
                else:
                    fyfric = fy * kineticFriction
                    fx = fx - fyfric
                    fz = fz - fyfric
            ax = fx / mass
            ay = fy / mass + gravity
            az = fz / mass
            # reset the force cache
            p[7] = 0
            p[8] = 0
            p[9] = 0
            p[3] = (ax * dt + p[3]) * dampening
            p[4] = (ay * dt + p[4]) * dampening
            p[5] = (az * dt + p[5]) * dampening
            p[0] += p[3]
            p[1] += p[4]
            p[2] += p[5]
        t += dt

def genPointsAndSprings():
    cache = {}
    points = []
    springs = []

    # Create the points
    for x in range(10):
        for y in range(10):
            for z in range(10):
                # (0,0,0) or (0.1,0.1,0.1) and all combinations
                p = numpy.array([x / 10.0, kDropHeight + y / 10.0, z / 10.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
                points.append(p)
                if not x in cache:
                    cache[x] = {}
                if not y in cache[x]:
                    cache[x][y] = {}
                cache[x][y][z] = p

    #Create the springs
    for x in range(10):
        for y in range(10):
            for z in range(10):
                p1 = cache[x][y][z]
                p1index = z + 10 * y + 100 * x
                for x1 in range(x, x+2):
                    if x1 == 10:
                        continue
                    for y1 in range(y, y+2):
                        if y1 == 10:
                            continue
                        for z1 in range(z, z+2):
                            if z1 == 10 or (x1 == x and y1 == y and z1 == z):
                                continue
                            p2 = cache[x1][y1][z1]
                            p2index = z1 + 10 * y1 + 100 * x1
                            length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
                            springs.append(numpy.array([kSpring, p1index, p2index, length, length]))
    return numpy.array(points), numpy.array(springs)

if __name__ == "__main__":
    main()
