import time
import math
import matplotlib.pyplot as plt


kSpring = 10000.0
kGround = 100000.0
kOscillationFrequency = 10000#100000
kDropHeight = .2

times = []
y_val = []
kinetic = []
potential = []
#test = []

ke = 0
pe = 0


class Point:
    def __init__(self, x, y, z, vx, vy, vz, mass, fx, fy, fz):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass
        self.fx = fx
        self.fy = fy
        self.fz = fz

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
    dampening = 1
    gravity = -9.81

    limit = 0.001
    print("num springs evaluated: ", len(springs))
    print("time multiplier: ",  0.001 / dt)

    start_time = time.time()
    sim(limit, staticFriction, kineticFriction, dt, dampening, gravity, points, springs)

    print("--- %s seconds ---" % (time.time() - start_time))

def sim(limit, staticFriction, kineticFriction, dt, dampening, gravity, points, springs):
    t = 0.0
    while t < limit:
        adjust = 1 + math.sin(t * kOscillationFrequency) * 0.1
        #print('t is', t)
        
        ke = 0
        pe = 0
        
        for l in springs:
            p1 = points[l.p1]
            p2 = points[l.p2]
            p1x = p1.x
            p1y = p1.y
            p1z = p1.z
            p2x = p2.x
            p2y = p2.y
            p2z = p2.z
            dist = math.sqrt((p1x - p2x)**2 + (p1y - p2y)**2 + (p1z - p2z)**2)

            # negative if repelling, positive if attracting
            f = l.k * (dist - (l.l0 * adjust))
            # distribute force across the axes
            dx = f * (p1x - p2x) / dist
            dy = f * (p1y - p2y) / dist
            dz = f * (p1z - p2z) / dist

            p1.fx -= dx
            p2.fx += dx

            p1.fy -= dy
            p2.fy += dy

            p1.fz -= dz
            p2.fz += dz
            
            spe = .5 * kSpring * (dist - (l.l0 * adjust))**2
            #print('spe is', spe)
            #if tester == 1:
            #    print(t, l, 'spe is', spe)
                
            pe = pe + spe
            #if tester == 1:
            #    print('pe is', pe)
            

        for p in points:
            fy = p.fy
            fx = p.fx
            fz = p.fz
            mass = p.mass
            
            
            
            if p.y < 0:
                fy += -kGround * p.y
                fh = math.sqrt(fx**2 + fz**2)
                #if fh < abs(fy * staticFriction):
                 #   fx = 0
                 #   p.vx = 0
                 #   fz = 0
                  #  p.vz = 0
                #else:
                #    fyfric = fy * kineticFriction
                #    fx = fx - fyfric
                 #   fz = fz - fyfric
            ax = fx / mass
            ay = fy / mass + gravity
            az = fz / mass
            # reset the force cache
            p.fx = 0
            p.fy = 0
            p.fz = 0
            p.vx = (ax * dt + p.vx) * dampening
            p.vy = (ay * dt + p.vy) * dampening
            p.vz = (az * dt + p.vz) * dampening
            p.x += p.vx
            p.y += p.vy
            p.z += p.vz
            
            v = math.sqrt(p.vx**2 + p.vy**2 + p.vz**2)
            
            p_ke = .5 * p.mass * v**2
            ke = ke + p_ke
            gpe = p.mass * p.y
            
            #if tester == 4:
            #    print(t, p, 'gpe is', gpe)
            #    print(v)
            pe = pe + gpe
            
            #if t == 0:
            #    print('pe is', pe)
            #new = ("test is", p.vx,p.vy,p.vz,v,p.mass,ke)
            #print(new)
            #test.append(new)
        
        if t % .5:
            kinetic.append(ke)
            times.append(t)
            potential.append(pe)
         
        ke=0
        pe=0
        #if t % 2:
        #    print('at t of', t)
        #    print(ke)
         #   print(pe)

        #tester = tester + 1
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
                p = Point(x / 10.0, kDropHeight + y / 10.0, z / 10.0, 0, 0, 0, 0.1, 0, 0, 0)
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
                            length = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
                            springs.append(Spring(kSpring, p1index, p2index, length, length))
    return points, springs

if __name__ == "__main__":
    main()

x = times
y1 = kinetic

y2 = potential

print('potential energy is', potential[0:15])
print('kinetic energy is', kinetic[0:15])
#plt.plot(x,y1)
plt.plot(x,y1,label = kinetic)
plt.plot(x,y2,label = potential)
plt.show()

#plt.plot(x,y2)
#plt.show()
#print(kinetic)