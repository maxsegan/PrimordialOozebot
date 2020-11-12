#!/usr/bin/swift

import Dispatch

let staticFriction = 0.5
let kineticFriction = 0.3
let dt = 0.0001
let dampening = 1 - (dt * 5)
let gravity = -9.81
let kNumPerSide = 20
let kSpring: Double = 500.0
let kGround: Double = 100000.0
let kOscillationFrequency: Double = 0
let kDropHeight: Double = 0.2
let kMaxSprings: Int = 28

struct Point {
  var x: Double // meters
  var y: Double // meters
  var z: Double // meters
  var vx: Double // meters/second
  var vy: Double // meters/second
  var vz: Double // meters/second
  let mass: Double // kg
  var numSprings: Int
}

struct Spring {
  let k: Double // N/m
  let p1: Int // Index of first point
  let p2: Int // Index of second point
  let l0: Double // meters
}

func main() {
  let ret = gen()
  
  updateSim(ps: ret.points, lines: ret.springs, springIndices: ret.springIndices)
}

func updateSim(ps: [Point], lines: [Spring], springIndices: [Int]) {
  var points = ps
  var t = 0.0
  // 60 fps - 0.000166
  let limit:Double = 1
  let realTime = DispatchTime.now()

  let numCores = 4

  var pointSplits:[Int] = []
  for i in 0...numCores {
    pointSplits.append(i * points.count / numCores)
  }
  
  let group = DispatchGroup()

  while t < limit {
    let adjust = 1 + sin(t * kOscillationFrequency) * 0.1

    for i in 0..<numCores-1 {
      group.enter()
      DispatchQueue.global().async {
        updatePoints(points: &points, lines: lines, springIndices: springIndices, adjust: adjust, start: pointSplits[i], stop: pointSplits[i+1])
        group.leave()
      }
    }
    updatePoints(points: &points, lines: lines, springIndices: springIndices, adjust: adjust, start: pointSplits[numCores - 1], stop: pointSplits[numCores])
    group.wait()

    t += dt
  }
  let ts = (Double(DispatchTime.now().uptimeNanoseconds - realTime.uptimeNanoseconds)) / 1000000000.0
  print("num springs evaluated: ", Double(lines.count) * limit / dt, ts)
  print(Double(lines.count) * limit / dt / ts)
  print(points[0].x, points[0].y, points[0].z)
}

func updatePoints(points: inout [Point], lines: [Spring], springIndices: [Int], adjust: Double, start: Int, stop: Int) {
  for i in start..<stop {
    var p = points[i]
      
    let mass = p.mass
    var fy:Double = gravity * mass
    var fx:Double = 0.0
    var fz:Double = 0.0

    for j in 0..<p.numSprings {
      let springIndex = springIndices[i * kMaxSprings + j]
      let s = lines[springIndex]

      let p1 = p
      var p2 = p
      if i == s.p1 {
        p2 = points[s.p2]
      } else {
        p2 = points[s.p1]
      }

      let xd:Double = p1.x - p2.x
      let yd:Double = p1.y - p2.y
      let zd:Double = p1.z - p2.z

      let dist:Double = sqrt(pow(xd, 2) + pow(yd, 2) + pow(zd, 2))

      // negative if repelling, positive if attracting
      let f:Double = s.k * (dist - (s.l0 * adjust))
      let fd:Double = f / dist

      // distribute force across the axes
      let dx:Double = fd * xd
      let dy:Double = fd * yd
      let dz:Double = fd * zd
      fx -= dx
      fy -= dy
      fz -= dz
    }
    
    let y = p.y
    var vx = p.vx
    var vy = p.vy
    var vz = p.vz

    if y <= 0 {
      let fh = sqrt(pow(fx, 2) + pow(fz, 2))
      let fyfric = abs(fy * staticFriction)
      if fh < fyfric {
        fx = 0
        fz = 0
      } else {
        let fykinetic = abs(fy * kineticFriction)
        fx = fx - fx / fh * fykinetic
        fz = fz - fz / fh * fykinetic
      }
      fy += -kGround * y
    }
    let ax = fx / mass
    let ay = fy / mass
    let az = fz / mass
    vx = (ax * dt + vx) * dampening
    p.vx = vx
    vy = (ay * dt + vy) * dampening
    p.vy = vy
    vz = (az * dt + vz) * dampening
    p.vz = vz
    p.x += vx * dt
    p.y += vy * dt
    p.z += vz * dt
    points[i] = p
  }
}

func gen() -> (points: [Point], springs: [Spring], springIndices: [Int]) {
  var points = [Point]()
  var springs = [Spring]()
  var springIndices = [Int]()
  
  for x in 0..<kNumPerSide {
    for y in 0..<kNumPerSide {
      for z in 0..<kNumPerSide {
        // (0,0,0) or (0.1,0.1,0.1) and all combinations
        let p = Point(x: Double(x) / 10.0,
                      y: kDropHeight + Double(y) / 10.0,
                      z: Double(z) / 10.0,
                      vx: 0,
                      vy: 0,
                      vz: 0,
                      mass: 0.1,
                      numSprings: 0)
        points.append(p)
      }
    }
  }
  for _ in 0..<points.count * kMaxSprings {
    springIndices.append(0)
  }
  
  var connected: [Int: [Int]] = [:]
  for x in 0..<kNumPerSide {
    for y in 0..<kNumPerSide {
      for z in 0..<kNumPerSide {
        let p1index = z + kNumPerSide * y + kNumPerSide * kNumPerSide * x
        if connected[p1index] == nil {
          connected[p1index] = []
        }
        
        var p1 = points[p1index]
        for x1 in (x-1)...(x+1) {
          if x1 == kNumPerSide || x1 < 0 {
            continue
          }
          for y1 in (y-1)...(y+1) {
            if y1 == kNumPerSide || y1 < 0 {
              continue
            }
            for z1 in (z-1)...(z+1) {
              if z1 == kNumPerSide || z1 < 0 || (x == x1 && y == y1 && z == z1) {
                continue
              }
              let p2index = z1 + kNumPerSide * y1 + kNumPerSide * kNumPerSide * x1
              if connected[p2index] == nil {
                connected[p2index] = []
              } else if connected[p1index]!.contains(p2index) {
                continue
              }
              connected[p1index]!.append(p2index)
              connected[p2index]!.append(p1index)
              let p2 = points[p2index]
              let length = (pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2)).squareRoot()
              let si1 = p1index * kMaxSprings + p1.numSprings
              let si2 = p2index * kMaxSprings + p2.numSprings
              springIndices[si1] = springs.count
              springIndices[si2] = springs.count
              springs.append(Spring(k: kSpring, p1: p1index, p2: p2index, l0: length))
              points[p1index].numSprings += 1
              points[p2index].numSprings += 1
              p1.numSprings += 1
            }
          }
        }
      }
    }
  }
  
  return (points, springs, springIndices)
}

main()
