
//
//  GameViewController.swift
//  Sim
//
//  Created by Maxwell Segan on 10/27/20.
//

import SceneKit
import QuartzCore

let kSpring: Double = 10000.0
let kGround: Double = 100000.0
let kOscillationFrequency: Double = 0//10000//100000
let kUseTetrahedron = false
let kDropHeight: Double = 0.2
let kUseThousand = true
let kNoRender = true

class GameViewController: NSViewController {
  let scene = SCNScene()
  var time: Double = 0.0
  var points: [Point]?
  var lines: [Spring]?
  
  override func viewDidLoad() {
    super.viewDidLoad()
    
    // create a new scene
    //let scene = SCNScene()//named: "art.scnassets/ship.scn")!
    
    // create and add a camera to the scene
    let cameraNode = SCNNode()
    cameraNode.camera = SCNCamera()
    cameraNode.camera?.zNear = 0.01
    scene.rootNode.addChildNode(cameraNode)
    
    // place the camera
    if kUseThousand {
      cameraNode.position = SCNVector3(x: 0.5, y: 0.5, z: 5)
    } else {
      cameraNode.position = SCNVector3(x: 0.05, y: 0.05, z: 1)
    }
    
    // create and add an ambient light to the scene
    let ambientLightNode = SCNNode()
    ambientLightNode.light = SCNLight()
    ambientLightNode.light!.type = .ambient
    ambientLightNode.light!.color = NSColor.yellow
    scene.rootNode.addChildNode(ambientLightNode)
    
    let floorNode = SCNNode()
    let floor = SCNFloor()
    
    floor.reflectivity = kUseThousand ? 0 : 0.5
    floorNode.geometry = floor
    scene.rootNode.addChildNode(floorNode)
    
    let ret = gen()
    self.points = ret.points
    self.lines = ret.springs
    draw(points: self.points!, lines: self.lines!, scene: scene)
    
    // animate the 3d object
    //let box = scene.rootNode.childNode(withName: "box", recursively: true)
    //box?.runAction(SCNAction.repeatForever(SCNAction.rotateBy(x: 0, y: 1, z: 0, duration: 1)))
    
    // retrieve the SCNView
    let scnView = self.view as! SCNView
    
    // set the scene to the view
    scnView.scene = scene
    
    // allows the user to manipulate the camera
    scnView.allowsCameraControl = true
    
    // show statistics such as fps and timing information
    scnView.showsStatistics = true
    
    // configure the view
    scnView.backgroundColor = NSColor.lightGray
    
    perform(#selector(update), with: nil, afterDelay: 0)
  }
  
  @objc func update() {
    let box = scene.rootNode.childNode(withName: "box", recursively: true)
    box?.removeFromParentNode()
    self.time = updateSim(points: &self.points!, lines: &self.lines!, time: self.time)
    draw(points: self.points!, lines: self.lines!, scene: scene)
    let delay:Double = 0.02
    perform(#selector(update), with: nil, afterDelay: delay)
  }
}

struct Point {
  var x: Double // meters
  var y: Double // meters
  var z: Double // meters
  var vx: Double // meters/second
  var vy: Double // meters/second
  var vz: Double // meters/second
  let mass: Double // kg
  var fx: Double // N - reset every iteration
  var fy: Double // N
  var fz: Double // N
}

struct Spring {
  let k: Double // N/m
  let p1: Int // Index of first point
  let p2: Int // Index of second point
  let l0: Double // meters
}

func gen() -> (points: [Point], springs: [Spring]) {
  var points = [Point]()
  var springs = [Spring]()
  var cache: [Int: [Int: [Int: Point]]] = [:]
  
  if kUseTetrahedron {
    // (+/-1, −1/√3, 0), (0, 2/√3, 0), (0, 0, 4/√6) all scaled by 10
    points.append(Point(x: 0.1,
                        y: kDropHeight,
                        z: -0.1 / sqrt(3),
                        vx: 0,
                        vy: 0,
                        vz: 0,
                        mass: 0.1,
                        fx: 0,
                        fy: 0,
                        fz: 0))
    points.append(Point(x: -0.1,
                        y: kDropHeight,
                        z: -0.1 / sqrt(3),
                        vx: 0,
                        vy: 0,
                        vz: 0,
                        mass: 0.1,
                        fx: 0,
                        fy: 0,
                        fz: 0))
    points.append(Point(x: 0,
                        y: kDropHeight,
                        z: 0.2 / sqrt(3),
                        vx: 0,
                        vy: 0,
                        vz: 0,
                        mass: 0.1,
                        fx: 0,
                        fy: 0,
                        fz: 0))
    points.append(Point(x: 0,
                        y: kDropHeight + 0.4 / sqrt(6),
                        z: 0,
                        vx: 0,
                        vy: 0,
                        vz: 0,
                        mass: 0.1,
                        fx: 0,
                        fy: 0,
                        fz: 0))
  } else if kUseThousand {
    for x in 0..<10 {
      for y in 0..<10 {
        for z in 0..<10 {
          // (0,0,0) or (0.1,0.1,0.1) and all combinations
          let p = Point(x: Double(x) / 10.0,
                        y: kDropHeight + Double(y) / 10.0,
                        z: Double(z) / 10.0,
                        vx: 0,
                        vy: 0,
                        vz: 0,
                        mass: 0.1,
                        fx: 0,
                        fy: 0,
                        fz: 0)
          points.append(p)
          if cache[x] == nil {
            cache[x] = [:]
          }
          if cache[x]![y] == nil {
            cache[x]![y] = [:]
          }
          cache[x]![y]![z] = p
        }
      }
    }
  } else {
    for x in 0...1 {
      for y in 0...1 {
        for z in 0...1 {
          // (0,0,0) or (0.1,0.1,0.1) and all combinations
          points.append(Point(x: Double(x) / 10.0,
                              y: kDropHeight + Double(y) / 10.0,
                              z: Double(z) / 10.0,
                              vx: 0,
                              vy: 0,
                              vz: 0,
                              mass: 0.1,
                              fx: 0,
                              fy: 0,
                              fz: 0))
        }
      }
    }
  }
  if kUseThousand {
    for x in 0..<10 {
      for y in 0..<10 {
        for z in 0..<10 {
          let p1 = cache[x]![y]![z]!
          let p1index = z + 10 * y + 100 * x
          // connect to the ones adjacent to it, assuming we never connect to lower numbers not to duplicate
          let ps: [(Point?, Int)] = [(cache[x + 1]?[y]?[z], z + 10 * y + 100 * (x+1)),
                                     (cache[x]?[y + 1]?[z], z + 10 * (y+1) + 100 * x),
                                      (cache[x]?[y]?[z + 1], z + 1 + 10 * y + 100 * x),
                                      (cache[x + 1]?[y + 1]?[z], z + 10 * (y+1) + 100 * (x+1)),
                                      (cache[x + 1]?[y]?[z + 1], z + 1 + 10 * y + 100 * (x+1)),
                                      (cache[x]?[y + 1]?[z + 1], z + 1 + 10 * (y+1) + 100 * x),
                                      (cache[x + 1]?[y + 1]?[z + 1], z + 1 + 10 * (y+1) + 100 * (x+1))]

          for tuple in ps {
            let p = tuple.0
            let p2index = tuple.1
            if p != nil {
              let p2 = p!
              let length = (pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2)).squareRoot()
              springs.append(Spring(k: kSpring, p1: p1index, p2: p2index, l0: length))
            }
          }
        }
      }
    }
  } else {
    for i in 0..<points.count {
      for j in (i+1)..<points.count {
        let p1 = points[i]
        let p2 = points[j]
        let length = (pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2)).squareRoot()
        springs.append(Spring(k: kSpring, p1: i, p2: j, l0: length))
      }
    }
  }
  return (points, springs)
}

func updateSim(points: inout [Point], lines: inout [Spring], time: Double) -> Double {
  var t = time
  let staticFriction = 0.5
  let kineticFriction = 0.3
  let dt = 0.0000005
  let dampening = 1 - (dt * 1000)
  let gravity = -9.81
  // 60 fps - 0.000166
  let limit: Double
  let realTime = CACurrentMediaTime()
  if kNoRender {
    limit = t + 0.1
  } else {
    limit = t + 0.00001
  }
  while t < limit {
    let adjust = 1 + sin(t * kOscillationFrequency) * 0.1
    for l in lines {
      let p1ind = l.p1
      let p2ind = l.p2

      let p1x = points[p1ind].x
      let p1y = points[p1ind].y
      let p1z = points[p1ind].z
      let p2x = points[p2ind].x
      let p2y = points[p2ind].y
      let p2z = points[p2ind].z
      let dist = sqrt(pow(p1x - p2x, 2) + pow(p1y - p2y, 2) + pow(p1z - p2z, 2))

      // negative if repelling, positive if attracting
      let f = l.k * (dist - (l.l0 * adjust))
      // distribute force across the axes
      let dx = f * (p1x - p2x) / dist
      points[p1ind].fx -= dx
      points[p2ind].fx += dx

      let dy = f * (p1y - p2y) / dist
      points[p1ind].fy -= dy
      points[p2ind].fy += dy

      let dz = f * (p1z - p2z) / dist
      points[p1ind].fz -= dz
      points[p2ind].fz += dz
    }
    for i in 0..<points.count {
      var p = points[i]
        
      let mass = p.mass
      var fy = p.fy + gravity * mass
      var fx = p.fx
      var fz = p.fz
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
      // reset the force cache
      p.fx = 0
      p.fy = 0
      p.fz = 0
      vx = (ax * dt + vx) * dampening
      p.vx = vx
      vy = (ay * dt + vy) * dampening
      p.vy = vy
      vz = (az * dt + vz) * dampening
      p.vz = vz
      p.x += vx
      p.y += vy
      p.z += vz
      points[i] = p
    }
    t += dt
  }
  if kNoRender {
    print("num springs evaluated: ", Double(lines.count) * 0.1 / dt, CACurrentMediaTime() - realTime)
  }
  return t
}

func draw(points: [Point], lines: [Spring], scene: SCNScene) {
  if kNoRender {
    return
  }
  let box = SCNNode()
  box.name = "box"
  for spring in lines {
    let p1 = points[spring.p1]
    let p2 = points[spring.p2]
    
    let vector = SCNVector3(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)
    let distance = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
    let midPosition = SCNVector3((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)
    
    let lineGeometry = SCNCylinder()
    lineGeometry.radius = 0.005
    lineGeometry.height = distance
    lineGeometry.radialSegmentCount = 5
    lineGeometry.firstMaterial!.diffuse.contents = NSColor.green
    
    let lineNode = SCNNode(geometry: lineGeometry)
    let lineName: String = String(spring.p1) + "," + String(spring.p2)
    lineNode.name = lineName
    lineNode.position = midPosition
    lineNode.look(at: SCNVector3(p2.x, p2.y, p2.z), up: scene.rootNode.worldUp, localFront: lineNode.worldUp)
    box.addChildNode(lineNode)
  }
  scene.rootNode.addChildNode(box)
}

func redraw(points: [Point], lines: [Spring], scene: SCNScene) {
  if kNoRender {
    return
  }
  let box = scene.rootNode.childNode(withName: "box", recursively: true)
  for spring in lines {
    let p1 = points[spring.p1]
    let p2 = points[spring.p2]
    
    let vector = SCNVector3(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)
    let distance = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
    let midPosition = SCNVector3((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)
    
    let lineName: String = String(spring.p1) + "," + String(spring.p2)
    let lineNode = box!.childNode(withName: lineName, recursively: true)
    let lineGeometry:SCNCylinder = lineNode?.geometry as! SCNCylinder
    lineGeometry.height = distance
    lineNode!.position = midPosition
    lineNode!.look(at: SCNVector3(p2.x, p2.y, p2.z), up: scene.rootNode.worldUp, localFront: lineNode!.worldUp)
  }
}

