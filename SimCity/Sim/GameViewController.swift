
//
//  GameViewController.swift
//  Sim
//
//  Created by Maxwell Segan on 10/27/20.
//

import SceneKit
import QuartzCore

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
    cameraNode.position = SCNVector3(x: 0.05, y: 0.05, z: 1)
    
    // create and add an ambient light to the scene
    let ambientLightNode = SCNNode()
    ambientLightNode.light = SCNLight()
    ambientLightNode.light!.type = .ambient
    ambientLightNode.light!.color = NSColor.yellow
    scene.rootNode.addChildNode(ambientLightNode)
    
    let floorNode = SCNNode()
    floorNode.geometry = SCNFloor()
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

    perform(#selector(update), with: nil, afterDelay: 0.08)
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
  var currentl: Double? // meters
}

func gen() -> (points: [Point], springs: [Spring]) {
  var points = [Point]()
  var springs = [Spring]()
  for x in 0...1 {
    for y in 0...1 {
      for z in 0...1 {
        // (0,0,0) or (0.1,0.1,0.1) and all combinations
        points.append(Point(x: Double(x) / 10.0,
                            y: 0.2 + Double(y) / 10.0,
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
  for i in 0..<points.count {
    for j in (i+1)..<points.count {
      let p1 = points[i]
      let p2 = points[j]
      let length = (pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2)).squareRoot()
      springs.append(Spring(k: 5000, p1: i, p2: j, l0: length, currentl: length))
    }
  }
  return (points, springs)
}

func updateSim(points: inout [Point], lines: inout [Spring], time: Double) -> Double {
  var t = time
  let dt = 0.0000005
  let dampening = 1 - (dt * 1000)
  let gravity = -9.81
  // 60 fps - 0.0166
  let limit = t + 0.00003
  while t < limit {
    let adjust = 1 + sin(t * 10000) * 0.1
    for l in lines {
      let dist = sqrt(pow(points[l.p1].x - points[l.p2].x, 2) + pow(points[l.p1].y - points[l.p2].y, 2) + pow(points[l.p1].z - points[l.p2].z, 2))

      // negative if repelling, positive if attracting
      let f = l.k * (dist - (l.l0 * adjust))
      // distribute force across the axes
      let dx = f * (points[l.p1].x - points[l.p2].x) / dist
      points[l.p1].fx -= dx
      points[l.p2].fx += dx

      let dy = f * (points[l.p1].y - points[l.p2].y) / dist
      points[l.p1].fy -= dy
      points[l.p2].fy += dy

      let dz = f * (points[l.p1].z - points[l.p2].z) / dist
      points[l.p1].fz -= dz
      points[l.p2].fz += dz
    }
    for i in 0..<points.count {
      var fy = points[i].fy
      if points[i].y < 0 {
        fy += -100000 * points[i].y
      }
      let ax = points[i].fx / points[i].mass
      let ay = fy / points[i].mass + gravity
      let az = points[i].fz / points[i].mass
      // reset the force cache
      points[i].fx = 0
      points[i].fy = 0
      points[i].fz = 0
      points[i].vx += ax * dt
      points[i].vy = (ay * dt + points[i].vy) * dampening
      points[i].vz += az * dt
      points[i].x += points[i].vx
      points[i].y += points[i].vy
      points[i].z += points[i].vz
    }
    t += dt
  }
  return t
}

func draw(points: [Point], lines: [Spring], scene: SCNScene) {
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
    lineNode.position = midPosition
    lineNode.look(at: SCNVector3(p2.x, p2.y, p2.z), up: scene.rootNode.worldUp, localFront: lineNode.worldUp)
    box.addChildNode(lineNode)
  }
  scene.rootNode.addChildNode(box)
}
