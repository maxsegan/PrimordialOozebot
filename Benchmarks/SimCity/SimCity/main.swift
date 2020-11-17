//
//  main.swift
//  SimCity
//
//  Created by Maxwell Segan on 10/27/20.
//

import Foundation
import ArgumentParser
import SceneKit

struct Point {
  var x: Double // meters
  var y: Double // meters
  var z: Double // meters
  var v: Double // meters/second
  var a: Double // meters/second^2
  let mass: Double // kg
}

struct Spring {
  let k: Double // N/m
  let l0: Double // meters
  let p1: Int // Index of first point
  let p2: Int // Index of second point
}

struct SimCity: ParsableCommand {
  static let configuration = CommandConfiguration(
    abstract: "Simulates a cube bouncing")
  
  // Argument and Option are both also valid
  @Flag(help: "Should it render or not")
  var render = false
  
  func run() {
    let date = NSDate.now
    let strDate = date.description
    print("start " + strDate)
    //let filePath = NSString(string:"~/Desktop/tsp.txt").expandingTildeInPath
    //let p = URL(fileURLWithPath: filePath)
    //let text = try? String(contentsOf: p)
    
    var points = [Point]()
    var springs = [Spring]()
    let gravity = (0, 0, -9.81)
    for x in 0...1 {
      for y in 0...1 {
        for z in 0...1 {
          // (0,0,0) or (0.1,0.1,0.1) and all combinations
          points.append(Point(x: Double(x) / 10.0, y: Double(y) / 10.0, z: Double(z) / 10.0, v: 0, a: 0, mass: 0.1))
        }
      }
    }
    for i in 0..<points.count {
      for j in (i+1)..<points.count {
        let p1 = points[i]
        let p2 = points[j]
        let length = (pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2)).squareRoot()
        springs.append(Spring(k: 10000, l0: length, p1: i, p2: j))
      }
    }

    
    /*let outfilePath = URL(fileURLWithPath:NSString(string:"~/Desktop/" + outfile + ".csv").expandingTildeInPath)
    do {
      try outText.write(to: outfilePath, atomically: true, encoding: String.Encoding.utf8)
    } catch {
      print("failed to write")
    }
    print("done " + NSDate.now.description)*/
  }
  
  /*static func selectWeightedRandom(weights: [Double]) -> Int {
    let r = Double.random(in: 0.0..<1.0)
    var accum = 0.0
    for i in 0..<weights.count {
      let p = weights[i]
      accum += p
      if r < accum {
        return i
      }
    }
    return -1
  }*/
}

SimCity.main()


