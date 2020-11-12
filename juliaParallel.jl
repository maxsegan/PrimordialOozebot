using Printf
using Profile
using Juno
using Distributed

JULIA_NUM_THREADS=4

struct Point
    x::Float64 # meters
    y::Float64 # meters
    z::Float64 # meters
    vx::Float64 # meters/second
    vy::Float64 # meters/second
    vz::Float64 # meters/second
    mass::Float64 # kg
    numSprings::Int64
end

struct Spring
    k::Float64 # N/m
    p1::Int # Index of first point
    p2::Int # Index of second point
    l0::Float64 # meters
    dx::Float64
    dy::Float64
    dz::Float64
end

const kMaxSprings = 28
const kSpring = 500.0
const kGround = 100000.0
const kOscillationFrequency = 0
const kDropHeight = 0.2
const kNumSide = 24
const staticFriction = 0.5
const kineticFriction = 0.3
const dt = 0.0001
const dampening = 0.9995
const gravity = -9.81

function main()
    t::Float64 = 0.0
    points::Array{Point} = []
    springs::Array{Spring} = []
    springIndices::Array{Int64} = Array{Int64}(undef, kNumSide * kNumSide * kNumSide * kMaxSprings)

    # Create the points
    for x = 1:kNumSide
        for y = 1:kNumSide
            for z = 1:kNumSide
                # (0,0,0) or (0.1,0.1,0.1) and all combinations
                p::Point = Point(x / 10.0, kDropHeight + y / 10.0, z / 10.0, 0, 0, 0, 0.1, 0)
                push!(points, p)
            end
        end
    end


    connected::Dict{Int64, Array{Int64}} = Dict()
    #Create the springs
    for x = 1:kNumSide
        for y = 1:kNumSide
            for z = 1:kNumSide
                p1index::Int = z + kNumSide * (y - 1) + kNumSide * kNumSide * (x - 1)
                if !haskey(connected, p1index)
                    connected[p1index] = []
                end
                p1::Point = points[p1index]
                for x1 = (x - 1):(x+1)
                    if x1 == kNumSide + 1 || x1 <= 0
                        continue
                    end
                    for y1 = (y - 1):(y+1)
                        if y1 == kNumSide + 1 || y1 <= 0
                            continue
                        end
                        for z1 = z:(z+1)
                            if z1 == kNumSide + 1 || z1 <= 0 || (x1 == x && y1 == y && z1 == z)
                                continue
                            end
                            p2index::Int = z1 + kNumSide * (y1 - 1) + kNumSide * kNumSide * (x1 - 1)
                            if !haskey(connected, p2index)
                                connected[p2index] = []
                            elseif p2index in connected[p1index]
                                continue
                            end
                            push!(connected[p1index], p2index)
                            push!(connected[p2index], p1index)
                            p2::Point = points[p2index]
                            length::Float64 = sqrt(abs2(p1.x - p2.x) + abs2(p1.y - p2.y) + abs2(p1.z - p2.z))
                            points[p1index] = Point(p1.x, p1.y, p1.z, p1.vx, p1.vy, p1.vz, p1.mass, p1.numSprings + 1)
                            p1 = points[p1index]
                            points[p2index] = Point(p2.x, p2.y, p2.z, p2.vx, p2.vy, p2.vz, p2.mass, p2.numSprings + 1)
                            p2 = points[p2index]
                            push!(springs, Spring(kSpring, p1index, p2index, length, 0, 0, 0))
                            currentNumSprings::Int64 = size(springs)[1]
                            springIndices[(p1index - 1) * kMaxSprings + p1.numSprings] = currentNumSprings
                            springIndices[(p2index - 1) * kMaxSprings + p2.numSprings] = currentNumSprings
                        end
                    end
                end
            end
        end
    end
    
    numSprings::Int64 = size(springs)[1]
    numPoints::Int64 = size(points)[1]

    numCores::Int64 = 4
    println(nworkers(), " ,", nprocs())
  
    springSplits::Array{Int64} = [0]
    for i in 1:numCores
        push!(springSplits, round(i * numSprings / numCores))
    end

    pointSplits::Array{Int64} = [0]
    for i in 1:numCores
        push!(pointSplits, round(i * numPoints / numCores))
    end

    limit::Float64 = 0.25
    println("num springs evaluated: ", numSprings)
    println("time multiplier: ",  limit / dt)

    @time begin
        while t < limit
            adjust::Float64 = 1 + sin(t * kOscillationFrequency) * 0.1

            #threads = []
            @sync @distributed for i = 1:(numCores)
                updateSprings(points, springs, adjust, springSplits[i] + 1, springSplits[i+1])
            end
            #updateSprings(points, springs, adjust, springSplits[numCores] + 1, springSplits[numCores+1])
            #for thread in threads
            #    wait(thread)
            #end=#
            #updateSprings(points, springs, adjust, 1, numSprings)
            #threads = []
            @sync @distributed for i = 1:(numCores)
                #push!(threads, remotecall(updatePoints, i, points, springs, springIndices, pointSplits[i] + 1, pointSplits[i+1]))
                updatePoints(points, springs, springIndices, pointSplits[i] + 1, pointSplits[i+1])
            end
            #updateSprings(points, springs, adjust, pointSplits[numCores] + 1, pointSplits[numCores+1])
            #for thread in threads
            #    wait(thread)
            #end
            #updatePoints(points, springs, springIndices, 1, numPoints)
            t += dt
        end
    end
    println(points[1].x, ",", points[1].y, ",", points[1].z)
end

function updateSprings(points::Array{Point}, springs::Array{Spring}, adjust::Float64, start::Int64, endIter::Int64)
    for i in start:endIter
        l::Spring = springs[i]

        p1::Point = points[l.p1]
        p2::Point = points[l.p2]
        k::Float64 = l.k
        l0::Float64 = l.l0

        px::Float64 = p1.x - p2.x
        py::Float64 = p1.y - p2.y
        pz::Float64 = p1.z - p2.z

        dist::Float64 = sqrt(abs2(px) + abs2(py) + abs2(pz))

        # negative if repelling, positive if attracting
        f::Float64 = k * (dist - (l.l0 * adjust))
        # distribute force across the axes
        fdist::Float64 = f / dist
        dx::Float64 = fdist * px
        dy::Float64 = fdist * py
        dz::Float64 = fdist * pz
        
        springs[i] = Spring(k, l.p1, l.p2, l0, dx, dy, dz)
    end
end

function updatePoints(points::Array{Point}, springs::Array{Spring}, springIndices::Array{Int64}, start::Int64, endIter::Int64)
    for i = start:endIter
        p::Point = points[i]

        fy = gravity * p.mass
        fx = 0
        fz = 0

        for j in 1:p.numSprings
            springIndex::Int64 = springIndices[(i - 1) * kMaxSprings + j]
            s::Spring = springs[springIndex]

            if s.p1 == i
                fx -= s.dx
                fy -= s.dy
                fz -= s.dz
            else
                fx += s.dx
                fy += s.dy
                fz += s.dz
            end
        end

        if p.y < 0
            fy += -kGround * p.y
            fh::Float64 = sqrt(abs2(fx) + abs2(fz))
            if fh < abs(fy * staticFriction)
                fx = 0
                fz = 0
            else
                fyfric = fy * kineticFriction
                fx = fx - fyfric
                fz = fz - fyfric
            end
        end
        ax::Float64 = fx / p.mass
        ay::Float64 = fy / p.mass + gravity
        az::Float64 = fz / p.mass
        vx::Float64 = (ax * dt + p.vx) * dampening
        vy::Float64 = (ay * dt + p.vy) * dampening
        vz::Float64 = (az * dt + p.vz) * dampening
        x::Float64 = p.x + p.vx * dt
        y::Float64 = p.y + p.vy * dt
        z::Float64 = p.z + p.vz * dt
        points[i] = Point(x, y, z, vx, vy, vz, p.mass, p.numSprings)
    end
end

main()
#@profile main()
Juno.profiler(; C = true)
