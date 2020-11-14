using Printf
using Profile
using Juno
using CUDA

struct Point
    x::Float32 # meters
    y::Float32 # meters
    z::Float32 # meters
    vx::Float32 # meters/second
    vy::Float32 # meters/second
    vz::Float32 # meters/second
    mass::Float32 # kg
    numSprings::Int32
end

struct Spring
    k::Float32 # N/m
    p1::Int # Index of first point
    p2::Int # Index of second point
    l0::Float32 # meters
    dx::Float32
    dy::Float32
    dz::Float32
end

const kMaxSprings = 28
const kSpring = 500.0
const kGround = 100000.0
const kOscillationFrequency = 0
const kDropHeight = 0.2
const kNumSide = 10
const staticFriction = 0.5
const kineticFriction = 0.3
const dt = 0.0001
const dampening = 0.9995
const gravity = -9.81

function main()
    t::Float64 = 0.0
    points::Array{Point} = []
    springs::Array{Spring} = []
    springIndices::Array{Int32} = Array{Int32}(undef, kNumSide * kNumSide * kNumSide * kMaxSprings)

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


    connected::Dict{Int32, Array{Int32}} = Dict()
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
                            length::Float32 = sqrt(abs2(p1.x - p2.x) + abs2(p1.y - p2.y) + abs2(p1.z - p2.z))
                            points[p1index] = Point(p1.x, p1.y, p1.z, p1.vx, p1.vy, p1.vz, p1.mass, p1.numSprings + 1)
                            p1 = points[p1index]
                            points[p2index] = Point(p2.x, p2.y, p2.z, p2.vx, p2.vy, p2.vz, p2.mass, p2.numSprings + 1)
                            p2 = points[p2index]
                            push!(springs, Spring(kSpring, p1index, p2index, length, 0, 0, 0))
                            currentNumSprings::Int32 = size(springs)[1]
                            springIndices[(p1index - 1) * kMaxSprings + p1.numSprings] = currentNumSprings
                            springIndices[(p2index - 1) * kMaxSprings + p2.numSprings] = currentNumSprings
                        end
                    end
                end
            end
        end
    end
    
    numSprings::Int32 = size(springs)[1]
    numPoints::Int32 = size(points)[1]


    points_d = CuArray(points)
    springs_d = CuArray(springs)
    springIndices_d = CuArray(springIndices)

    limit::Float64 = 5
    println("num springs evaluated: ", numSprings)
    println("time multiplier: ",  limit / dt)

    @time begin
        while t < limit
            adjust::Float32 = 1 + sin(t * kOscillationFrequency) * 0.1

            @cuda updateSprings(points_d, springs_d, adjust, numSprings)
            @cuda updatePoints(points_d, springs_d, springIndices_d, numPoints)
            
            t += dt
        end
    end
    points = Array(points_d)
    println(points[1].x, ",", points[1].y, ",", points[1].z)
end

function updateSprings(points::Array{Point}, springs::Array{Spring}, adjust::Float32, n::Float32)
    i::Int32 = (blockIdx().x-1) * blockDim().x + threadIdx().x
    l::Spring = springs[i]

    p1::Point = points[l.p1]
    p2::Point = points[l.p2]
    k::Float32 = l.k
    l0::Float32 = l.l0

    px::Float32 = p1.x - p2.x
    py::Float32 = p1.y - p2.y
    pz::Float32 = p1.z - p2.z

    dist::Float32 = sqrt(abs2(px) + abs2(py) + abs2(pz))

    # negative if repelling, positive if attracting
    f::Float32 = k * (dist - (l.l0 * adjust))
    # distribute force across the axes
    fdist::Float32 = f / dist
    dx::Float32 = fdist * px
    dy::Float32 = fdist * py
    dz::Float32 = fdist * pz
    
    springs[i] = Spring(k, l.p1, l.p2, l0, dx, dy, dz)
end

function updatePoints(points::Array{Point}, springs::Array{Spring}, springIndices::Array{Int32}, n::Int32)
    i::Int32 = (blockIdx().x-1) * blockDim().x + threadIdx().x
    p::Point = points[i]

    fy = gravity * p.mass
    fx = 0
    fz = 0

    for j in 1:p.numSprings
        springIndex::Int32 = springIndices[(i - 1) * kMaxSprings + j]
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
        fh::Float32 = sqrt(abs2(fx) + abs2(fz))
        if fh < abs(fy * staticFriction)
            fx = 0
            fz = 0
        else
            fyfric = fy * kineticFriction
            fx = fx - fyfric
            fz = fz - fyfric
        end
    end
    ax::Float32 = fx / p.mass
    ay::Float32 = fy / p.mass + gravity
    az::Float32 = fz / p.mass
    vx::Float32 = (ax * dt + p.vx) * dampening
    vy::Float32 = (ay * dt + p.vy) * dampening
    vz::Float32 = (az * dt + p.vz) * dampening
    x::Float32 = p.x + p.vx * dt
    y::Float32 = p.y + p.vy * dt
    z::Float32 = p.z + p.vz * dt
    points[i] = Point(x, y, z, vx, vy, vz, p.mass, p.numSprings)
end

main()
#@profile main()
