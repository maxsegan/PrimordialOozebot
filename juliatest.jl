using Printf
using Juno

mutable struct Point
    x::Float64 # meters
    y::Float64 # meters
    z::Float64 # meters
    vx::Float64 # meters/second
    vy::Float64 # meters/second
    vz::Float64 # meters/second
    mass::Float64 # kg
    fx::Float64 # N - reset every iteration
    fy::Float64 # N
    fz::Float64 # N
end

mutable struct Spring
    k::Float64 # N/m
    p1::Int # Index of first point
    p2::Int # Index of second point
    l0::Float64 # meters
    currentl::Float64 # meters
end

function main()
    kSpring::Float64 = 500.0
    kGround::Float64 = 100000.0
    kOscillationFrequency::Float64 = 0#10000.0
    kDropHeight::Float64 = 0.2

    t::Float64 = 0.0
    points::Array{Point} = []
    springs::Array{Spring} = []
    cache::Dict{Int64, Dict{Int64, Dict{Int64, Point}}} = Dict()

    # Create the points
    for x = 1:10
        for y = 1:10
            for z = 1:10
                # (0,0,0) or (0.1,0.1,0.1) and all combinations
                p::Point = Point(x / 10.0, kDropHeight + y / 10.0, z / 10.0, 0, 0, 0, 0.1, 0, 0, 0)
                push!(points, p)
                if !haskey(cache, x)
                    cache[x] = Dict()
                end
                if !haskey(cache[x], y)
                    cache[x][y] = Dict()
                end
                cache[x][y][z] = p
            end
        end
    end

    connected::Dict{Int64, Array{Int64}} = Dict()
    #Create the springs
    for x = 1:10
        for y = 1:10
            for z = 1:10
                p1index::Int = z + 10 * (y - 1) + 100 * (x - 1)
                if !haskey(connected, p1index)
                    connected[p1index] = []
                end
                p1::Point = cache[x][y][z]
                for x1 = (x - 1):(x+1)
                    if x1 == 11 || x1 <= 0
                        continue
                    end
                    for y1 = (y - 1):(y+1)
                        if y1 == 11 || y1 <= 0
                            continue
                        end
                        for z1 = z:(z+1)
                            if z1 == 11 || z1 <= 0 || (x1 == x && y1 == y && z1 == z)
                                continue
                            end
                            p2index::Int = z1 + 10 * (y1 - 1) + 100 * (x1 - 1)
                            if !haskey(connected, p2index)
                                connected[p2index] = []
                            elseif p2index in connected[p1index]
                                continue
                            end
                            push!(connected[p1index], p2index)
                            push!(connected[p2index], p1index)
                            p2::Point = cache[x1][y1][z1]
                            length::Float64 = sqrt(abs2(p1.x - p2.x) + abs2(p1.y - p2.y) + abs2(p1.z - p2.z))
                            push!(springs, Spring(kSpring, p1index, p2index, length, length))
                        end
                    end
                end
            end
        end
    end
    staticFriction::Float64 = 0.5
    kineticFriction::Float64 = 0.3
    dt::Float64 = 0.0001
    dampening::Float64 = 1 - (dt * 5)
    gravity::Float64 = -9.81

    limit::Float64 = 1
    println("num springs evaluated: ", size(springs)[1])
    println("time multiplier: ",  limit / dt)

    @time begin
        while t < limit
            adjust::Float64 = 1 + sin(t * kOscillationFrequency) * 0.1
            for l in springs
                p1::Point = points[l.p1]
                p2::Point = points[l.p2]
                p1x = p1.x
                p1y = p1.y
                p1z = p1.z
                p2x = p2.x
                p2y = p2.y
                p2z = p2.z
                dist::Float64 = sqrt(abs2(p1x - p2x) + abs2(p1y - p2y) + abs2(p1z - p2z))

                # negative if repelling, positive if attracting
                f::Float64 = l.k * (dist - (l.l0 * adjust))
                # distribute force across the axes
                dx::Float64 = f * (p1x - p2x) / dist
                dy::Float64 = f * (p1y - p2y) / dist
                dz::Float64 = f * (p1z - p2z) / dist

                p1.fx -= dx
                p2.fx += dx

                p1.fy -= dy
                p2.fy += dy

                p1.fz -= dz
                p2.fz += dz
            end

            for p in points
                fy = p.fy
                fx = p.fx
                fz = p.fz
                mass = p.mass

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
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.z += p.vz * dt
            end
            t += dt
        end
    end
end

main()
#@profiler main()
