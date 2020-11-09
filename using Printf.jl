using Printf

mutable struct Point 
    x::Double # meters
    y::Double # meters
    z::Double # meters
    vx::Double # meters/second
    vy::Double # meters/second
    vz::Double # meters/second
    mass::Double # kg
    fx::Double # N - reset every iteration
    fy::Double # N
    fz::Double # N
end

mutable struct Spring
    k::Double # N/m
    p1::Int # Index of first point
    p2::Int # Index of second point
    l0::Double # meters
    currentl::Double # meters
end

kSpring::Double = 10000.0
kGround::Double = 100000.0
kOscillationFrequency::Double = 10000.0
kUseTetrahedron::bool = false
kUseThousand::bool = false
kDropHeight::Double = 0.2

t::Double = 0.0
points::Array{Point} = []
lines::Array{Spring} = []
cache::Dict{Int64, Dict{Int64, Dict{Int64, Point}}} = Dict()
println(typeof(cache))

x = "Hello World"
println(x)