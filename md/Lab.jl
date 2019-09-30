#module Constants

include("Legacy_Math.jl")

#export c, h, ħ, kB, e, euler, ϵ₀, K, amu2kg, Atom, Barium, Ytterbium
c = 299792458 # m/s
h = 6.626070040e-34 # Js
ħ = h / (2π) # Js
kB = 1.38064852e-23 # J/K
e = 1.6021766208e-19 # C
ϵ₀ = 8.854187817e-12 # F/m
K = 1. / (4π*ϵ₀); # N*(m/C)^2 - coulombs constant

# conversion factors
amu2kg =  1.660539040e-27;

# Elements
struct Atom
    name::String
    symbol::String
    number::Int64
    mass::Float64
    charge::Float64
    cooling::Bool
    Γ::Float64 # natural line width
    cooling_λ::Float64 # cooling laser wavelength
    Ω::Float64 # rabi freq
end

Barium = Atom("Barium", "Ba", 56, 137.327 * amu2kg, e, true, 2π*15e6, 493.4077e-9, 62.1012557471e6);
Ytterbium = Atom("Ytterbium", "Yb", 70, 171. * amu2kg, e, false, -1, -1, -1);

struct Particle
    name::String
    symbol::String
    mass::Float64
    charge::Float64
end;

struct LJParticle
    name::String
    symbol::String
    mass::Float64
    σ::Float64
    ϵ::Float64
end;

struct PaulTrap
    ω₀::Float64 # AC angular freq
    V₀::Float64 # transverse amp
    U₀::Float64 # axial DC
    γ::Float64 # anisotropy factor for transverse (x,y)
    d₀::Float64 # trap size
    #lasers::Float64
end;

function define_paul_trap(f₀, V₀, U₀, γ, d₀)
    return PaulTrap(2π*f₀, V₀, U₀, γ, d₀)
end

function characterize_paul_trap(trap, mass=-1)
    if mass < 0
        mass = Barium.mass
    end
    ax = 8*(1 + trap.γ) * e * trap.U₀ / (mass * trap.d₀^2 * trap.ω₀^2)
    ay = 8*(1 - trap.γ) * e * trap.U₀ / (mass * trap.d₀^2 * trap.ω₀^2)
    az = -16e * trap.U₀ / (mass* trap.d₀^2 * trap.ω₀^2)

    q = qx = qy = - 4e * trap.V₀ / (mass * trap.d₀^2 * trap.ω₀^2)
    qz = -2q
    return ax, ay, az, qx, qy, qz
end

function freq_paul_trap(trap, mass=-1)
    ax, ay, az, qx, qy, qz = characterize_paul_trap(trap, mass)
    βx = sqrt(ax + qx^2/2)
    βy = sqrt(ay + qy^2/2)
    βz = sqrt(az + qz^2/2)
    ωx = βx * trap.ω₀ / 2
    ωy = βy * trap.ω₀ / 2
    ωz = βz * trap.ω₀ / 2
    return ωx/2π, ωy/2π, ωz/2π
end

function build_paul_trap(ωx, ωy, ωz; ω₀=-1, V₀=-1, U₀=-1, d₀=-1, γ=-1)
    @assert (ω₀ + V₀ + U₀ + d₀ + γ >= -3) "Must specify exactly two optional parameters (too few specified)"
    @assert (sum([x==-1 for x in [ω₀, V₀, U₀, d₀, γ]]) == 3) "Must specify exactly two optional parameters (too many specified)"

end

six_dir = cat(eye(3), -eye(3), dims=2);
single_dir = 1/√3 * ones(3,1);

DuanTrap = define_paul_trap(50e6, 90, -1.1, 0.01, 200e-6);

function eq_distance(n, trap) # atmost for linear traps
    N(n) = 1 + n * (-digamma(1) + digamma(n) - 1)
    M(n) = n % 2 == 1 ? (n-1)*n*(n+1) : n^3-n-6

    if n == 1
        return 0.
    elseif n == 2
        return (2K * e * trap.Z^2 / trap.V₁)^(1. /3)
    end

    α = 6K * e * trap.Z^2 / (trap.V₁)
    combo = N(n) / M(n)
    return (α * combo)^(1. /3)
end

# Linear Paul traps
struct LinearTrap
    ω₀::Float64
    V₀::Float64 # transverse amp
    U₀::Float64 # axial DC
    X::Float64
    Y::Float64
    Z::Float64
    ω::Vector{Float64} # secular x,y,z freq
    lasers::Array{Float64,2}
end;


linear_trap_a_param(trap) = (8*e*trap.V₁) / (Barium.mass*trap.ω₀^2*(trap.X^2 + trap.Y^2 + trap.Z^2))
linear_trap_q_param(trap) = (4*e*trap.V₀) / (Barium.mass*trap.ω₀^2*(trap.X^2 + trap.Y^2 + trap.Z^2))

# example traps

LinearUWTrap = LinearTrap(2π*18.2e6, 1000, 10, 0.930e-3, 0.793e-3, 2.84e-3, [0,0,0], six_dir);

#end
