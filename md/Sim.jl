<<<<<<< HEAD
#module Sim

#export SimInfo, calcTemp

#include("Lab.jl")

#using Constants, IonTraps

struct SimInfo # for trapped ions
    trap::PaulTrap
    atoms::Vector{Atom}
    N::Int64
    m::Vector{Float64}
    q::Vector{Float64}
end;

struct EnsembleSim # simulates NVT, NPT, or NVE ensembles (using Berendsen baro and/or thermostat)
    N::Int64 # number of atoms
    D::Int64 # number of spatial dimensions
    δt::Float64 # timestep
    K::Float64 # coulomb constant
    Lᵣ::Float64 # reference cell length (cubed root of Volume)
    Pᵣ::Float64 # reference pressure
    τₚ::Float64 # barostat coupling
    Tᵣ::Float64 # reference temperature
    τₜ::Float64 # thermostat coupling
    m::Vector{Float64} # masses
    q::Vector{Float64} # charge
end;

function scaleAdd!(X, a, Y) # X += a*Y
    for i in eachindex(X)
        X[i] += a * Y[i]
    end
end;

function calc_Eₖ(ṙ, m)
    D, N = size(ṙ)
    Σmv² = 0.
    for i=1:N, k=1:D
        Σmv² += m[i] * ṙ[k,i]^2
    end
    return Σmv² / 2.
end

function simStep!(ṙ, r, t, L, info) # as written in Berendsen paper
    D,N = size(ṙ)
    r̈ = zeros(D,N)
    δt = info.δt

    # compute force (and virial)
    U, Ξ = dynamics!(r̈, r, info.q, info.K, L)

    # compute acc
    for i=1:sim.N
        r̈[:,i] ./= info.m[i]
    end

    # compute kinetic energy
    Eₖ = calc_Eₖ(ṙ, info.m)

    # compute pressure from L, Eₖ and Ξ
    P = (Eₖ - Ξ) / (D*L^D)

    # compute barostat term
    if info.Pᵣ > 0
        μ = (1 + δt/info.τₚ*(P - info.Pᵣ))^(1/D)
    end

    # compute temperature from Eₖ
    T = 2*Eₖ/kB/(D*N-3)

    # compute thermostat term
    if info.Tᵣ > 0
        λ = √(1 + (δt/info.τₜ)*(info.Tᵣ/T - 1))
    end

    # integrate vel
    scaleAdd!(ṙ, δt, r̈)

    # scale velocities
    if info.Tᵣ > 0
        ṙ .*= λ
    end

    # integrate pos
    scaleAdd!(r, δt, ṙ)

    # scale pos
    if info.Pᵣ > 0
        r .*= μ
        L *= μ
        for i in eachindex(r) # enforce pbc
            if r[i] > L
                r[i] -= L
            elseif r[i] < 0
                r[i] += L
            end
        end
    end

    return t+δt, L, Eₖ, U, T, P
end

function vvSimStep!(ṙ, r, t, L, info) # as written in Berendsen paper
    D,N = size(ṙ)
    r̈ = zeros(D,N)
    δt = info.δt

    # compute force, potential energy (and virial)
    U, Ξ = dynamics!(r̈, r, info.q, info.K, L)

    # compute kinetic energy
    Eₖ = calc_Eₖ(ṙ, info.m)

    # compute pressure and temperature
    P = (Eₖ - Ξ) / (D*L^D)
    T = 2*Eₖ/kB/(D*(N-1))

    # compute acc
    for i=1:N
        r̈[:,i] ./= info.m[i]
    end

    # integrate vel
    scaleAdd!(ṙ, δt, r̈)

    # apply thermostat
    if info.Tᵣ > 0
        λ = √(1 + (δt/info.τₜ)*(info.Tᵣ/T - 1))
        ṙ .*= λ
    end

    # integrate pos
    scaleAdd!(r, δt, ṙ)

    # apply barostat
    if info.Pᵣ > 0
        μ = (1 + δt/info.τₚ*(P - info.Pᵣ))^(1/D)
        r .*= μ
        L *= μ
    end

    for i in eachindex(r) # enforce pbc
        r[i] %= L
    end

    return t+δt, L, Eₖ, U, T, P
end

function vvSimRun!(ṙ, r, t, info, L=0., n=1)
    L = L > 0 ? L : info.Lᵣ
    Eₖ, U, T, P = 0, 0, 0, 0
    for i=1:n
        t, L, Eₖ, U, T, P = vvSimStep!(ṙ, r, t, L, info)
    end
    return t, Eₖ, U, L, T, P
end

function vvStep!(r̈, ṙ, r, t, δt, dynamics!) # Velocity Verlet Integration Step (inplace updates pos, vel, acc)
    scaleAdd!(ṙ, δt/2, r̈)   # update vel from previous acc (half step)
    scaleAdd!(r, δt, ṙ)     # update pos from current vel
    t += δt                 # increment time
    dynamics!(r̈, ṙ, r, t)   # compute acc from current pos and vel
    scaleAdd!(ṙ, δt/2, r̈)   # update vel from current acc (half step)
    return r̈, ṙ, r, t
end;

function calcTemp(ṙ, sim) # 3 x N
    N = size(ṙ,2)
    Σmv² = 0.
    for i=1:N
        Σmv² += sim.m[i] * (ṙ[1,i]^2 + ṙ[2,i]^2 + ṙ[3,i]^2)
    end
    return Σmv² / (3N*kB)
end;

function gridPos(N, D, L)
    @assert D == 2

    S = ceil(N^(1/D))

    dx = L / S

    pos = hcat(map(collect, Base.product([0:dx:L for _=1:D]...))...)
    pos .+= dx/2
    return pos

end
=======
#module Sim

#export SimInfo, calcTemp

#include("Lab.jl")

#using Constants, IonTraps

struct SimInfo # for trapped ions
    trap::PaulTrap
    atoms::Vector{Atom}
    N::Int64
    m::Vector{Float64}
    q::Vector{Float64}
end;

struct EnsembleSim # simulates NVT, NPT, or NVE ensembles (using Berendsen baro and/or thermostat)
    N::Int64 # number of atoms
    D::Int64 # number of spatial dimensions
    δt::Float64 # timestep
    K::Float64 # coulomb constant
    Lᵣ::Float64 # reference cell length (cubed root of Volume)
    Pᵣ::Float64 # reference pressure
    τₚ::Float64 # barostat coupling
    Tᵣ::Float64 # reference temperature
    τₜ::Float64 # thermostat coupling
    m::Vector{Float64} # masses
    q::Vector{Float64} # charge
end;

struct LJEnsembleSim # simulates NVT, NPT, or NVE ensembles (using Berendsen baro and/or thermostat)
    N::Int64 # number of atoms
    D::Int64 # number of spatial dimensions
    δt::Float64 # timestep
    Lᵣ::Float64 # reference cell length (cubed root of Volume)
    Pᵣ::Float64 # reference pressure
    τₚ::Float64 # barostat coupling
    Tᵣ::Float64 # reference temperature
    τₜ::Float64 # thermostat coupling
    m::Vector{Float64} # masses
    σ::Vector{Float64} # LJ distance
    ϵ::Vector{Float64} # LJ well depth
end;

function scaleAdd!(X, a, Y) # X += a*Y
    for i in eachindex(X)
        X[i] += a * Y[i]
    end
end;

function calc_Eₖ(ṙ, m)
    D, N = size(ṙ)
    Σmv² = 0.
    for i=1:N, k=1:D
        Σmv² += m[i] * ṙ[k,i]^2
    end
    return Σmv² / 2.
end

function simStep!(ṙ, r, t, L, info) # as written in Berendsen paper
    D,N = size(ṙ)
    r̈ = zeros(D,N)
    δt = info.δt

    # compute force (and virial)
    U, Ξ = dynamics!(r̈, r, info.q, info.K, L)

    # compute acc
    for i=1:sim.N
        r̈[:,i] ./= info.m[i]
    end

    # compute kinetic energy
    Eₖ = calc_Eₖ(ṙ, info.m)

    # compute pressure from L, Eₖ and Ξ
    P = (Eₖ - Ξ) / (D*L^D)

    # compute barostat term
    if info.Pᵣ > 0
        μ = (1 + δt/info.τₚ*(P - info.Pᵣ))^(1/D)
    end

    # compute temperature from Eₖ
    T = 2*Eₖ/kB/(D*N-3)

    # compute thermostat term
    if info.Tᵣ > 0
        λ = √(1 + (δt/info.τₜ)*(info.Tᵣ/T - 1))
    end

    # integrate vel
    scaleAdd!(ṙ, δt, r̈)

    # scale velocities
    if info.Tᵣ > 0
        ṙ .*= λ
    end

    # integrate pos
    scaleAdd!(r, δt, ṙ)

    # scale pos
    if info.Pᵣ > 0
        r .*= μ
        L *= μ
        for i in eachindex(r) # enforce pbc
            if r[i] > L
                r[i] -= L
            elseif r[i] < 0
                r[i] += L
            end
        end
    end

    return t+δt, L, Eₖ, U, T, P
end

function vvSimStep!(ṙ, r, t, L, info)
    D,N = size(ṙ)
    r̈ = zeros(D,N)
    δt = info.δt

    # compute force, potential energy (and virial)
    U, Ξ = lj_dynamics!(r̈, r, info.σ, info.ϵ, L)

    # compute kinetic energy
    Eₖ = calc_Eₖ(ṙ, info.m)

    # compute pressure and temperature
    P = (Eₖ - Ξ) / (D*L^D)
    T = 2*Eₖ/kB/(D*(N-1))

    # compute acc
    for i=1:N
        r̈[:,i] ./= info.m[i]
    end

    @show r̈

    # integrate vel
    scaleAdd!(ṙ, δt, r̈)

    # apply thermostat
    if info.Tᵣ > 0
        λ = √(1 + (δt/info.τₜ)*(info.Tᵣ/T - 1))
        ṙ .*= λ
    end

    # integrate pos
    scaleAdd!(r, δt, ṙ)

    # apply barostat
    if info.Pᵣ > 0
        μ = (1 + δt/info.τₚ*(P - info.Pᵣ))^(1/D)
        r .*= μ
        L *= μ
    end

    pbc!(r,L)

    return t+δt, L, Eₖ, U, T, P
end

function pbc!(r, L)
    for i=eachindex(r)
        if r[i] > L/2
            r[i] -= L
        elseif r[i] < -L/2
            r[i] += L
        end
        #r[i] = mod(r[i], L)
    end
    r
end

function vvSimRun!(ṙ, r, t, info, L=0., n=1)
    L = L > 0 ? L : info.Lᵣ
    Eₖ, U, T, P = 0, 0, 0, 0
    for i=1:n
        t, L, Eₖ, U, T, P = vvSimStep!(ṙ, r, t, L, info)
    end
    return t, Eₖ, U, L, T, P
end

function vvStep!(r̈, ṙ, r, t, δt, dynamics!) # Velocity Verlet Integration Step (inplace updates pos, vel, acc)
    scaleAdd!(ṙ, δt/2, r̈)   # update vel from previous acc (half step)
    scaleAdd!(r, δt, ṙ)     # update pos from current vel
    t += δt                 # increment time
    dynamics!(r̈, ṙ, r, t)   # compute acc from current pos and vel
    scaleAdd!(ṙ, δt/2, r̈)   # update vel from current acc (half step)
    return r̈, ṙ, r, t
end;

function calcTemp(ṙ, sim) # 3 x N
    N = size(ṙ,2)
    Σmv² = 0.
    for i=1:N
        Σmv² += sim.m[i] * (ṙ[1,i]^2 + ṙ[2,i]^2 + ṙ[3,i]^2)
    end
    return Σmv² / (3N*kB)
end;

function gridPos(N, D, L)
    #@assert D == 2

    S = ceil(N^(1/D))

    dx = L / S

    pos = hcat(map(collect, Base.product([0:dx:L-dx/2 for _=1:D]...))...)
    #pos .+= dx/2
    #pos .%= L
    return pos[:,1:N]

end
>>>>>>> 2096a344ef6377844ae82703b89c9cea856661c9
