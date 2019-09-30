

function Harmonic!(f, r, t, sim)
    for i=1:sim.N
        f[1,i] -= sim.trap.ωx * r[1,i]
        f[2,i] -= sim.trap.ωy * r[2,i]
        f[3,i] -= sim.trap.ωz * r[3,i]
    end
end

function Damping!(f, ṙ, t, sim, γ)
    for i in eachindex(f)
        f[i] -= γ * ṙ[i]
    end
end

function Random_Heating!(f, ṙ, t, sim)
    for i=1:sim.N
        0
    end
end

struct Cooling_Settings
    no_cooling::Bool
    η::Vector{Float64} # ħ*k*Γ/2 * I_ratio - numerator for each atom
    I_ratios::Vector{Float64} # I/I_sat for each atom
    δ::Vector{Float64} # detuning for each atom
    ks::Vector{Float64} # wave vector magnitude of cooling laser for each atom
end
# pg 180 in Foot
function Scattering_Force!(f, ṙ, t, sim, settings) # laser cooling
    if settings.no_cooling
        return
    end

    speeds = ṙ' * sim.trap.lasers # N x L
    for l=1:num_lasers
        for i=1:sim.N
            if sim.atoms[i].cooling
                mag = settings.η[i] / (1 + settings.I_ratios[i] + 4 *
                    ((settings.δ[i] + settings.ks[i] * speeds[i,l])/sim.atoms[i].Γ)^2)
                for j=1:3
                    f[j,i] += mag * sim.trap.lasers[j,i]
                end
            end
        end
    end
end

function Energy_Coulomb(r, t, sim)
    U = 0
    for i=1:sim.N, j=i+1:sim.N
        d = ((r[1,j]-r[1,i])^2+(r[2,j]-r[2,i])^2+(r[3,j]-r[3,i])^2)^0.5
        U += K * sim.q[i] * sim.q[j] / d
    end
    return U
end;

function Energy_E_Field(r, t, sim)
    U = 0
    AC = sim.trap.V₀ * cos(sim.trap.ω₀ * t) / sim.trap.d₀^2
    DC = sim.trap.U₀ / sim.trap.d₀^2
    for i=1:sim.N
        U += DC * ((1+γ)*r[1,i]^2 + (1-γ)*r[2,i]^2 - 2*r[3,i]^2)
        U += AC * (r[1,i]^2 + r[2,i]^2 - 2*r[3,i]^2)
    end
    return U
end

function Coulomb!(f, r, t, sim)
    for i=1:sim.N, j=i+1:sim.N
        d = ((r[1,j]-r[1,i])^2+(r[2,j]-r[2,i])^2+(r[3,j]-r[3,i])^2)^1.5
        C = K * sim.q[i] * sim.q[j] / d
        for n=1:3
            q = C * (r[n,i] - r[n,j])
            f[n,i] += q
            f[n,j] -= q
        end
    end
end;

function Paul_E_Field!(f, r, t, sim)
    AC = 2 * sim.trap.V₀ * cos(sim.trap.ω₀ * t) / sim.trap.d₀^2
    DC = 2 * sim.trap.U₀ / sim.trap.d₀^2
    for i=1:sim.N
        f[1,i] -= sim.q[i] * r[1,i] * (AC + (1 + sim.trap.γ) * DC)
        f[2,i] -= sim.q[i] * r[2,i] * (AC + (1 - sim.trap.γ) * DC)
        f[3,i] -= sim.q[i] * -2 * r[3,i] * (AC + DC)
    end
end;

function Full_Paul_E_Field!(f, r, t, sim)
    ϕ = cos(sim.trap.ω₀ * t)
    for i=1:sim.N
        f[1,i] += sim.q[i] * r[1,i] * (sim.trap.V₀ / sim.trap.X^2 * ϕ + sim.trap.U₀ / sim.trap.Z^2)
        f[2,i] += sim.q[i] * r[2,i] * (sim.trap.V₀ / sim.trap.Y^2 * -ϕ + sim.trap.U₀ / sim.trap.Z^2)
        f[3,i] += sim.q[i] * -2 * r[3,i] * sim.trap.U₀ / sim.trap.Z^2
    end
end;

function Linear_E_Field!(f, r, t, sim)
    ϕ = cos(sim.trap.ω₀ * t)
    for i=1:sim.N
        f[1,i] += sim.q[i] * r[1,i] * (sim.trap.V₀ / sim.trap.X^2 * ϕ + sim.trap.U₀ / sim.trap.Z^2)
        f[2,i] += sim.q[i] * r[2,i] * (sim.trap.V₀ / sim.trap.Y^2 * -ϕ + sim.trap.U₀ / sim.trap.Z^2)
        f[3,i] += sim.q[i] * -2 * r[3,i] * sim.trap.U₀ / sim.trap.Z^2
    end
end;


#=
# explicitly simulating photon collitions
natural_line_widths = [atom.natural_line_width for atom in config.atoms]
rabi_frequencies = [atom.rabi_freq for atom in config.atoms]
intensity_ratios = 2 .* rabi_frequencies.^2 ./ natural_line_widths.^2
num_lasers = size(config.trap.lasers, 2)
cooling_laser_λ = [atom.cooling_laser_wavelength for atom in config.atoms]
cooling_laser_k = 2π ./ cooling_laser_λ

no_cooling = !any([atom.cooling for atom in config.atoms])
detunings = natural_line_widths ./ 2
pᵧ = h ./ cooling_laser_λ
samples = rand(config.N, num_lasers)
last_event = zeros(config.N, num_lasers)
scatter_rate_constants = natural_line_widths .* intensity_ratios ./ 2
photon_events = zeros(config.N, num_lasers)

function Laser_Cooling!(f, ṙ, t, sim, params) # causes impulse (requires constant timestep)
    if no_cooling
        return
    end

    speeds = ṙ' * sim.trap.lasers # N x L
    for i=1:sim.N
        if sim.atoms[i].cooling
            for l=1:num_lasers
                rate = scatter_rate_constants[i] / (1 + intensity_ratios[i] + 4 *
                    ((detunings[i] + cooling_laser_k[i] * speeds[i,l])/natural_line_widths[i])^2)
                prob = 1. - exp(- rate * (t - last_event[i,l]))
                #println(t, " ", prob)
                if prob - samples[i,l] > 0 # absorption + emission event
                    last_event[i,l] = t
                    samples[i,l] = rand()
                    photon_events[i,l] += 1
                    delta = normalize!(randn(3)) # emission
                    delta += sim.trap.lasers[:,l] # absorption
                    delta .*= pᵧ[i] # magnitude
                    f[:,i] += delta ./ sim.dt
                end
            end
        end
    end
end
=#

# Generalized Coulomb Force (with pbc) (with virial + energy calc)
function coulomb_dynamics!(f, r, q, K, L=0)
    D, N = size(r)
    Ξ = 0. # virial (for pressure measurment)
    U = 0. # potential energy
    for i=1:N, j=i+1:N # all unique pairs
        d = 0. # distance^2
        for k=1:D
            Δ = r[k,j] - r[k,i]
            if L>0
                Δ = abs(Δ) > L/2 ? Δ - L : Δ # enforce pbc
            end
            d += Δ^2
        end
        C = K * q[i] * q[j]
        U += C / d^0.5
        C /= d^1.5
        for k=1:D
            Δ = r[k,i] - r[k,j]
            if L>0
                Δ = abs(Δ) < L/2 ? Δ : (Δ > L/2 ? Δ - L : Δ + L) # enforce pbc
            end
            fk = C * Δ
            f[k,i] += fk
            f[k,j] -= fk
            Ξ += fk * Δ # f*r
        end
    end
    return U, -Ξ / 2.
end;

# Generalized Coulomb Force (with pbc) (with virial + energy calc)
function lj_dynamics!(f, r, σs, ϵs, L=0)
    D, N = size(r)
    Ξ = 0. # virial (for pressure measurment)
    U = 0. # potential energy
    for i=1:N, j=i+1:N # all unique pairs
        d² = 0. # distance^2
        for k=1:D
            Δ = r[k,j] - r[k,i]
            if L>0
                Δ = abs(Δ) > L/2 ? Δ - L : Δ # enforce pbc
            end
            d² += Δ^2
        end
        d = √d²
        σ = (σs[i] + σs[j]) / 2 # Lorentz-Berthelot rules
        ϵ = √(ϵs[i] * ϵs[j]) # Lorentz-Berthelot rules
        r6 = (σ/d)^6
        U += 4*ϵ*(r6^2 - r6)
        F = 48*ϵ/d² * (r6^2 - r6/2)
        for k=1:D
            Δ = r[k,i] - r[k,j]
            if L>0
                Δ = abs(Δ) < L/2 ? Δ : (Δ > L/2 ? Δ - L : Δ + L) # enforce pbc
            end
            fk = F * Δ
            f[k,i] += fk
            f[k,j] -= fk
            Ξ += fk * Δ # f*r
        end
    end
    return U, -Ξ / 2
end;
