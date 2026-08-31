/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_MOLLER_H_
#define WARPX_PARTICLES_COLLISION_MOLLER_H_

#include "Particles/Collision/ScatteringProcess.H"

#include "Particles/Algorithms/KineticEnergy.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <cmath>

/** @file
 *
 * This file contains filter and transform functors for impact ionization
 */

/**
 * \brief Filter functor for impact ionization
 */
class MollerFilterFunc
{
public:

    /**
    * \brief Constructor of the ImpactIonizationFilterFunc functor.
    *
    * This function sample a random number and compares it to the total
    * collision probability to see if the given particle should be considered
    * for an ionization event. If the particle passes this stage the collision
    * cross-section is calculated given its energy and another random number
    * is used to determine whether it actually collides.
    * Note that the mass and energy quantities have to be stored as doubles to
    * ensure accurate energy calculations (otherwise errors occur with single
    * or mixed precision builds of WarpX).
    *
    * @param[in] mcc_process an ScatteringProcess object associated with the ionization
    * @param[in] mass colliding particle's mass (could also assume electron)
    * @param[in] total_collision_prob total probability for a collision to occur
    * @param[in] nu_max maximum collision frequency
    * @param[in] n_a_func ParserExecutor<4> function to get the background
                 density in m^-3 as a function of space and time
    * @param[in] t the current simulation time
    */
    MollerFilterFunc(
        ScatteringProcess const& mcc_process,
        double const mass,
        amrex::ParticleReal const total_collision_prob,
        amrex::ParticleReal const nu_max,
        amrex::ParserExecutor<4> const& n_a_func,
        amrex::Real t
    ) : m_mcc_process(mcc_process.executor()), m_mass(mass),
        m_total_collision_prob(total_collision_prob),
        m_nu_max(nu_max), m_n_a_func(n_a_func), m_t(t) { }

    /**
    * \brief Functor call. This method determines if a given (electron) particle
    * should undergo an ionization collision.
    *
    * @param[in] ptd particle tile data
    * @param[in] i particle index
    * @param[in] engine the random number state and factory
    * @return true if a collision occurs, false otherwise
    */
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (
        const PData& ptd, int const i, amrex::RandomEngine const& engine
    ) const noexcept
    {
        using namespace amrex;
        using std::sqrt;

        // determine if this particle should collide
        if (Random(engine) > m_total_collision_prob) { return false; }

        // get references to the particle to get its position
        const auto& p = ptd.getSuperParticle(i);
        ParticleReal x, y, z;
        double E_coll;
        get_particle_position(p, x, y, z);

        // calculate neutral density at particle location
        const ParticleReal n_a = m_n_a_func(x, y, z, m_t);

        // get the particle velocity
        const ParticleReal ux = ptd.m_rdata[PIdx::ux][i];
        const ParticleReal uy = ptd.m_rdata[PIdx::uy][i];
        const ParticleReal uz = ptd.m_rdata[PIdx::uz][i];
        const ParticleReal u_coll2 = ux*ux + uy*uy + uz*uz;

        // calculate kinetic energy
        constexpr auto eV = PhysConst::q_e;
        E_coll = Algorithms::KineticEnergy<double>(ux, uy, uz, static_cast<amrex::ParticleReal>(m_mass))/eV;
        const amrex::ParticleReal E_min = m_mcc_process.m_energy_penalty;
        amrex::ParticleReal Z_m=15.0; //Valable in the air
        // get collision cross-section
        if (E_coll <= E_min) {
            return false;
        }

        // rest energy in eV, consistent with E_coll's units
        amrex::ParticleReal mc2 = m_mass*PhysConst::c2/eV;
        amrex::ParticleReal E_minus_Emin = E_coll - E_min;
        amrex::ParticleReal mc2_plus_E  = mc2 + E_coll;
        amrex::ParticleReal mc2_plus_E_sq = mc2_plus_E * mc2_plus_E;

        // Terme 1: 1 / E_min - 1 / (E - E_min)
        amrex::ParticleReal term1 = (1.0_prt / E_min) - (1.0 / E_minus_Emin);

        // Terme 2: (E - 2 * E_min) / (2 * (mc^2 + E)^2)
        amrex::ParticleReal term2 = (E_coll - 2.0_prt * E_min) / (2.0 * mc2_plus_E_sq);

        // Terme 3: [mc^2 * (mc^2 + 2 * E) / (E * (mc^2 + E)^2)] * log(E_min / (E - E_min))
        amrex::ParticleReal term3_factor = (mc2 * (mc2 + 2.0 * E_coll)) / (E_coll * mc2_plus_E_sq);
        amrex::ParticleReal term3_log    = std::log(E_min / E_minus_Emin);
        amrex::ParticleReal term3        = term3_factor * term3_log;

        // kappa = 2*pi*r_e^2 * mc^2 * (c/v)^2 ; (c/v)^2 = 1 + c^2/u^2 since
        // u = gamma*v is the code's momentum variable (see KineticEnergy.H)
        const amrex::ParticleReal beta_inv2 = 1.0_prt + PhysConst::c2 / u_coll2;
        const amrex::ParticleReal kappa = 2.0_prt * MathConst::pi * PhysConst::r_e * PhysConst::r_e * mc2 * beta_inv2;

        // Somme complète
        amrex::ParticleReal Sigma_tot = kappa * (term1 + term2 + term3);
        // calculate normalized collision frequency
        const ParticleReal nu_i = Z_m*n_a * Sigma_tot * sqrt(u_coll2) / m_nu_max;

        // check if this collision should be performed
        return (Random(engine) <= nu_i);
    }

private:
    ScatteringProcess::Executor m_mcc_process;
    double m_mass;
    amrex::ParticleReal m_total_collision_prob = 0;
    amrex::ParticleReal m_nu_max;
    amrex::ParserExecutor<4> m_n_a_func;
    amrex::Real m_t;
};


/**
 * \brief Transform functor for impact ionization
 */
class MollerTransformFunc
{
public:

    /**
    * \brief Constructor of the ImpactIonizationTransformFunc functor.
    *
    * The transform is responsible for appropriately decreasing the kinetic
    * energy of the colliding particle and assigning appropriate velocities
    * to the two newly created particles. To this end the energy cost of
    * ionization is passed to the constructor as well as the mass of the
    * colliding species and the standard deviation of the ion velocity
    * (normalized temperature).
    * Note that the mass and energy quantities have to be stored as doubles to
    * ensure accurate energy calculations (otherwise errors occur with single
    * or mixed precision builds of WarpX).
    *
    * @param[in] energy_cost energy cost of ionization
    * @param[in] mass1 mass of the colliding species
    * @param[in] sqrt_kb_m value of sqrt(kB/m), where kB is Boltzmann's constant
                 and m is the background neutral mass
    * @param[in] T_a_func ParserExecutor<4> function to get the background
                 temperature in Kelvin as a function of space and time
    * @param[in] t the current simulation time
    */
    MollerTransformFunc(
        amrex::ParticleReal energy_cost, double mass1, amrex::ParticleReal sqrt_kb_m,
        amrex::ParserExecutor<4> const& T_a_func, amrex::Real t, amrex::ParticleReal E_min
    ) :  m_energy_cost(energy_cost), m_mass1(mass1),
         m_sqrt_kb_m(sqrt_kb_m), m_T_a_func(T_a_func), m_t(t), m_E_min(E_min) { }

    /**
    * \brief Functor call. It determines the properties of the generated pair
    * and decreases the kinetic energy of the colliding particle. Inputs are
    * standard from the FilterCopyTransfrom::filterCopyTransformParticles
    * function.
    *
    * @param[in,out] dst1 target species 1 (electrons)
    * @param[in,out] dst2 target species 2 (ions)
    * @param[in] src source species (electrons)
    * @param[in] i_src particle index of the source species
    * @param[in] i_dst1 particle index of target species 1
    * @param[in] i_dst2 particle index of target species 2
    * @param[in] engine random number generator engine
    */
    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& dst1, SrcData& src,
        int const i_src, int const i_dst1,
        amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex;
        using std::sqrt;

        // get references to the particle to get its position
        const auto& p = src.getSuperParticle(i_src);
        ParticleReal x, y, z;
        double E_coll;
        get_particle_position(p, x, y, z);

        // get references to the original particle's velocity
        auto& ux = src.m_rdata[PIdx::ux][i_src];
        auto& uy = src.m_rdata[PIdx::uy][i_src];
        auto& uz = src.m_rdata[PIdx::uz][i_src];

        // get references to the new particles' velocities
        auto& e_ux = dst1.m_rdata[PIdx::ux][i_dst1];
        auto& e_uy = dst1.m_rdata[PIdx::uy][i_dst1];
        auto& e_uz = dst1.m_rdata[PIdx::uz][i_dst1];

        // calculate kinetic energy
        constexpr auto eV = PhysConst::q_e;
        E_coll = Algorithms::KineticEnergy<double>(ux, uy, uz, m_mass1)/eV;

        amrex::Real X=Random(engine);
        // each electron gets half the energy (could change this later)
        double denominator = E_coll - X * (E_coll - 2.0 * m_E_min);

        const auto E_out = (E_coll * m_E_min) / denominator; //static_cast<amrex::ParticleReal>((E_coll - m_energy_cost) / 2.0_prt * eV);

        // precalculate often used value
        constexpr auto c2 = PhysConst::c2;
        const auto mc2 = m_mass1*c2/eV;

        //const amrex::ParticleReal up = sqrt(E_out * (E_out + 2.0_prt*mc2) / c2) / m_mass1;

        // --- Input ---
        // E0 : Énergie cinétique de l'électron incident (E_inc)
        // E2 : Énergie cinétique de l'électron éjecté / secondaire (E_sec, tiré via X)
        // mc2 : Énergie au repos de l'électron (0.511 MeV)

        // 1. Conservation de l'énergie (E1 = E0 - E2)
        const amrex::ParticleReal E1 = E_coll - E_out; 

        // 2. Normes des quadri-impulsions / impulsions |p| = sqrt(E * (E + 2mc2)) / c
        // (E, mc2 sont en eV ; conversion en Joules via eV = PhysConst::q_e pour obtenir
        // U_norm = p/mass en unites SI, coherent avec KineticEnergy.H)
        const amrex::ParticleReal U1_norm = eV * std::sqrt(E1 * (E1 + 2.0_prt * mc2)) / m_mass1 / PhysConst::c;
        const amrex::ParticleReal U2_norm = eV * std::sqrt(E_out * (E_out + 2.0_prt * mc2)) / m_mass1 / PhysConst::c;

        // 3. Calculs des cosinus des angles d'éjection (Équation 2.19a)
        const amrex::ParticleReal cos_theta1 = std::sqrt((E1 * (E_coll + 2.0_prt * mc2)) / (E_coll * (E1 + 2.0_prt * mc2)));
        const amrex::ParticleReal sin_theta1 = std::sqrt((2.0_prt * mc2 * (E_coll - E1)) / (E_coll * (E1 + 2.0_prt * mc2))); // Eq. 2.19b

        // Pour le deuxième électron (échange des indices 1 -> 2)
        const amrex::ParticleReal cos_theta2 = std::sqrt((E_out * (E_coll + 2.0_prt * mc2)) / (E_coll * (E_out + 2.0_prt * mc2)));
        const amrex::ParticleReal sin_theta2 = std::sqrt((2.0_prt * mc2 * (E_coll - E_out)) / (E_coll * (E_out + 2.0_prt * mc2)));

        // 4. Seul tirage aléatoire : L'angle azimutal phi dans le plan transversal [0, 2pi[
        const amrex::ParticleReal phi2 = 2.0_prt * M_PI * amrex::Random(engine);
        const amrex::ParticleReal phi1 = phi2 + M_PI; // Plan de collision opposé à 180°

        // 5. Composantes locales des impulsions p_local = { p*sin(theta)*cos(phi), p*sin(theta)*sin(phi), p*cos(theta) }
        ux = U1_norm * sin_theta1 * std::cos(phi1);
        uy = U1_norm * sin_theta1 * std::sin(phi1);
        uz = U1_norm * cos_theta1;

        e_ux = U2_norm * sin_theta2 * std::cos(phi2);
        e_uy = U2_norm * sin_theta2 * std::sin(phi2);
        e_uz = U2_norm * cos_theta2;

    }

private:
    amrex::ParticleReal m_energy_cost;
    double m_mass1;
    amrex::ParticleReal m_sqrt_kb_m;
    amrex::ParserExecutor<4> m_T_a_func;
    amrex::Real m_t;
    amrex::ParticleReal m_E_min;
};
#endif // WARPX_PARTICLES_COLLISION_MOLLER_H_
