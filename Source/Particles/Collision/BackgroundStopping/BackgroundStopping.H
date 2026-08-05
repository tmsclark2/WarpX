/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_BACKGROUNSTOPPING_H_
#define WARPX_PARTICLES_COLLISION_BACKGROUNSTOPPING_H_

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"

#include <AMReX_REAL.H>

#include <string>

enum class BackgroundStoppingType {
    INVALID,
    ELECTRONS,
    IONS,
    NEUTRAL,
};

class BackgroundStopping final
    : public CollisionBase
{
public:
    explicit BackgroundStopping (const std::string& collision_name);

    ~BackgroundStopping () override = default;

    BackgroundStopping ( BackgroundStopping const &)             = delete;
    BackgroundStopping& operator= ( BackgroundStopping const & ) = delete;
    BackgroundStopping ( BackgroundStopping&& )                  = delete;
    BackgroundStopping& operator= ( BackgroundStopping&& )       = delete;

    /** Perform the stopping calculation
     *
     * @param cur_time Current time
     * @param dt Time step size
     * @param mypc Container of species involved
     *
     */
    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override;

    /** Perform the stopping calculation within a tile for stopping on electrons
     *
     * @param pti particle iterator
     * @param dt time step size
     * @param t current simulation time
     * @param species_mass mass of the active species
     * @param species_charge charge of the active species
     *
     */
    void doBackgroundStoppingOnElectronsWithinTile (WarpXParIter& pti, amrex::Real dt, amrex::Real t,
                                                    amrex::ParticleReal species_mass, amrex::ParticleReal species_charge);

    /** Perform the stopping calculation within a tile for stopping on ions
     *
     * @param pti particle iterator
     * @param dt time step size
     * @param t current simulation time
     * @param species_mass mass of the active species
     * @param species_charge charge of the active species
     *
     */
    void doBackgroundStoppingOnIonsWithinTile (WarpXParIter& pti, amrex::Real dt, amrex::Real t,
                                               amrex::ParticleReal species_mass, amrex::ParticleReal species_charge);
    /** Perform the stopping calculation within a tile for stopping on a neutral gas
     *
     * Dynamic-friction (Bethe) energy loss by excitation and ionization,
     * Bethe & Ashkin 1953 p. 254 — eq. (2.2), and its restricted form eq. (2.8).
     *
     * @param pti particle iterator
     * @param dt time step size
     * @param t current simulation time
     * @param species_mass mass of the active species
     * @param species_charge charge of the active species
     *
     */
    void doBackgroundStoppingOnNeutralGasWithinTile (WarpXParIter& pti, amrex::Real dt, amrex::Real t,
                                                     amrex::ParticleReal species_mass, amrex::ParticleReal species_charge);
private:

    amrex::ParticleReal m_background_mass;
    amrex::ParticleReal m_background_charge_state;
    BackgroundStoppingType m_background_type;

    // Neutral-gas (Bethe) stopping parameters
    amrex::ParticleReal m_background_Z = 1.0;   // Z_m : electrons per molecule
    amrex::ParticleReal m_E_ion        = 0.0;   // ℰ_ion : ionization energy
    amrex::ParticleReal m_E_min        = 0.0;   // ℰ_min : exclusion energy (eq. 2.8)

    amrex::Parser m_background_density_parser;
    amrex::Parser m_background_temperature_parser;

    amrex::ParserExecutor<4> m_background_density_func;
    amrex::ParserExecutor<4> m_background_temperature_func;

};

#endif // WARPX_PARTICLES_COLLISION_BACKGROUNSTOPPING_H_
