/* Energy compaction resampling (Russian roulette + probabilistic splitting)
 * following Schmalzried, Luque & Lehtinen, Comput. Phys. Commun. 277, 108366 (2022).
 *
 * This file is meant to live in Source/Particles/Resampling/ of WarpX.
 * License: BSD-3-Clause-LBNL (same as WarpX)
 */
#ifndef WARPX_RUSSIAN_ROULETTE_H_
#define WARPX_RUSSIAN_ROULETTE_H_

#include "Resampling.H"

#include <AMReX_REAL.H>

#include <string>

/**
 * \brief This class implements an energy-compaction resampling algorithm.
 *
 * The electron kinetic-energy axis is divided into logarithmic bins between
 * m_energy_min_eV and m_energy_max_eV. In each bin, the number of
 * macroparticles is driven towards a target count m_target_ppb:
 *   - bins with an excess of macroparticles (typically the low-energy bulk)
 *     undergo Russian roulette: each particle survives with probability
 *     r = target/count and its weight is divided by r (conservation in
 *     expectation);
 *   - bins with a deficit of macroparticles (the high-energy tail) undergo
 *     probabilistic splitting: each particle is replaced by
 *     k = floor(r) + Bernoulli(r - floor(r)) copies of weight w/k
 *     (exact conservation).
 *
 * This keeps the number of macroparticles per energy decade roughly constant,
 * i.e. it fixes the spectral resolution of the energy distribution,
 * regardless of how steep the physical spectrum is.
 */
class RussianRoulette final : public ResamplingAlgorithm {
public:

    /**
     * \brief Default constructor
     */
    RussianRoulette () = default;

    /**
     * \brief Constructor: reads the input parameters of the algorithm
     *
     * \param[in] species_name name of the resampled species
     */
    RussianRoulette (const std::string& species_name);

    /**
     * \brief Performs the energy compaction on one particle tile
     *
     * \param[in] geom_lev the geometry at this level (unused)
     * \param[in] pti WarpX particle iterator of the current tile
     * \param[in] lev the MR level
     * \param[in] pc pointer to the particle container
     */
    void operator() (const amrex::Geometry& geom_lev, WarpXParIter& pti, int lev,
                     WarpXParticleContainer* pc) const final;

private:
    /** Lower edge of the energy grid, in eV */
    amrex::Real m_energy_min_eV = 1.e-2;
    /** Upper edge of the energy grid, in eV */
    amrex::Real m_energy_max_eV = 1.e6;
    /** Number of logarithmic bins per energy decade */
    int m_bins_per_decade = 8;
    /** Target number of macroparticles per bin (per tile) */
    amrex::Real m_target_ppb = 1000.;
    /** Maximum splitting factor applied in a single resampling call.
     *  Copies are created identical and only decorrelate through subsequent
     *  stochastic collisions, so large splitting factors should be reached
     *  over several resampling calls rather than at once. */
    int m_max_split = 8;
    /** Minimum macroparticle weight below which splitting is not applied,
     *  to avoid generating particles of degenerate weight. */
    amrex::Real m_min_weight = 1.;
};

#endif // WARPX_RUSSIAN_ROULETTE_H_
