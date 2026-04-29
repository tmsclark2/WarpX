/* Copyright 2021-2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Modern Electron, Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_
#define WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_

#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_Vector.H>

#include <string>
#include <cmath>

enum class ScatteringProcessType {
    INVALID,
    ELASTIC,
    BACK,
    TWOPRODUCT_REACTION,
    EXCITATION,
    IONIZATION,
    FORWARD,
};

class ScatteringProcess
{
public:
    static bool m_is_log;
    ScatteringProcess (
                const std::string& scattering_process,
                const std::string& cross_section_file,
                amrex::ParticleReal energy,
                amrex::ParticleReal cutoff_energy
                );

    template <typename InputVector>
    ScatteringProcess (
                const std::string& scattering_process,
                const InputVector&& energies,
                const InputVector&& sigmas,
                amrex::ParticleReal energy,
                amrex::ParticleReal cutoff_energy
                );

    ~ScatteringProcess() = default;

    ScatteringProcess (ScatteringProcess const&)            = delete;
    ScatteringProcess& operator= (ScatteringProcess const&) = delete;
    ScatteringProcess (ScatteringProcess &&)                = default;
    ScatteringProcess& operator= (ScatteringProcess &&)     = default;

    /** Read the given cross-section data file to memory.
     *
     * @param cross_section_file the path to the file containing the cross-
     *        section data
     * @param energies vector storing energy values in eV
     * @param sigmas vector storing cross-section values
     *
     */
    static
    void readCrossSectionFile (
                               const std::string& cross_section_file,
                               amrex::Vector<amrex::ParticleReal>& energies,
                               amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas
                               );

    static
    void sanityCheckEnergyGrid (
                                const amrex::Vector<amrex::ParticleReal>& energies,
                                amrex::ParticleReal dE_line, amrex::ParticleReal dE_log, amrex::ParticleReal cutoff_energy
                                );

    struct Executor {
        /** Get the collision cross-section using a simple linear interpolator. If
         * the energy value is lower (higher) than the given energy range the
         * first (last) cross-section value is used.
         *
         * @param E_coll collision energy in eV
         *
         */
        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::ParticleReal getCrossSection (amrex::ParticleReal E_coll, amrex::ParticleReal cutoff_energy) const
{
    if (E_coll <= m_energy_lo) return m_sigma_lo;
    if (E_coll >= m_energy_hi) return m_sigma_hi;

    using amrex::Math::floor;
    using std::log;
    using std::exp;

    // Détermination de la méthode d'interpolation basée sur le seuil d'énergie
    const bool use_log = (E_coll >= cutoff_energy);

    amrex::ParticleReal temp;
    if (use_log) {
        temp = (log(E_coll) - log(m_energy_lo)) / m_dE_log;
    } else {
        temp = (E_coll - m_energy_lo) / m_dE;
    }

    const int idx_1 = static_cast<int>(floor(temp));
    const int idx_2 = idx_1 + 1;
    const amrex::ParticleReal weight = temp - idx_1;

    if (use_log) {
        // Interpolation logarithmique (linéaire dans l'espace log-log)
        amrex::ParticleReal log_sigma = log(m_sigmas_data[idx_1]) 
                                      + weight * (log(m_sigmas_data[idx_2]) - log(m_sigmas_data[idx_1]));
        return exp(log_sigma);
    } else {
        // Interpolation linéaire standard
        return m_sigmas_data[idx_1] + weight * (m_sigmas_data[idx_2] - m_sigmas_data[idx_1]);
    }
}

        amrex::ParticleReal* m_sigmas_data = nullptr;
        amrex::ParticleReal m_energy_lo, m_energy_hi, m_sigma_lo, m_sigma_hi, m_dE, m_dE_log, m_dE_lin;
        amrex::ParticleReal m_energy_penalty;
        ScatteringProcessType m_type;
    };

    [[nodiscard]]
    Executor const& executor () const {
#ifdef AMREX_USE_GPU
        return m_exe_d;
#else
        return m_exe_h;
#endif
    }

    [[nodiscard]] amrex::ParticleReal getCrossSection (amrex::ParticleReal E_coll, amrex::ParticleReal cutoff_energy) const
    {
        return m_exe_h.getCrossSection(E_coll, cutoff_energy);
    }

    [[nodiscard]] amrex::ParticleReal getEnergyPenalty () const { return m_exe_h.m_energy_penalty; }
    [[nodiscard]] amrex::ParticleReal getMinEnergyInput () const { return m_exe_h.m_energy_lo; }
    [[nodiscard]] amrex::ParticleReal getMaxEnergyInput () const { return m_exe_h.m_energy_hi; }
    [[nodiscard]] amrex::ParticleReal getEnergyInputStep () const { return m_exe_h.m_dE; }
    [[nodiscard]] amrex::ParticleReal getEnergyInputStepLog () const { return m_exe_h.m_dE_log; }

    [[nodiscard]] ScatteringProcessType type () const { return m_exe_h.m_type; }

private:

    static
    ScatteringProcessType parseProcessType(const std::string& process);

    void init (const std::string& scattering_process, amrex::ParticleReal energy, amrex::ParticleReal cutoff_energy);

    amrex::Vector<amrex::ParticleReal> m_energies;

#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_sigmas_d;
    Executor m_exe_d;
#endif
    amrex::Gpu::HostVector<amrex::ParticleReal> m_sigmas_h;
    Executor m_exe_h;

    int m_grid_size;
};

#endif // WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_
