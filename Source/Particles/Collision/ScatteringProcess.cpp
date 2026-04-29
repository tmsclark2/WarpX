/* Copyright 2021-2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Modern Electron, Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ScatteringProcess.H"

#include "Utils/TextMsg.H"

ScatteringProcess::ScatteringProcess (
                        const std::string& scattering_process,
                        const std::string& cross_section_file,
                        const amrex::ParticleReal energy,
                        const amrex::ParticleReal cutoff_energy)
{   
    // read the cross-section data file into memory
    readCrossSectionFile(cross_section_file, m_energies, m_sigmas_h);

    init(scattering_process, energy, cutoff_energy);
}

template <typename InputVector>
ScatteringProcess::ScatteringProcess (
                        const std::string& scattering_process,
                        const InputVector&& energies,
                        const InputVector&& sigmas,
                        const amrex::ParticleReal energy,
                        const amrex::ParticleReal cutoff_energy)
{
    m_energies.insert(m_energies.begin(), std::begin(energies), std::end(energies));
    m_sigmas_h.insert(m_sigmas_h.begin(), std::begin(sigmas),   std::end(sigmas));

    init(scattering_process, energy, cutoff_energy);
}

void
ScatteringProcess::init (const std::string& scattering_process, const amrex::ParticleReal energy, const amrex::ParticleReal cutof_energy)
{   
    double E_log=20;
    amrex::ParticleReal cutoff_energy = static_cast<amrex::ParticleReal>(E_log);

    using namespace amrex::literals;
    using std::log;
    using std::exp;
    m_exe_h.m_sigmas_data = m_sigmas_h.data();

    // save energy grid parameters for easy use
    m_grid_size = static_cast<int>(m_energies.size());
    m_exe_h.m_energy_lo = m_energies[0];
    m_exe_h.m_energy_hi = m_energies[m_grid_size-1];
    m_exe_h.m_sigma_lo = m_sigmas_h[0];
    m_exe_h.m_sigma_hi = m_sigmas_h[m_grid_size-1];

    // Calcul des pas constant
    m_exe_h.m_dE_log = (log(m_energies[m_grid_size-1]) - log(m_energies[m_grid_size-2]));
    m_exe_h.m_dE_lin = (m_energies[1] - m_energies[0]);

    // Sélection dynamique du dE effectif basé sur le cutoff
    sanityCheckEnergyGrid(m_energies, m_exe_h.m_dE_lin, m_exe_h.m_dE_log, cutoff_energy);
    m_exe_h.m_energy_penalty = energy;
    m_exe_h.m_type = parseProcessType(scattering_process);

    // sanity check cross-section energy grid

    // check that the cross-section is 0 at the energy cost if the energy
    // cost is > 0 - this is to prevent the possibility of negative left
    // over energy after a collision event
    if (m_exe_h.m_energy_penalty > 0) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (getCrossSection(m_exe_h.m_energy_penalty,0.0) == 0),
            "Cross-section > 0 at energy cost for collision."
        );
    }


#ifdef AMREX_USE_GPU
    m_exe_d = m_exe_h;
    m_sigmas_d.resize(m_sigmas_h.size());
    m_exe_d.m_sigmas_data = m_sigmas_d.data();
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_sigmas_h.begin(), m_sigmas_h.end(),
                          m_sigmas_d.begin());
    amrex::Gpu::streamSynchronize();
#endif
}

ScatteringProcessType
ScatteringProcess::parseProcessType(const std::string& scattering_process)
{
    if (scattering_process == "elastic") {
        return ScatteringProcessType::ELASTIC;
    } else if (scattering_process == "back") {
        return ScatteringProcessType::BACK;
    } else if (scattering_process == "charge_exchange") {
        return ScatteringProcessType::TWOPRODUCT_REACTION;
    } else if (scattering_process == "two_product_reaction") {
        return ScatteringProcessType::TWOPRODUCT_REACTION;
    } else if (scattering_process == "ionization") {
        return ScatteringProcessType::IONIZATION;
    } else if (scattering_process.find("excitation") != std::string::npos) {
        return ScatteringProcessType::EXCITATION;
    } else if (scattering_process.find("forward") != std::string::npos) {
        return ScatteringProcessType::FORWARD;
    } else {
        return ScatteringProcessType::INVALID;
    }
}

void
ScatteringProcess::readCrossSectionFile (
                                  const std::string& cross_section_file,
                                  amrex::Vector<amrex::ParticleReal>& energies,
                                  amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas )
{
    std::ifstream infile(cross_section_file);
    if(!infile.is_open()) { WARPX_ABORT_WITH_MESSAGE("Failed to open cross-section data file"); }

    amrex::ParticleReal energy, sigma;
    while (infile >> energy >> sigma) {
        energies.push_back(energy);
        sigmas.push_back(sigma);
    }
    if (infile.bad()) { WARPX_ABORT_WITH_MESSAGE("Failed to read cross-section data from file."); }
    infile.close();
}

void
ScatteringProcess::sanityCheckEnergyGrid (
    const amrex::Vector<amrex::ParticleReal>& energies,
    amrex::ParticleReal dE_line, amrex::ParticleReal dE_log, amrex::ParticleReal cutoff_energy
)
{
    amrex::Print() << cutoff_energy << std::endl;
    for (unsigned i = 1; i < energies.size(); i++) {

        if(energies[i] <= cutoff_energy){
            amrex::Print() << "dans le lin" << energies[i] << std::endl;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (abs(energies[i] - energies[i-1] - dE_line) < dE_line / 100.0),
                "Energy grid not evenly spaced (linear scale at low energy)."
            );
        }
        else{
            amrex::Print() << "dans le log" << energies[i] << std::endl;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (abs(log(energies[i]) - log(energies[i-1]) - dE_log) < abs(dE_log) / 100.0),
                "Energy grid not evenly spaced (logarithmic scale at high energy)."
            );            
        }
    }
}
