/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BackgroundMCCCollision.H"

#include "ImpactIonization.H"
#include "ImpactPhotoIonization.H"
#include "ImpactAttachment.H"
#include "ImpactThreeBodyAttachment.H"
#include "Particles/Algorithms/KineticEnergy.H"
#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Collision/BinaryCollision/TwoProductUtil.H"
#include "Particles/ParticleCreation/FilterCopyTransform.H"
#include "Particles/ParticleCreation/FilterCopyTransformCreate.H"
#include "Particles/ParticleCreation/FilterCopyTransformDelete.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>

BackgroundMCCCollision::BackgroundMCCCollision(std::string const& collision_name)
    : CollisionBase(collision_name)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_species_names.size() == 1,
                                     "Background MCC must have exactly one species.");

    const amrex::ParmParse pp_collision_name(collision_name);

    amrex::ParticleReal background_density = 0;
    if (utils::parser::queryWithParser(pp_collision_name, "background_density", background_density)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (background_density > 0),
            "The background density must be greater than 0.");
        m_background_density_parser =
            utils::parser::makeParser(
                std::to_string(background_density), {"x", "y", "z", "t"});
    }
    else {
        std::string background_density_str;
        utils::parser::Store_parserString(pp_collision_name, "background_density(x,y,z,t)", background_density_str);
        m_background_density_parser =
            utils::parser::makeParser(background_density_str, {"x", "y", "z", "t"});
    }

    amrex::ParticleReal background_temperature;
    if (utils::parser::queryWithParser(pp_collision_name, "background_temperature", background_temperature)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (background_temperature >= 0), "The background temperature must be positive."
        );
        m_background_temperature_parser =
            utils::parser::makeParser(std::to_string(background_temperature), {"x", "y", "z", "t"});
    }
    else {
        std::string background_temperature_str;
        utils::parser::Store_parserString(pp_collision_name, "background_temperature(x,y,z,t)", background_temperature_str);
        m_background_temperature_parser =
            utils::parser::makeParser(background_temperature_str, {"x", "y", "z", "t"});
    }

    // compile parsers for background density and temperature
    m_background_density_func = m_background_density_parser.compile<4>();
    m_background_temperature_func = m_background_temperature_parser.compile<4>();

    utils::parser::queryWithParser(
        pp_collision_name, "max_background_density", m_max_background_density);
    // if the background density is constant we can use that number to calculate
    // the maximum collision probability, if `max_background_density` was not
    // specified
    if (m_max_background_density == 0 && background_density != 0) {
        m_max_background_density = background_density;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_max_background_density > 0),
        "The maximum background density must be greater than 0."
    );

    // if the neutral mass is specified use it, but if ionization is
    // included the mass of the secondary species of that interaction
    // will be used. If no neutral mass is specified and ionization is not
    // included the mass of the colliding species will be used
    m_background_mass = -1;
    utils::parser::queryWithParser(
        pp_collision_name, "background_mass", m_background_mass);

    // Parse the list of scattering processes (these could be elastic,
    // excitation, charge_exchange, etc.) and create a vector of
    // ScatteringProcess objects from each scattering process name.
    amrex::Vector<ScatteringProcess> scattering_processes = BinaryCollisionUtils::parse_scattering_processes(collision_name);

    for (auto& process : scattering_processes) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(process.type() != ScatteringProcessType::INVALID,
                                         "Cannot add an unknown scattering process type");

        // if the scattering process is ionization get the secondary species
        // only one ionization process is supported, the vector
        // m_ionization_processes is only used to make it simple to calculate
        // the maximum collision frequency with the same function used for
        // particle conserving processes
        if (process.type() == ScatteringProcessType::IONIZATION) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!ionization_flag,
                                             "Background MCC only supports a single ionization process");
            ionization_flag = true;
            pp_collision_name.query("do_photoionization", photo_ionization_flag);

            std::string secondary_species;
            std::string photoioni_species;
            pp_collision_name.get("ionization_species", secondary_species);
            m_species_names.push_back(secondary_species);
            if (photo_ionization_flag) {
                pp_collision_name.get("photoionization_species", photoioni_species);
                m_photo_species_name = photoioni_species;
                amrex::Vector<double> frequencies_photo;
                pp_collision_name.getarr("frequencies_photo", frequencies_photo);
                f1 = frequencies_photo[0];
                f2 = frequencies_photo[1];

                pp_collision_name.get("O2_pressure", PO2);
                pp_collision_name.get("quenching_ratio", pq);
                pp_collision_name.get("electric_field_ratio", q_E);
                pp_collision_name.get("total_pressure", tot_press);

                K1 = 3.5 * PO2; // Torr-1 m^-1
                K2 = 200.0 * PO2; // Torr-1 m^-1
   
                q_press = pq/(tot_press+pq); 

                total_collision_prob_photo = q_press*q_E;
            }

            m_ionization_processes.push_back(std::move(process));
        }  
        else if (process.type() == ScatteringProcessType::ATTACHMENT) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!attachment_flag,
                                             "Background MCC only supports a single attachment process");
            attachment_flag = true;
            // doBackgroundAttachment() only ever creates a particle in the
            // first species -- any additional name is parsed but unused, so
            // only read the first one here.
            amrex::Vector<std::string> attachment_species;
            pp_collision_name.getarr("attachment_species", attachment_species);
            m_species_names.push_back(attachment_species[0]);

            m_attachment_processes.push_back(std::move(process));
        }
        else if (process.type() == ScatteringProcessType::THREE_BODY) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!three_body_attachment_flag,
                                             "Background MCC only supports a single three-body attachment process");
            three_body_attachment_flag = true;
            amrex::ParticleReal background_density_2 = 0;
            if (utils::parser::queryWithParser(pp_collision_name, "background_density_second_specie", background_density_2)) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (background_density_2 > 0),
                "The background density must be greater than 0.");
                m_background_density_parser_2 = utils::parser::makeParser(
                    std::to_string(background_density_2), {"x", "y", "z", "t"});
                utils::parser::makeParser(
                    std::to_string(background_density_2), {"x", "y", "z", "t"});
                }
            else {
                std::string background_density_str_2;
                utils::parser::Store_parserString(pp_collision_name, "background_density_second_specie(x,y,z,t)", background_density_str_2);
                m_background_density_parser_2 =
                utils::parser::makeParser(background_density_str_2, {"x", "y", "z", "t"});
                utils::parser::queryWithParser(
                    pp_collision_name, "max_background_density_second_specie", m_max_background_density_2);
                // if the background density is constant we can use that number to calculate
                // the maximum collision probability, if `max_background_density` was not
                // specified
                }
                if (m_max_background_density_2 == 0 && background_density_2 != 0) {
                    m_max_background_density_2 = background_density_2;
                }
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (m_max_background_density_2 > 0),
                    "The maximum background density of second species must be greater than 0."
                );
            // doBackgroundThreeBodyAttachment() only ever creates a particle
            // in the first species -- any additional name is parsed but
            // unused, so only read the first one here.
            amrex::Vector<std::string> three_body_attachment_species;
            pp_collision_name.getarr("three_body.species", three_body_attachment_species);
            m_species_names.push_back(three_body_attachment_species[0]);

            m_three_body_attachment_processes.push_back(std::move(process));
        }
        else {
            m_scattering_processes.push_back(std::move(process));
        }
    }

#ifdef AMREX_USE_GPU
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_scattering_processes_exe;
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_ionization_processes_exe;
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_attachment_processes_exe;
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_three_body_attachment_processes_exe;
    for (auto const& p : m_scattering_processes) {
        h_scattering_processes_exe.push_back(p.executor());
    }
    for (auto const& p : m_ionization_processes) {
        h_ionization_processes_exe.push_back(p.executor());
    }    
    for (auto const& p : m_attachment_processes) {
        h_attachment_processes_exe.push_back(p.executor());
        //amrex::Print() << "les différents attachment"  << p.executor() << std::endl;

    }
    for (auto const& p : m_three_body_attachment_processes) {
        h_three_body_attachment_processes_exe.push_back(p.executor());
    }
    m_scattering_processes_exe.resize(h_scattering_processes_exe.size());
    m_ionization_processes_exe.resize(h_ionization_processes_exe.size());
    m_attachment_processes_exe.resize(h_attachment_processes_exe.size());
    m_three_body_attachment_processes_exe.resize(h_three_body_attachment_processes_exe.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_scattering_processes_exe.begin(),
                          h_scattering_processes_exe.end(), m_scattering_processes_exe.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_ionization_processes_exe.begin(),
                          h_ionization_processes_exe.end(), m_ionization_processes_exe.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_attachment_processes_exe.begin(),
                          h_attachment_processes_exe.end(), m_attachment_processes_exe.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_three_body_attachment_processes_exe.begin(),
                          h_three_body_attachment_processes_exe.end(), m_three_body_attachment_processes_exe.begin());

    amrex::Gpu::streamSynchronize();
#else
    for (auto const& p : m_scattering_processes) {
        m_scattering_processes_exe.push_back(p.executor());
    }
    for (auto const& p : m_ionization_processes) {
        m_ionization_processes_exe.push_back(p.executor());
    }
    for (auto const& p : m_attachment_processes) {
        m_attachment_processes_exe.push_back(p.executor());
    }
    for (auto const& p : m_three_body_attachment_processes) {
        m_three_body_attachment_processes_exe.push_back(p.executor());
    }
#endif
}

/** Calculate the maximum collision frequency using a fixed energy grid that
 *  ranges from 1e-4 to 5000 eV in 0.2 eV increments
 */
amrex::ParticleReal
BackgroundMCCCollision::get_nu_max(amrex::Vector<ScatteringProcess> const& mcc_processes) const
{
    using namespace amrex::literals;
    amrex::ParticleReal nu, nu_max = 0.0;
    amrex::ParticleReal E_start = 1e-4_prt;
    amrex::ParticleReal E_end = 5000._prt;
    amrex::ParticleReal E_step = 0.2_prt;

    // set the energy limits and step size for calculating nu_max based
    // on the given cross-section inputs
    for (const auto &process : mcc_processes) {
        auto energy_lo = process.getMinEnergyInput();
        E_start = (energy_lo < E_start) ? energy_lo : E_start;
        auto energy_hi = process.getMaxEnergyInput();
        E_end = (energy_hi > E_end) ? energy_hi : E_end;
        auto energy_step = process.getEnergyInputStep();
        E_step = (energy_step < E_step) ? energy_step : E_step;
    }

    amrex::ParticleReal E = E_start;
    while(E < E_end){
        amrex::ParticleReal sigma_E = 0.0;

        // loop through all collision pathways
        for (const auto &scattering_process : mcc_processes) {
            // get collision cross-section
            sigma_E += scattering_process.getCrossSection(E);
        }

        // calculate collision frequency
        nu = (
              m_max_background_density
              * std::sqrt(2.0_prt / m_mass1 * PhysConst::q_e)
              * sigma_E * std::sqrt(E)
              );
        nu_max = std::max(nu_max, nu);
        E+=E_step;
    }
    return nu_max;
}

/** Calculate the maximum collision frequency using a fixed energy grid that
 *  ranges from 1e-4 to 5000 eV in 0.2 eV increments
 */
amrex::ParticleReal
BackgroundMCCCollision::get_nu_max_threebody(amrex::Vector<ScatteringProcess> const& mcc_processes) const
{
    using namespace amrex::literals;
    amrex::ParticleReal nu, nu_max = 0.0;
    amrex::ParticleReal E_start = 1e-4_prt;
    amrex::ParticleReal E_end = 5000._prt;
    amrex::ParticleReal E_step = 0.2_prt;

    // set the energy limits and step size for calculating nu_max based
    // on the given cross-section inputs
    for (const auto &process : mcc_processes) {
        auto energy_lo = process.getMinEnergyInput();
        E_start = (energy_lo < E_start) ? energy_lo : E_start;
        auto energy_hi = process.getMaxEnergyInput();
        E_end = (energy_hi > E_end) ? energy_hi : E_end;
        auto energy_step = process.getEnergyInputStep();
        E_step = (energy_step < E_step) ? energy_step : E_step;
    }

    amrex::ParticleReal E = E_start;
    while(E < E_end){
        amrex::ParticleReal sigma_E = 0.0;

        // loop through all collision pathways
        for (const auto &scattering_process : mcc_processes) {
            // get collision cross-section
            sigma_E += scattering_process.getCrossSection(E);
        }

        // calculate collision frequency
        nu = (
              m_max_background_density*m_max_background_density_2
              * std::sqrt(2.0_prt / m_mass1 * PhysConst::q_e)
              * sigma_E * std::sqrt(E)*1e-6
              );
        if (nu > nu_max) {
            nu_max = nu;
        }

        E+=E_step;
    }
    return nu_max;
}

void
BackgroundMCCCollision::doCollisions(amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc)
{
    ABLASTR_PROFILE("BackgroundMCCCollision::doCollisions()");
    using namespace amrex::literals;
    auto& species1 = mypc->GetParticleContainerFromName(m_species_names[0]);
    // this is a very ugly hack to have species2 be a reference and be
    // defined in the scope of doCollisions
    auto& species2 = (
                      (m_species_names.size() >= 2) ?
                      mypc->GetParticleContainerFromName(m_species_names[1]) :
                      mypc->GetParticleContainerFromName(m_species_names[0])
                      );
    if (!init_flag) {
        m_mass1 = species1.getMass();

        // calculate maximum collision frequency without ionization
        m_nu_max = get_nu_max(m_scattering_processes);

        // calculate total collision probability
        auto coll_n = m_nu_max * dt;
        m_total_collision_prob = 1.0_prt - std::exp(-coll_n);

        // dt has to be small enough that a linear expansion of the collision
        // probability is sufficiently accurately, otherwise the MCC results
        // will be very heavily affected by small changes in the timestep
        if (coll_n > 0.1_prt) {
            ablastr::warn_manager::WMRecordWarning("BackgroundMCC Collisions",
                     "dt is too large to ensure accurate MCC results , coll_n: " +
                      std::to_string(coll_n) + " is > 0.1 and collision probability is = " +
                      std::to_string(m_total_collision_prob) + "\n");
        }

        if (ionization_flag) {
            // calculate maximum collision frequency for ionization
            m_nu_max_ioniz = get_nu_max(m_ionization_processes);

            // calculate total ionization probability
            auto coll_n_ioniz = m_nu_max_ioniz * dt;
            m_total_collision_prob_ioniz = 1.0_prt - std::exp(-coll_n_ioniz);

            if (coll_n_ioniz > 0.1_prt) {
                ablastr::warn_manager::WMRecordWarning("BackgroundMCC Collisions",
                         "dt is too large to ensure accurate MCC ionization , coll_n_ionization: " +
                          std::to_string(coll_n_ioniz) + " is > 0.1 and ionization probability is = " +
                          std::to_string(m_total_collision_prob_ioniz) + "\n");
            }
            
            // if an ionization process is included the secondary species mass
            // is taken as the background mass
            m_background_mass = species2.getMass();
        }
        else if (attachment_flag) {
            // calculate maximum collision frequency for attachment
            m_nu_max_attach = get_nu_max(m_attachment_processes);

            // calculate total attachment probability
            auto coll_n_attach = m_nu_max_attach * dt;
            m_total_collision_prob_attach = 1.0_prt - std::exp(-coll_n_attach);

            if (coll_n_attach > 0.1_prt) {
                ablastr::warn_manager::WMRecordWarning("BackgroundMCC Collisions",
                         "dt is too large to ensure accurate MCC attachment , coll_n_attachment: " +
                          std::to_string(coll_n_attach) + " is > 0.1 and attachment probability is = " +
                          std::to_string(m_total_collision_prob_attach) + "\n");
            }

            // the attachment product species mass is taken as the background
            // mass, same convention as for ionization/three-body above
            m_background_mass = species2.getMass();
        }

        else if (three_body_attachment_flag) {
            // calculate maximum collision frequency for ionization
            m_nu_max_threebody = get_nu_max_threebody(m_three_body_attachment_processes);

            // calculate total ionization probability
            auto coll_n_threebody = m_nu_max_threebody * dt;
            m_total_collision_prob_threebody = 1.0_prt - std::exp(-coll_n_threebody);

            if (coll_n_threebody > 0.1_prt) {
                ablastr::warn_manager::WMRecordWarning("BackgroundMCC Collisions",
                         "dt is too large to ensure accurate MCC three body attachment , coll_n_attachment: " +
                          std::to_string(coll_n_threebody) + " is > 0.1 and three body attachment probability is = " +
                          std::to_string(m_total_collision_prob_threebody) + "\n");
            }

            // if an ionization process is included the secondary species mass
            // is taken as the background mass
            m_background_mass = species2.getMass();
        }
        // if no neutral species mass was specified and ionization is not
        // included assume that the collisions will be with neutrals of the
        // same mass as the colliding species (as in ion-neutral collisions)
        else if (m_background_mass == -1) {
            m_background_mass = species1.getMass();
        }

        amrex::Print() << Utils::TextMsg::Info(
            "Setting up Monte-Carlo collisions for " + m_species_names[0] + " with:\n"
            + "     total non-ionization collision probability: "
            + std::to_string(m_total_collision_prob)
            + "\n     total ionization collision probability: "
            + std::to_string(m_total_collision_prob_ioniz)
            + "\n     total attachment collision probability: "
            + std::to_string(m_total_collision_prob_attach)
            + "\n     total three-body attachment collision probability: "
            + std::to_string(m_total_collision_prob_threebody)
        );

        init_flag = true;

    }

    // Loop over refinement levels
    auto const flvl = species1.finestLevel();
    for (int lev = 0; lev <= flvl; ++lev) {

        auto *cost = WarpX::getCosts(lev);

        // firstly loop over particles box by box and do all particle conserving
        // scattering
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(species1, lev); pti.isValid(); ++pti) {
            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            auto wt = static_cast<amrex::Real>(amrex::second());

            doBackgroundCollisionsWithinTile(pti, cur_time);

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
                wt = static_cast<amrex::Real>(amrex::second()) - wt;
                amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
            }
        }

        // secondly perform ionization through the SmartCopyFactory if needed, species3 is in the case of photoionization
        if (ionization_flag) {
            if (photo_ionization_flag) {
                auto& species3 = mypc->GetParticleContainerFromName(m_photo_species_name);
                doBackgroundPhotoIonization(lev, cost, K1, K2, total_collision_prob_photo, species1, species2, species3, cur_time);
            } 
            else {
            doBackgroundIonization(lev, cost, species1, species2, cur_time);
        };
    }
    else if (attachment_flag) {
            doBackgroundAttachment(lev, cost, species1, species2, cur_time);
    }
    else if (three_body_attachment_flag) {
            doBackgroundThreeBodyAttachment(lev, cost, species1, species2, cur_time);
        }
    }

    // Photoionization displaces the newly created electron/ion pair away from
    // the source particle by the photon's mean free path, so they can land
    // outside the box/tile they were appended to. A Redistribute() is needed
    // here (unlike the other creation processes above, which keep new
    // particles at the source particle's position) so that GatherAndPush,
    // which runs right after doCollisions() and before the next full
    // Redistribute(), does not index into field data using an out-of-tile
    // particle position.
    if (ionization_flag && photo_ionization_flag) {
        auto& species3 = mypc->GetParticleContainerFromName(m_photo_species_name);
        species1.Redistribute();
        species2.Redistribute();
        species3.Redistribute();
    }
}

void BackgroundMCCCollision::doBackgroundCollisionsWithinTile
( WarpXParIter& pti, amrex::Real t )
{
    using namespace amrex::literals;

    // So that CUDA code gets its intrinsic, not the host-only C++ library version
    using std::sqrt;

    // get particle count
    const long np = pti.numParticles();

    // get parsers for the background density and temperature
    auto n_a_func = m_background_density_func;
    auto T_a_func = m_background_temperature_func;

    // get collision parameters
    auto *scattering_processes = m_scattering_processes_exe.data();
    auto const process_count  = static_cast<int>(m_scattering_processes_exe.size());

    auto const total_collision_prob = m_total_collision_prob;
    auto const nu_max = m_nu_max;

    // store projectile and target masses
    auto const m = m_mass1;
    auto const M = m_background_mass;

    // we need particle positions in order to calculate the local density
    // and temperature
    auto GetPosition = GetParticlePosition<PIdx>(pti);

    // get Struct-Of-Array particle data, also called attribs
    auto& attribs = pti.GetAttribs();
    amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

    amrex::ParallelForRNG(np,
                          [=] AMREX_GPU_HOST_DEVICE (long ip, amrex::RandomEngine const& engine)
                          {
                              // determine if this particle should collide
                              if (amrex::Random(engine) > total_collision_prob) { return; }

                              // The background density and temperature parsers take Cartesian
                              // coordinates as arguments, in all geometries.
                              amrex::ParticleReal x, y, z;
                              GetPosition(ip, x, y, z);

                              const amrex::ParticleReal n_a = n_a_func(x, y, z, t);
                              const amrex::ParticleReal T_a = T_a_func(x, y, z, t);

                              amrex::ParticleReal v_coll, v_coll2, sigma_E, nu_i = 0;
                              double gamma, E_coll;
                              amrex::ParticleReal ua_x, ua_y, ua_z, vx, vy, vz;
                              const amrex::ParticleReal col_select = amrex::Random(engine);

                              // get velocities of gas particles from a Maxwellian distribution
                              auto const vel_std = sqrt(PhysConst::kb * T_a / M);
                              ua_x = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);
                              ua_y = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);
                              ua_z = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);

                              // we assume the target particle is not relativistic (in
                              // the lab frame) and therefore we can transform the projectile
                              // velocity to a frame in which the target is stationary with
                              // a simple Galilean boost
                              // not doing the full Lorentz boost here saves us computation
                              // since most particles will not actually collide
                              vx = ux[ip] - ua_x;
                              vy = uy[ip] - ua_y;
                              vz = uz[ip] - ua_z;
                              v_coll2 = (vx*vx + vy*vy + vz*vz);
                              v_coll = std::sqrt(v_coll2);

                              // calculate the collision energy in eV
                              ParticleUtils::getCollisionEnergy(v_coll2, m, M, gamma, E_coll);

                              // loop through all collision pathways
                              for (int i = 0; i < process_count; i++) {
                                  auto const& scattering_process = *(scattering_processes + i);
                                  // get collision cross-section
                                  sigma_E = scattering_process.getCrossSection(static_cast<amrex::ParticleReal>(E_coll));

                                  // calculate normalized collision frequency
                                  nu_i += n_a * sigma_E * v_coll / nu_max;

                                  // check if this collision should be performed
                                  if (col_select > nu_i) { continue; }

                                  // At this point the given particle has been chosen for a
                                  // collision with a background-gas particle of velocity
                                  // (ua_x, ua_y, ua_z). Compute the post-collision momentum of
                                  // the projectile using conservation of energy and momentum.
                                  // The angular distribution in the center-of-mass frame is set
                                  // by the process's scattering angle model, and any inelastic
                                  // energy loss is passed as the (released) reaction energy.
                                  // The background particle is treated as a reservoir: its recoil
                                  // is computed as the second product but discarded.
                                  amrex::ParticleReal u1x_out, u1y_out, u1z_out;
                                  amrex::ParticleReal u2x_out, u2y_out, u2z_out;
                                  TwoProductComputeProductMomenta(
                                      ux[ip], uy[ip], uz[ip], m,
                                      ua_x, ua_y, ua_z, M,
                                      u1x_out, u1y_out, u1z_out, m,
                                      u2x_out, u2y_out, u2z_out, M,
                                      -scattering_process.m_energy_penalty*PhysConst::q_e,
                                      // TwoProductComputeProductMomenta expects the *released* energy here, hence
                                      // the negative sign; the energy penalty is also converted from eV to Joules.
                                      scattering_process.m_scattering_angle_model,
                                      engine);

                                  // update projectile velocity with new components in labframe
                                  // (the background-gas recoil u2*_out is discarded)
                                  ux[ip] = u1x_out;
                                  uy[ip] = u1y_out;
                                  uz[ip] = u1z_out;
                                  break;
                              }
                          }
                          );
}


void BackgroundMCCCollision::doBackgroundIonization
( int lev, amrex::LayoutData<amrex::Real>* cost,
  WarpXParticleContainer& species1, WarpXParticleContainer& species2, amrex::Real t)
{
    ABLASTR_PROFILE("BackgroundMCCCollision::doBackgroundIonization()");

    const SmartCopyFactory copy_factory_elec(species1, species1);
    const SmartCopyFactory copy_factory_ion(species1, species2);
    const auto CopyElec = copy_factory_elec.getSmartCopy();
    const auto CopyIon = copy_factory_ion.getSmartCopy();

    const auto Filter = ImpactIonizationFilterFunc(
                                                   m_ionization_processes[0],
                                                   m_mass1, m_total_collision_prob_ioniz,
                                                   m_nu_max_ioniz, m_background_density_func, t
                                                   );

    const amrex::ParticleReal sqrt_kb_m = std::sqrt(PhysConst::kb / m_background_mass);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter pti(species1, lev); pti.isValid(); ++pti) {

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        auto& elec_tile = species1.ParticlesAt(lev, pti);
        auto& ion_tile = species2.ParticlesAt(lev, pti);

        const auto np_elec = elec_tile.numParticles();
        const auto np_ion = ion_tile.numParticles();

        auto Transform = ImpactIonizationTransformFunc(
                                                       m_ionization_processes[0].getEnergyPenalty(),
                                                       m_mass1, sqrt_kb_m, m_background_temperature_func, t
                                                       );

        const auto num_added = filterCopyTransformParticles<1>(species1, species2,
                                                               elec_tile, ion_tile, elec_tile, np_elec, np_ion,
                                                               Filter, CopyElec, CopyIon, Transform
                                                               );

        setNewParticleIDs(elec_tile, np_elec, num_added);
        setNewParticleIDs(ion_tile, np_ion, num_added);

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
        }

    }
    }


void BackgroundMCCCollision::doBackgroundPhotoIonization
( int lev, amrex::LayoutData<amrex::Real>* cost, double K1_, double K2_, double total_collision_prob_photo_,
  WarpXParticleContainer& species1, WarpXParticleContainer& species2, WarpXParticleContainer& species3, amrex::Real t)
{
    ABLASTR_PROFILE("BackgroundMCCCollision::doBackgroundPhotoIonization()");
    // Probability of collision for photoionization
    const SmartCopyFactory copy_factory_elec(species1, species1);
    const SmartCopyFactory copy_factory_ion(species1, species2);
    const SmartCopyFactory copy_factory_ion_2(species1, species3);

    const auto CopyElec = copy_factory_elec.getSmartCopy();
    const auto CopyIon = copy_factory_ion.getSmartCopy();
    const auto CopyIon2 = copy_factory_ion_2.getSmartCopy();

    const auto Filter = ImpactIonizationFilterFunc(
                                                   m_ionization_processes[0],
                                                   m_mass1, m_total_collision_prob_ioniz,
                                                   m_nu_max_ioniz, m_background_density_func, t
                                                   );

    const auto Filter2 = ImpactPhotoIonizationFilterFunc(
                                                   total_collision_prob_photo_, t
                                                   );
    const amrex::ParticleReal sqrt_kb_m = std::sqrt(PhysConst::kb / m_background_mass);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter pti(species1, lev); pti.isValid(); ++pti) {

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        auto& elec_tile = species1.ParticlesAt(lev, pti);
        auto& ion_tile = species2.ParticlesAt(lev, pti);
        auto& ion_tile_2 = species3.ParticlesAt(lev, pti);

        const auto np_elec = elec_tile.numParticles();
        const auto np_ion = ion_tile.numParticles();
        const auto np_ion_2 = ion_tile_2.numParticles();
        Nphotons = 10;
        auto Transform = ImpactIonizationTransformFunc(
                                                       m_ionization_processes[0].getEnergyPenalty(),
                                                       m_mass1, sqrt_kb_m, m_background_temperature_func, t
                                                       );
        auto Transform2 = ImpactPhotoIonizationTransformFunc(
                                                        f1, f2,
                                                        PO2, K1_, K2_, t
                                                    );

        const auto [num_added, num_added2] = filterCopyTransformCreateParticles<1>(species1, species2, species3,
                                                               elec_tile, ion_tile, ion_tile_2, elec_tile, np_elec, np_ion, np_ion_2,
                                                               Filter, Filter2, CopyElec, CopyIon, CopyIon2, Transform, Transform2
                                                               );       
        setNewParticleIDs(elec_tile, np_elec, num_added+num_added2);
        setNewParticleIDs(ion_tile, np_ion, num_added);
        setNewParticleIDs(ion_tile_2, np_ion_2, num_added2);
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
        }
}
};

void BackgroundMCCCollision::doBackgroundAttachment
( int lev, amrex::LayoutData<amrex::Real>* cost,
  WarpXParticleContainer& species1, WarpXParticleContainer& species2, amrex::Real t)
{
    ABLASTR_PROFILE("BackgroundMCCCollision::doBackgroundAttachment()");
    const SmartCopyFactory copy_factory_ion(species1, species2);
    const auto CopyIon = copy_factory_ion.getSmartCopy();

    const auto Filter = ImpactAttachmentFilterFunc(
                                                   m_attachment_processes[0],
                                                   m_mass1, m_total_collision_prob_attach,
                                                   m_nu_max_attach, m_background_density_func, t
                                                   );

    const amrex::ParticleReal sqrt_kb_m = std::sqrt(PhysConst::kb / m_background_mass);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter pti(species1, lev); pti.isValid(); ++pti) {

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        auto& elec_tile = species1.ParticlesAt(lev, pti);
        auto& ion_tile = species2.ParticlesAt(lev, pti);

        const auto np_ion = ion_tile.numParticles();

        auto Transform = ImpactAttachmentTransformFunc(
                                                       m_attachment_processes[0].getEnergyPenalty(),
                                                       m_mass1, sqrt_kb_m, m_background_temperature_func, t
                                                       );

        const auto num_added = filterCopyTransformParticles<1>(species2,
                                                               ion_tile, elec_tile, np_ion,
                                                               Filter, CopyIon, Transform
                                                               );
        setNewParticleIDs(ion_tile, np_ion, num_added);

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
        }
    }
}

void BackgroundMCCCollision::doBackgroundThreeBodyAttachment
( int lev, amrex::LayoutData<amrex::Real>* cost,
  WarpXParticleContainer& species1, WarpXParticleContainer& species2, amrex::Real t)
{
    ABLASTR_PROFILE("BackgroundMCCCollision::doBackgroundThreeBodyAttachment()");
    const SmartCopyFactory copy_factory_ion(species1, species2);
    const auto CopyIon = copy_factory_ion.getSmartCopy();

    const auto Filter = ImpactThreeBodyAttachmentFilterFunc(
                                                   m_three_body_attachment_processes[0],
                                                   m_mass1, m_total_collision_prob_threebody,
                                                   m_nu_max_threebody, m_background_density_func, m_background_density_func_2, t
                                                   );

    const amrex::ParticleReal sqrt_kb_m = std::sqrt(PhysConst::kb / m_background_mass);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter pti(species1, lev); pti.isValid(); ++pti) {

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        auto& elec_tile = species1.ParticlesAt(lev, pti);
        auto& ion_tile = species2.ParticlesAt(lev, pti);

        const auto np_ion = ion_tile.numParticles();

        auto Transform = ImpactThreeBodyAttachmentTransformFunc(
                                                       m_three_body_attachment_processes[0].getEnergyPenalty(),
                                                       m_mass1, sqrt_kb_m, m_background_temperature_func, t
                                                       );

        const auto num_added = filterCopyTransformParticles<1>(species2,
                                                               ion_tile, elec_tile, np_ion,
                                                               Filter, CopyIon, Transform
                                                               );
        setNewParticleIDs(ion_tile, np_ion, num_added);

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
        }
    }
}


