/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "MCCSwarmParameters.H"

#include "Fields.H"
#include "Particles/Collision/BackgroundMCC/BackgroundMCCCollision.H"
#include "Particles/Collision/CollisionHandler.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_RealBox.H>
#include <AMReX_Tuple.H>

#include <algorithm>
#include <fstream>
#include <limits>

using namespace amrex;

// constructor
MCCSwarmParameters::MCCSwarmParameters (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    const amrex::ParmParse pp_rd_name(rd_name);
    pp_rd_name.get("species", m_species_name);

    // 6 outputs: mean energy, drift velocity, mean field, mobility, Townsend
    // (ionization) coefficient, attachment coefficient
    m_data.resize(6, 0.0_rt);

    // Only the background_mcc collisions targeting this species need to pay
    // for the per-step event-weight bookkeeping this diagnostic relies on;
    // everyone else's runs are unaffected (see BackgroundMCCCollision::EnableEventTracking).
    auto & mypc = WarpX::GetInstance().GetPartContainer();
    for (const auto& collision : mypc.GetCollisionHandler().GetAllCollisions())
    {
        auto* mcc = dynamic_cast<BackgroundMCCCollision*>(collision.get());
        if (mcc == nullptr) { continue; }
        const auto& coll_species_names = mcc->get_species_names();
        if (!coll_species_names.empty() && coll_species_names[0] == m_species_name) {
            mcc->EnableEventTracking();
        }
    }

    if (ParallelDescriptor::IOProcessor())
    {
        if (m_write_header)
        {
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]" << m_species_name + "_mean_energy(eV)";
            ofs << m_sep;
            ofs << "[" << c++ << "]" << m_species_name + "_drift_velocity_z(m/s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]" << m_species_name + "_mean_field_z(V/m)";
            ofs << m_sep;
            ofs << "[" << c++ << "]" << m_species_name + "_mobility(Torr*cm2/(V*s))";
            ofs << m_sep;
            ofs << "[" << c++ << "]" << m_species_name + "_townsend_alpha(1/(cm*Torr))";
            ofs << m_sep;
            ofs << "[" << c++ << "]" << m_species_name + "_attachment_coefficient(1/(cm*Torr))";
            ofs << "\n";
            ofs.close();
        }
    }
}

void MCCSwarmParameters::ComputeDiags (int step)
{
    if (!m_intervals.contains(step+1)) { return; }

    auto & warpx = WarpX::GetInstance();
    auto & mypc = warpx.GetPartContainer();

    const auto species_names = mypc.GetSpeciesNames();
    const auto it = std::find(species_names.begin(), species_names.end(), m_species_name);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(it != species_names.end(),
        "MCCSwarmParameters: unknown species '" + m_species_name + "'.");
    const int i_s = static_cast<int>(std::distance(species_names.begin(), it));

    auto & myspc = mypc.GetParticleContainer(i_s);

    // --- mean energy (eV) and total weight ---
    auto [Etot, Wtot] = myspc.sumParticleWeightAndEnergy(false);
    const amrex::Real mean_energy_eV =
        (Wtot > std::numeric_limits<Real>::min()) ? (Etot / Wtot) / PhysConst::q_e : 0.0_rt;

    // --- drift velocity (m/s), relativistically-correct mean velocity ---
    const auto mean_v = myspc.meanParticleVelocity(false);
    const amrex::Real v_drift_z = mean_v[2];

    // --- mean axial field (V/m), level 0 only ---
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;
    constexpr int lev = 0;
    amrex::Geometry const & geom = warpx.Geom(lev);
    const amrex::MultiFab & Ez = *warpx.m_fields.get(FieldType::Efield_aux, Direction{2}, lev);

    const amrex::GpuArray<int,3> cellCenteredType{0,0,0};
    const amrex::GpuArray<int,3> coarsening_ratio{1,1,1};
    constexpr int comp = 0;
    amrex::GpuArray<int,3> Eztype{0,0,0};
    for (int i = 0; i < AMREX_SPACEDIM; ++i) { Eztype[i] = Ez.ixType()[i]; }

    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(Ez, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = enclosedCells(mfi.nodaltilebox());
        const auto& arrEz = Ez[mfi].array();
        reduce_op.eval(box, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            return ablastr::coarsen::sample::Interp(arrEz, Eztype, cellCenteredType,
                                                     coarsening_ratio, i, j, k, comp);
        });
    }
    amrex::Real Ez_sum = amrex::get<0>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceRealSum(Ez_sum);
    // the level's full domain cell count is known analytically, no MPI reduce needed
    const amrex::Long n_cells = geom.Domain().numPts();
    const amrex::Real Ez_mean = (n_cells > 0) ? Ez_sum / static_cast<amrex::Real>(n_cells) : 0.0_rt;

    // --- mobility (m2/(V.s)) ---
    const amrex::Real mobility =
        (std::abs(Ez_mean) > std::numeric_limits<Real>::min()) ? v_drift_z / Ez_mean : 0.0_rt;

    // --- Townsend ionization / attachment coefficients (1/m) ---
    // Obtained from the weighted count of ionization/attachment events that
    // the background_mcc collision(s) targeting this species actually
    // performed during the step that was just completed (rather than from a
    // separate re-derivation using cross-sections): gamma = event_weight /
    // (dt * Wtot) is the mean event frequency per electron, and
    // alpha = gamma / v_drift_z is the corresponding Townsend coefficient.
    const amrex::Real dt = warpx.getdt(lev);
    amrex::Real ioniz_weight = 0.0_rt;
    amrex::Real attach_weight = 0.0_rt;
    // background gas number density (m^-3) and temperature (K), used below to
    // convert to the conventional (Torr-normalized) swarm-parameter units;
    // taken from the first matching collision (swarm setups use one uniform
    // background gas for all the background_mcc collisions of a species)
    amrex::Real background_N = 0.0_rt;
    amrex::Real background_T = 0.0_rt;
    bool found_background_gas = false;
    for (const auto& collision : mypc.GetCollisionHandler().GetAllCollisions())
    {
        auto* mcc = dynamic_cast<BackgroundMCCCollision*>(collision.get());
        if (mcc == nullptr) { continue; }
        const auto& coll_species_names = mcc->get_species_names();
        if (coll_species_names.empty() || coll_species_names[0] != m_species_name) { continue; }

        if (mcc->does_ionization()) { ioniz_weight += mcc->get_ionization_event_weight(); }
        if (mcc->does_attachment()) { attach_weight += mcc->get_attachment_event_weight(); }

        if (!found_background_gas) {
            background_N = mcc->get_max_background_density();
            background_T = mcc->get_background_temperature_func()(0.0_rt, 0.0_rt, 0.0_rt, warpx.gett_new(lev));
            found_background_gas = true;
        }
    }

    // background pressure (Torr) of the actual simulated gas -- P = N*kB*T,
    // converted from Pa to Torr (1 Torr = 133.322368 Pa)
    const amrex::Real P_torr = background_N * PhysConst::kb * background_T / 133.322368_rt;
    const bool have_pressure = P_torr > std::numeric_limits<Real>::min();

    amrex::Real townsend_alpha = 0.0_rt;
    amrex::Real attachment_coeff = 0.0_rt;
    if (Wtot > std::numeric_limits<Real>::min() &&
        std::abs(v_drift_z) > std::numeric_limits<Real>::min() && dt > 0.0_rt)
    {
        const amrex::Real gamma_ioniz = ioniz_weight / (dt * Wtot);
        const amrex::Real gamma_attach = attach_weight / (dt * Wtot);
        townsend_alpha = gamma_ioniz / v_drift_z;
        attachment_coeff = gamma_attach / v_drift_z;
    }

    // --- convert to the conventional swarm-parameter units ---
    // mobility: m2/(V.s) -> Torr*cm2/(V.s)      (x1e4 for m2->cm2, xP[Torr])
    // alpha, a: 1/m      -> 1/(cm*Torr)          (x1e-2 for 1/m->1/cm, /P[Torr])
    const amrex::Real mobility_Torr = have_pressure ? mobility * 1.0e4_rt * P_torr : 0.0_rt;
    const amrex::Real townsend_alpha_Torr = have_pressure ? townsend_alpha * 1.0e-2_rt / P_torr : 0.0_rt;
    const amrex::Real attachment_coeff_Torr = have_pressure ? attachment_coeff * 1.0e-2_rt / P_torr : 0.0_rt;

    m_data[0] = mean_energy_eV;
    m_data[1] = v_drift_z;
    m_data[2] = Ez_mean;
    m_data[3] = mobility_Torr;
    m_data[4] = townsend_alpha_Torr;
    m_data[5] = attachment_coeff_Torr;
}
