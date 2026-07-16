/* Energy compaction resampling (Russian roulette + probabilistic splitting)

 * License: BSD-3-Clause-LBNL (same as WarpX)
 */
#include "RussianRoulette.H"

#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_ParticleTransformation.H>
#include <AMReX_Particles.H>
#include <AMReX_Random.H>
#include <AMReX_Scan.H>
#include <AMReX_StructOfArrays.H>

#include <cmath>
#include <limits>

RussianRoulette::RussianRoulette (const std::string& species_name)
{
    using namespace amrex::literals;

    const amrex::ParmParse pp_species_name(species_name);

    utils::parser::queryWithParser(
        pp_species_name, "resampling_energy_min_eV", m_energy_min_eV);
    utils::parser::queryWithParser(
        pp_species_name, "resampling_energy_max_eV", m_energy_max_eV);
    utils::parser::queryWithParser(
        pp_species_name, "resampling_bins_per_decade", m_bins_per_decade);
    utils::parser::queryWithParser(
        pp_species_name, "resampling_target_ppb", m_target_ppb);
    utils::parser::queryWithParser(
        pp_species_name, "resampling_max_split", m_max_split);
    utils::parser::queryWithParser(
        pp_species_name, "resampling_min_weight", m_min_weight);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_energy_min_eV > 0._rt &&
                                     m_energy_max_eV > m_energy_min_eV,
        "Energy compaction: 0 < resampling_energy_min_eV < resampling_energy_max_eV required");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_bins_per_decade >= 1,
        "Energy compaction: resampling_bins_per_decade must be >= 1");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_target_ppb > 0._rt,
        "Energy compaction: resampling_target_ppb must be > 0");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_max_split >= 1,
        "Energy compaction: resampling_max_split must be >= 1");
}

void RussianRoulette::operator() (
    const amrex::Geometry& /*geom_lev*/, WarpXParIter& pti,
    const int lev, WarpXParticleContainer * const pc) const
{
    using namespace amrex::literals;

    auto& ptile = pc->ParticlesAt(lev, pti);
    const int np = static_cast<int>(ptile.numParticles());
    if (np == 0) { return; }

    auto& soa = ptile.GetStructOfArrays();
    amrex::ParticleReal * const AMREX_RESTRICT w  = soa.GetRealData(PIdx::w).data();
    amrex::ParticleReal * const AMREX_RESTRICT ux = soa.GetRealData(PIdx::ux).data();
    amrex::ParticleReal * const AMREX_RESTRICT uy = soa.GetRealData(PIdx::uy).data();
    amrex::ParticleReal * const AMREX_RESTRICT uz = soa.GetRealData(PIdx::uz).data();
    auto * const AMREX_RESTRICT idcpu = soa.GetIdCPUData().data();

    // Rest-mass energy in eV; note: in WarpX (ux,uy,uz) = gamma*v [m/s]
    const amrex::ParticleReal mc2_eV =
        static_cast<amrex::ParticleReal>(pc->getMass()
        * PhysConst::c * PhysConst::c / PhysConst::q_e);
    const amrex::ParticleReal inv_c2 =
        static_cast<amrex::ParticleReal>(1._rt/(PhysConst::c*PhysConst::c));

    const amrex::Real log10_emin = std::log10(m_energy_min_eV);
    const amrex::Real bpd = static_cast<amrex::Real>(m_bins_per_decade);
    const int n_bins = static_cast<int>(std::ceil(
        (std::log10(m_energy_max_eV) - log10_emin) * bpd));

    // Device functor: log-energy bin index of particle ip (clamped: the
    // underflow region falls into bin 0, the overflow region into the last
    // bin, so all particles are always managed).
    auto bin_of = [=] AMREX_GPU_DEVICE (int ip) noexcept -> int
    {
        const amrex::ParticleReal u2 =
            (ux[ip]*ux[ip] + uy[ip]*uy[ip] + uz[ip]*uz[ip])*inv_c2;
        // gamma - 1 = u2/(1 + sqrt(1+u2)), numerically stable at low energy
        const amrex::ParticleReal gm1 = u2/(1._prt + std::sqrt(1._prt + u2));
        const amrex::ParticleReal e_eV = amrex::max(
            gm1*mc2_eV, std::numeric_limits<amrex::ParticleReal>::min());
        const int b = static_cast<int>(amrex::Math::floor(
            (std::log10(static_cast<amrex::Real>(e_eV)) - log10_emin)*bpd));
        return amrex::min(amrex::max(b, 0), n_bins - 1);
    };

    // -------------------------------------------------------------------
    // Pass 1: histogram of macroparticle counts per energy bin
    // -------------------------------------------------------------------
    amrex::Gpu::DeviceVector<unsigned long long> d_counts(n_bins, 0ull);
    auto * const AMREX_RESTRICT counts = d_counts.data();

    amrex::ParallelFor(np,
        [=] AMREX_GPU_DEVICE (int ip) noexcept
        {
            if (idcpu[ip] == amrex::ParticleIdCpus::Invalid) { return; }
            amrex::Gpu::Atomic::Add(&counts[bin_of(ip)], 1ull);
        });

    // Survival/splitting ratio per bin: r = target/count
    amrex::Gpu::DeviceVector<amrex::Real> d_ratio(n_bins);
    auto * const AMREX_RESTRICT ratio = d_ratio.data();
    const amrex::Real target_ppb = m_target_ppb;

    amrex::ParallelFor(n_bins,
        [=] AMREX_GPU_DEVICE (int b) noexcept
        {
            ratio[b] = (counts[b] > 0ull)
                ? target_ppb/static_cast<amrex::Real>(counts[b]) : 1._rt;
        });

    // -------------------------------------------------------------------
    // Pass 2: Russian roulette (r < 1) or splitting decision (r > 1)
    // -------------------------------------------------------------------
    amrex::Gpu::DeviceVector<int> d_num_extra(np, 0);
    auto * const AMREX_RESTRICT num_extra = d_num_extra.data();
    const int max_split = m_max_split;
    const amrex::ParticleReal min_weight =
        static_cast<amrex::ParticleReal>(m_min_weight);

    amrex::ParallelForRNG(np,
        [=] AMREX_GPU_DEVICE (int ip, amrex::RandomEngine const& engine) noexcept
        {
            if (idcpu[ip] == amrex::ParticleIdCpus::Invalid) { return; }
            const amrex::Real r = ratio[bin_of(ip)];

            if (r < 1._rt)
            {
                // Russian roulette: survive with probability r,
                // survivors carry weight w/r (conservation in expectation)
                if (amrex::Random(engine) < r) {
                    w[ip] = static_cast<amrex::ParticleReal>(w[ip]/r);
                } else {
                    idcpu[ip] = amrex::ParticleIdCpus::Invalid;
                }
            }
            else if (r > 1._rt)
            {
                // Probabilistic splitting: k = floor(r) + Bernoulli(frac(r)),
                // capped at max_split; k copies of weight w/k (exact conservation)
                const amrex::Real rc = amrex::min(
                    r, static_cast<amrex::Real>(max_split));
                int k = static_cast<int>(rc);
                if (amrex::Random(engine) < rc - static_cast<amrex::Real>(k)) {
                    k += 1;
                }
                if (k > 1 && w[ip]/k > min_weight)
                {
                    //amrex::Print() << "RussianRoulette: min_weight = " << w[ip]/k  << " " << k <<  " "<< min_weight << "\n";
                    w[ip] = static_cast<amrex::ParticleReal>(w[ip]/k);
                    num_extra[ip] = k - 1;
                }
            }
        });

    // -------------------------------------------------------------------
    // Pass 3: create the split copies
    // -------------------------------------------------------------------
    amrex::Gpu::DeviceVector<int> d_offsets(np);
    auto * const AMREX_RESTRICT offsets = d_offsets.data();
    const int total_new = amrex::Scan::ExclusiveSum(
        np, num_extra, offsets, amrex::Scan::RetSum{true});

    if (total_new == 0) { return; }

    const int old_np = np;
    ptile.resize(old_np + total_new);
    // NB: resize may reallocate; do not reuse w/ux/... pointers below.
    // Use the ParticleTileData accessor instead.
    auto ptd = ptile.getParticleTileData();

    // Reserve globally-unique particle IDs for the new copies
    const amrex::Long pid = WarpXParticleContainer::ParticleType::NextID();
    WarpXParticleContainer::ParticleType::NextID(pid + total_new);
    const int cpuid = amrex::ParallelDescriptor::MyProc();

    amrex::ParallelFor(old_np,
        [=] AMREX_GPU_DEVICE (int ip) noexcept
        {
            for (int j = 0; j < num_extra[ip]; ++j)
            {
                const int inew = old_np + offsets[ip] + j;
                // Copy every SoA component (position, momenta, weight -
                // already divided by k -, and all runtime attributes)
                amrex::copyParticle(ptd, ptd, ip, inew);
                // Give the copy its own valid ID
                ptd.m_idcpu[inew] = amrex::SetParticleIDandCPU(
                    pid + static_cast<amrex::Long>(offsets[ip]) + j, cpuid);
            }
        });

    amrex::Gpu::streamSynchronize();
}
