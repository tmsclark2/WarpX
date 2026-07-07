/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_IMPACT_PHOTO_IONIZATION_H_
#define WARPX_PARTICLES_COLLISION_IMPACT_PHOTO_IONIZATION_H_

#include "Particles/Collision/ScatteringProcess.H"

#include "Particles/Algorithms/KineticEnergy.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Random.H>
#include <AMReX_REAL.H>

class ImpactPhotoIonizationFilterFunc
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
    ImpactPhotoIonizationFilterFunc(
        double total_collision_prob_photo,
        amrex::Real t
    ) : m_total_collision_prob_photo(total_collision_prob_photo), m_t(t) { }

    /**
    * \brief Functor call. This method determines if a given (electron) particle
    * should undergo an ionization collision.
    *
    * @param[in] ptd particle tile data
    * @param[in] i particle index
    * @param[in] engine the random number state and factory
    * @return true if a collision occurs, false otherwise
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (
        int const mask, amrex::RandomEngine const& engine
    ) const noexcept
    {
        // determine if this particle should collide
        if (!mask) return false;
        // check if this collision should be performed
 
        return (Random(engine) < m_total_collision_prob_photo); 
    }

private:
    double m_total_collision_prob_photo = 0;
    amrex::Real m_t;
};

/** @file
 *
 * This file contains filter and transform functors for impact ionization
 */


/**
 * \brief Transform functor for impact ionization
 */
class ImpactPhotoIonizationTransformFunc
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
    * @param[in] f1 energy cost of ionization
    * @param[in] f2 mass of the colliding species
    * @param[in] PO2 value of sqrt(kB/m), where kB is Boltzmann's constant
                 and m is the background neutral mass
    * @param[in] K1 ParserExecutor<4> function to get the background
                 temperature in Kelvin as a function of space and time
    * @param[in] K2 ParserExecutor<4> function to get the background
                 temperature in Kelvin as a function of space and time
    * @param[in] t the current simulation time
    */
    ImpactPhotoIonizationTransformFunc(
        double f1, double f2, double PO2,
        double K1, double K2, amrex::Real t
    ) :  m_f1(f1), m_f2(f2),
         m_PO2(PO2), m_K1(K1), m_K2(K2), m_t(t) { }

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
    void operator() (DstData& dst1, DstData& dst2, SrcData& src,
        int const i_src, int const i_dst1, int const i_dst2, int Nphotons,
        amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex;
        using std::sqrt;
        // get references to the particle to get its position
        const auto& p = src.getSuperParticle(i_src);
        ParticleReal x, y, z;
        //double efficiency_photoionization = 0.5; // efficacité de la photoionisation 
        get_particle_position(p, x, y, z);

        // Tirage aléatoire pour fréquence et distance
        const amrex::Real Rg2 = Random(engine);
        const amrex::Real Rg3 = Random(engine);
        // Fréquence du photon
        const double f = m_f1 + Rg2 * (m_f2 - m_f1);
        // Mean free path 
        const double Lambda_gamma = m_K1*std::pow(m_K2 / m_K1,(f - m_f1)/(m_f2 - m_f1));

        // Distance parcourue par le photon
        const double d = -pow(Lambda_gamma,-1) * std::log(Rg3);

        // Tirage aléatoire angles theta/phi
        const double theta = Random(engine) * 2.0*M_PI;
        const double phi   = std::acos(2.0 * Random(engine) - 1.0);
        const double dx = d * std::sin(phi) * std::cos(theta);
        const double dz = d * std::cos(phi);
        // get references to the original particle weight
        const ParticleReal w_src = src.m_rdata[PIdx::w][i_src];

        // get references to the new particles' velocities
        auto& e_ux = dst1.m_rdata[PIdx::ux][i_dst1];
        auto& e_uy = dst1.m_rdata[PIdx::uy][i_dst1];
        auto& e_uz = dst1.m_rdata[PIdx::uz][i_dst1];
        auto& i_ux = dst2.m_rdata[PIdx::ux][i_dst2];
        auto& i_uy = dst2.m_rdata[PIdx::uy][i_dst2];
        auto& i_uz = dst2.m_rdata[PIdx::uz][i_dst2];
        auto& i_w = dst2.m_rdata[PIdx::w][i_dst2];
        auto& e_w = dst1.m_rdata[PIdx::w][i_dst1];

        auto& e_x = dst1.m_rdata[PIdx::x][i_dst1];
        auto& e_z = dst1.m_rdata[PIdx::z][i_dst1];
        auto& i_x = dst2.m_rdata[PIdx::x][i_dst2];
        auto& i_z = dst2.m_rdata[PIdx::z][i_dst2];
        
        e_x = x + dx;
        e_z = z + dz;            
        i_x = x + dx;
        i_z = z + dz;
        i_uy = .0_prt;
        i_ux = .0_prt;
        i_uz = .0_prt;
        e_uy = .0_prt;
        e_ux = .0_prt;
        e_uz = .0_prt;
        e_w = w_src/Nphotons; // assign weight to new electron
        i_w = w_src/Nphotons; // assign weight to new ion

#ifdef WARPX_DIM_XYZ
        const double dy = d * std::sin(theta) * std::sin(phi);
        auto& i_y = dst_ion.m_rdata[PIdx::y][i_dst2];
        auto& e_y = dst1.m_rdata[PIdx::y][i_dst1];
        i_y = y + dy;
        e_y = .0_prt;
        
#endif

}

private:
    double m_f1;
    double m_f2;
    double m_PO2;
    double m_K1;
    double m_K2;
    int m_Nphotons;
    amrex::Real m_t;
};
#endif // WARPX_PARTICLES_COLLISION_IMPACT_PHOTO_IONIZATION_H_
