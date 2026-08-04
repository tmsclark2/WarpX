/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_BACKGROUNDMCCCOLLISION_H_
#define WARPX_PARTICLES_COLLISION_BACKGROUNDMCCCOLLISION_H_

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"
#include "Particles/Collision/ScatteringProcess.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <memory>
#include <string>

class BackgroundMCCCollision final
    : public CollisionBase
{
public:
    explicit BackgroundMCCCollision (std::string const& collision_name);

    ~BackgroundMCCCollision () override = default;

    BackgroundMCCCollision ( BackgroundMCCCollision const &)             = delete;
    BackgroundMCCCollision& operator= ( BackgroundMCCCollision const & ) = delete;
    BackgroundMCCCollision ( BackgroundMCCCollision&& )                  = delete;
    BackgroundMCCCollision& operator= ( BackgroundMCCCollision&& )       = delete;

    [[nodiscard]] amrex::ParticleReal get_nu_max (amrex::Vector<ScatteringProcess> const& mcc_processes) const;
    [[nodiscard]] amrex::ParticleReal get_nu_max_threebody (amrex::Vector<ScatteringProcess> const& mcc_processes) const;

    //! whether this collision creates particles through electron-impact (or photo-) ionization
    [[nodiscard]] bool does_ionization () const { return ionization_flag; }
    //! whether this collision removes electrons through attachment (two- or three-body)
    [[nodiscard]] bool does_attachment () const { return attachment_flag || three_body_attachment_flag; }

    //! sum of the statistical weight of all ionization events performed the last time
    //! doCollisions() was called (i.e. during the most recently completed step)
    [[nodiscard]] amrex::Real get_ionization_event_weight () const { return m_ioniz_event_weight; }
    //! sum of the statistical weight of all attachment events (two- or three-body) performed
    //! the last time doCollisions() was called (i.e. during the most recently completed step)
    [[nodiscard]] amrex::Real get_attachment_event_weight () const { return m_attach_event_weight; }

    //! Turns on the per-step event-weight bookkeeping used by
    //! get_ionization_event_weight()/get_attachment_event_weight(). Off by
    //! default so that runs that do not use the MCCSwarmParameters reduced
    //! diagnostic do not pay for it; called by that diagnostic's constructor
    //! on the collisions it needs.
    void EnableEventTracking () { m_track_events = true; }

    //! background gas number density used to normalize the collision
    //! probabilities (assumed spatially uniform for swarm-parameter runs)
    [[nodiscard]] amrex::ParticleReal get_max_background_density () const { return m_max_background_density; }
    //! background gas temperature parser, evaluated as T(x,y,z,t) [K]
    [[nodiscard]] amrex::ParserExecutor<4> const& get_background_temperature_func () const
        { return m_background_temperature_func; }

    /** Perform the collisions
     *
     * @param cur_time Current time
     * @param dt Time step size
     * @param mypc Container of species involved
     *
     */
    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override;

    /** Perform particle conserving MCC collisions within a tile
     *
     * @param pti particle iterator
     * @param t current time
     *
     */
    void doBackgroundCollisionsWithinTile ( WarpXParIter& pti, amrex::Real t);

    /** Perform MCC ionization interactions
     *
     * @param[in] lev the mesh-refinement level
     * @param[in,out] cost pointer to (load balancing) cost corresponding to box where present particles are ionized.
     * @param[in,out] species1,species2 reference to species container used to inject
     * new particles
     * @param t current time
     *
     */
    void doBackgroundIonization (
                                 int lev,
                                 amrex::LayoutData<amrex::Real>* cost,
                                 WarpXParticleContainer& species1,
                                 WarpXParticleContainer& species2,
                                 amrex::Real t
                                 );

    /** Perform MCC ionization interactions in case of a photoionization process
     *
     * @param[in] lev the mesh-refinement level
     * @param[in,out] cost pointer to (load balancing) cost corresponding to box where present particles are ionized.
     * @param[in,out] species1,species2, species3 reference to species container used to inject
     * new particles
     * @param t current time
     *
     */

    void doBackgroundPhotoIonization (
                                 int lev,
                                 amrex::LayoutData<amrex::Real>* cost,
                                 double K1, 
                                 double K2, 
                                 double total_collision_prob_photo,
                                 WarpXParticleContainer& species1,
                                 WarpXParticleContainer& species2,
                                 WarpXParticleContainer& species3,
                                 amrex::Real t
                                 );

    void doBackgroundAttachment (
                                int lev,
                                amrex::LayoutData<amrex::Real>* cost,
                                WarpXParticleContainer& species1,
                                WarpXParticleContainer& species2,
                                amrex::Real t
                                );
    void doBackgroundThreeBodyAttachment (
                                int lev,
                                amrex::LayoutData<amrex::Real>* cost,
                                WarpXParticleContainer& species1,
                                WarpXParticleContainer& species2,
                                amrex::Real t
                                );


private:

    amrex::Vector<ScatteringProcess> m_scattering_processes;
    amrex::Vector<ScatteringProcess> m_ionization_processes;
    amrex::Vector<ScatteringProcess> m_attachment_processes;
    amrex::Vector<ScatteringProcess> m_three_body_attachment_processes;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_scattering_processes_exe;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_ionization_processes_exe;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_attachment_processes_exe;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_three_body_attachment_processes_exe;

    bool init_flag = false;
    bool ionization_flag = false;
    bool photo_ionization_flag = false;
    bool attachment_flag = false;
    bool three_body_attachment_flag = false;

    //! weighted count of ionization/attachment events performed during the
    //! most recent doCollisions() call, reset at the start of each call --
    //! used by the MCCSwarmParameters reduced diagnostic
    amrex::Real m_ioniz_event_weight = 0.0;
    amrex::Real m_attach_event_weight = 0.0;
    //! guards the (small but non-zero) cost of maintaining the counters
    //! above so that it is only paid when a diagnostic actually reads them
    bool m_track_events = false;

    amrex::ParticleReal m_mass1;
    amrex::ParticleReal m_max_background_density = 0;
    amrex::ParticleReal m_max_background_density_2 = 0;
    amrex::ParticleReal m_background_mass;
    amrex::ParticleReal m_total_collision_prob;
    amrex::ParticleReal m_total_collision_prob_ioniz = 0;
    amrex::ParticleReal m_total_collision_prob_attach = 0;
    amrex::ParticleReal m_total_collision_prob_threebody = 0;
    amrex::ParticleReal m_nu_max;
    amrex::ParticleReal m_nu_max_threebody;
    amrex::ParticleReal m_nu_max_attach;
    amrex::ParticleReal m_nu_max_ioniz;

    amrex::Parser m_background_density_parser;
    amrex::Parser m_background_density_parser_2;
    amrex::Parser m_background_temperature_parser;

    amrex::ParserExecutor<4> m_background_density_func;
    amrex::ParserExecutor<4> m_background_density_func_2;
    amrex::ParserExecutor<4> m_background_temperature_func;

    int Nphotons;
    double f1;
    double f2; 

    double PO2; 
    double K1; 
    double K2; 
    double pq; 
    double tot_press; 
    double q_press; 
    double q_E; 
    double total_collision_prob_photo;
    //int N_photons; 

};


void remove_colliding_particle (
    uint64_t& idcpu);

#endif // WARPX_PARTICLES_COLLISION_BACKGROUNDMCCCOLLISION_H_