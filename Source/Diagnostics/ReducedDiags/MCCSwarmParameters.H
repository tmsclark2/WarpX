/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_MCCSWARMPARAMETERS_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_MCCSWARMPARAMETERS_H_

#include "ReducedDiags.H"

#include <string>

/**
 *  This class computes the electron-swarm parameters conventionally used to
 *  characterize a background_mcc collision model against reference data
 *  (e.g. BOLSIG+ or LXCat swarm data), for a single species and as a
 *  function of the reduced field E/N: the mean electron energy, the axial
 *  drift velocity, the electron mobility, and the (reduced) Townsend
 *  ionization and attachment coefficients.
 *
 *  The ionization/attachment coefficients are obtained from the actual
 *  weighted count of ionization/attachment events performed by the
 *  background_mcc collision(s) targeting this species during the most
 *  recently completed step (see BackgroundMCCCollision::get_ionization_event_weight
 *  and get_attachment_event_weight), rather than by re-deriving a rate from
 *  the cross-sections -- this way the reduced diagnostic always reports
 *  exactly what the MCC model did, not a theoretical re-estimate of it.
 */
class MCCSwarmParameters : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    explicit MCCSwarmParameters (const std::string& rd_name);

    /**
     * This function computes the swarm parameters for the requested species.
     *
     * @param[in] step current time step
     */
    void ComputeDiags (int step) final;

private:

    //! name of the species (e.g. electrons) the swarm parameters are computed for
    std::string m_species_name;
};

#endif // WARPX_DIAGNOSTICS_REDUCEDDIAGS_MCCSWARMPARAMETERS_H_
