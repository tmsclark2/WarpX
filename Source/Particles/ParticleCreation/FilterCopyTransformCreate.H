/* Copyright 2019-2020 Andrew Myers, Axel Huebl,
 * Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FILTER_COPY_TRANSFORM_CREATE_H_
#define WARPX_FILTER_COPY_TRANSFORM_CREATE_H_

#include "Particles/ParticleCreation/DefaultInitialization.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_TypeTraits.H>


/**
 * \brief Apply a filter, copy, and transform operation to the particles
 * in src, in that order, writing the results to dst1 and dst2, starting
 * at dst1_index and dst2_index. The dst tiles will be extended so all the
 * particles will fit, if needed.
 *
 * Note that the transform function operates on all of src, dst1, and dst2,
 * so all of them can be modified.
 *
 * This version of the function takes as inputs a mask which can be obtained
 * using another version of filterCopyTransformParticles that takes a filter
 * function as input.
 *
 * \tparam N number of particles created in the dst(s) for each filtered src particle
 * \tparam DstTile the dst particle tile type
 * \tparam SrcTile the src particle tile type
 * \tparam Index the index type, e.g. unsigned int
 * \tparam TransFunc the transform function type
 * \tparam CopyFunc1 the copy function type for src-->dst1
 * \tparam CopyFunc2 the copy function type for src-->dst2
 *
 * \param pc1  the first destination particle container
 * \param pc2  the second destination particle container
 * \param dst1 the destination tile
 * \param dst2 the destination tile
 * \param src the source tile
 * \param mask pointer to the mask: 1 means copy, 0 means don't copy
 * \param dst1_index the location at which to start writing the result to dst1
 * \param dst2_index the location at which to start writing the result to dst2
 * \param copy1 callable that defines what will be done for the "copy" step for src-->dst1.
 * \param copy2 callable that defines what will be done for the "copy" step for src-->dst2.
 * \param transform callable that defines the transformation to apply on dst and src.
 *
 *        The form of the callable should model:
 *            template <typename DstData, typename SrcData>
 *            AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 *            void operator() (DstData& dst, SrcData& src, int i_src, int i_dst);
 *
 *        where dst and src refer to the destination and source tiles and
 *        i_src and i_dst and the particle indices in each tile.
 *
 * \return num_added the number of particles that were written to dst.
 */
template <int N, typename DstPC, typename DstTile, typename SrcTile, typename Index,
          typename TransFunc, typename TransFunc2, typename CopyFunc1, typename CopyFunc2, typename CopyFunc3,
          amrex::EnableIf_t<std::is_integral_v<Index>, int> foo = 0>
std::pair<Index, Index>  filterCopyTransformCreateParticles (DstPC& pc1, DstPC& pc2, DstPC& pc3, DstTile& dst1, DstTile& dst2, DstTile& dst3, SrcTile& src, Index* mask, Index* mask2,
                                    Index dst1_index, Index dst2_index, Index dst3_index,
                                    CopyFunc1&& copy1, CopyFunc2&& copy2, CopyFunc3&& copy3,
                                    TransFunc&& transform, TransFunc2&& transform2) noexcept
{
    // Pour les électrons je dois rajouté le double de place car deux électrons créés
    using namespace amrex;

    auto np = src.numParticles();
    if (np == 0) { return {0,0}; }
    Gpu::DeviceVector<Index> offsets(np);
    Gpu::DeviceVector<Index> offsets2(np);

    auto total = amrex::Scan::ExclusiveSum(np, mask, offsets.data());
    auto total2 = amrex::Scan::ExclusiveSum(np, mask2, offsets2.data());

    const Index num_added = total;
    const Index num_added2 = N * total2;

    auto old_np1 = dst1.size();

    auto new_np1 = std::max(dst1_index + num_added + num_added2, dst1.numParticles());

    dst1.resize(new_np1);

    // fin Print pour le debug

    auto old_np2 = dst2.size();

    auto new_np2 = std::max(dst2_index + num_added, dst2.numParticles());

    dst2.resize(new_np2);

    auto old_np3= dst3.size();
    auto new_np3 = std::max(dst3_index + num_added2, dst3.numParticles());

    dst3.resize(new_np3);
    auto *p_offsets = offsets.dataPtr();
    auto *p_offsets2 = offsets2.dataPtr();

    const auto src_data  =  src.getParticleTileData();
    const auto dst1_data = dst1.getParticleTileData();
    const auto dst2_data = dst2.getParticleTileData();
    const auto dst3_data = dst3.getParticleTileData();

    amrex::ParallelForRNG(np,
    [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
    {
        if (mask[i])
        {
            copy1(dst1_data, src_data, i, p_offsets[i] + dst1_index, engine);
            copy2(dst2_data, src_data, i, p_offsets[i] + dst2_index, engine);
            transform(dst1_data, dst2_data, src_data, i,
                      p_offsets[i] + dst1_index,
                      p_offsets[i] + dst2_index,
                      engine);
            if (mask2[i])
            {
                for (int j = 0; j < N; ++j)
                {      
                    // electron correspond à la première espèce ici
                    copy1(dst1_data, src_data, i, num_added+N*p_offsets2[i] + dst1_index + j, engine);
                    copy3(dst3_data, src_data, i, N*p_offsets2[i] + dst3_index + j, engine);             
                    transform2(dst1_data, dst3_data, src_data, i,
                      num_added+N*p_offsets2[i] + dst1_index + j,
                      N*p_offsets2[i] + dst3_index + j, N,
                      engine);   
                    }
        }
    }
    });

    ParticleCreation::DefaultInitializeRuntimeAttributes(dst1,
                                       0, 0,
                                       pc1.getUserRealAttribs(), pc1.getUserIntAttribs(),
                                       pc1.GetRealSoANames(), pc1.GetIntSoANames(),
                                       pc1.getUserRealAttribParser(),
                                       pc1.getUserIntAttribParser(),
#ifdef WARPX_QED
                                       false, // do not initialize QED quantities, since they were initialized
                                              // when calling the CopyFunc functor
                                       pc1.get_breit_wheeler_engine_ptr(),
                                       pc1.get_quantum_sync_engine_ptr(),
#endif
                                       pc1.getIonizationInitialLevel(),
                                       old_np1, new_np1);
    ParticleCreation::DefaultInitializeRuntimeAttributes(dst2,
                                       0, 0,
                                       pc2.getUserRealAttribs(), pc2.getUserIntAttribs(),
                                       pc2.GetRealSoANames(), pc2.GetIntSoANames(),
                                       pc2.getUserRealAttribParser(),
                                       pc2.getUserIntAttribParser(),
#ifdef WARPX_QED
                                       false, // do not initialize QED quantities, since they were initialized
                                              // when calling the CopyFunc functor
                                       pc2.get_breit_wheeler_engine_ptr(),
                                       pc2.get_quantum_sync_engine_ptr(),
#endif
                                       pc2.getIonizationInitialLevel(),
                                       old_np2, new_np2);
    ParticleCreation::DefaultInitializeRuntimeAttributes(dst3,
                                       0, 0,
                                       pc3.getUserRealAttribs(), pc3.getUserIntAttribs(),
                                       pc3.GetRealSoANames(), pc3.GetIntSoANames(),
                                       pc3.getUserRealAttribParser(),
                                       pc3.getUserIntAttribParser(),
#ifdef WARPX_QED
                                       false, // do not initialize QED quantities, since they were initialized
                                              // when calling the CopyFunc functor
                                       pc3.get_breit_wheeler_engine_ptr(),
                                       pc3.get_quantum_sync_engine_ptr(),
#endif
                                       pc3.getIonizationInitialLevel(),
                                       old_np3, new_np3);
    Gpu::synchronize();
    return {num_added, num_added2};
}



/**
 * \brief Apply a filter, copy, and transform operation to the particles
 * in src, in that order, writing the results to dst1 and dst2, starting
 * at dst1_index and dst2_index. The dst tiles will be extended so all the
 * particles will fit, if needed.
 *
 * Note that the transform function operates on all of src, dst1, and dst2,
 * so all of them can be modified.
 *
 * This version of the function takes as inputs a mask which can be obtained
 * using another version of filterCopyTransformParticles that takes a filter
 * function as input.
 *
 * \tparam N number of particles created in the dst(s) for each filtered src particle
 * \tparam DstTile the dst particle tile type
 * \tparam SrcTile the src particle tile type
 * \tparam Index the index type, e.g. unsigned int
 * \tparam TransFunc the transform function type
 * \tparam CopyFunc1 the copy function type for src-->dst1
 * \tparam CopyFunc2 the copy function type for src-->dst2
 *
 * \param pc1  the first destination particle container
 * \param pc2  the second destination particle container
 * \param dst1 the destination tile
 * \param dst2 the destination tile
 * \param src the source tile
 * \param mask pointer to the mask: 1 means copy, 0 means don't copy
 * \param dst1_index the location at which to start writing the result to dst1
 * \param dst2_index the location at which to start writing the result to dst2
 * \param copy1 callable that defines what will be done for the "copy" step for src-->dst1.
 * \param copy2 callable that defines what will be done for the "copy" step for src-->dst2.
 * \param transform callable that defines the transformation to apply on dst and src.
 *
 *        The form of the callable should model:
 *            template <typename DstData, typename SrcData>
 *            AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 *            void operator() (DstData& dst, SrcData& src, int i_src, int i_dst);
 *
 *        where dst and src refer to the destination and source tiles and
 *        i_src and i_dst and the particle indices in each tile.
 *
 * \return num_added the number of particles that were written to dst.
 */

template <int N, typename DstPC, typename DstTile, typename SrcTile, typename Index,
          typename PredFunc, typename PredFunc2, typename TransFunc, typename TransFunc2, typename CopyFunc1, typename CopyFunc2, typename CopyFunc3>
std::pair<Index, Index>  filterCopyTransformCreateParticles (DstPC& pc1, DstPC& pc2, DstPC& pc3, DstTile& dst1, DstTile& dst2, DstTile& dst3, SrcTile& src,
                                    Index dst1_index, Index dst2_index, Index dst3_index,
                                    PredFunc&& filter, PredFunc2&& filter2, CopyFunc1&& copy1, CopyFunc2&& copy2, CopyFunc3&& copy3,
                                    TransFunc&& transform, TransFunc2&& transform2) noexcept
{
    using namespace amrex;

    auto np = src.numParticles();
    if (np == 0) { return {0,0}; }

    Gpu::DeviceVector<Index> mask(np);
    Gpu::DeviceVector<Index> mask2(np);

    auto *p_mask = mask.dataPtr();
    auto *p_mask2 = mask2.dataPtr();

    const auto src_data = src.getParticleTileData();

    amrex::ParallelForRNG(np,
    [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine)
    {
        p_mask[i] = filter(src_data, i, engine);
        p_mask2[i] = filter2(p_mask[i], engine);
    });

    return filterCopyTransformCreateParticles<N>(pc1, pc2, pc3, dst1, dst2, dst3, src, mask.dataPtr(), mask2.dataPtr(),
                                        dst1_index, dst2_index, dst3_index,
                                        std::forward<CopyFunc1>(copy1),
                                        std::forward<CopyFunc2>(copy2),
                                        std::forward<CopyFunc3>(copy3),
                                        std::forward<TransFunc>(transform),
                                        std::forward<TransFunc2>(transform2));
}

#endif //WARPX_FILTER_COPY_TRANSFORM_H_
