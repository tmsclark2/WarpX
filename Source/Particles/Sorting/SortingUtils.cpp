/* Copyright 2019-2020 Andrew Myers, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "SortingUtils.H"

void fillWithConsecutiveIntegers( amrex::Gpu::DeviceVector<long>& v )
{
#ifdef AMREX_USE_GPU
    // On GPU: Use amrex
    auto data = v.data();
    auto N = v.size();
    AMREX_FOR_1D( N, i, {data[i] = i;});
#else
    // On CPU: Use std library
    std::iota( v.begin(), v.end(), 0L );
#endif
}

void fillWithConsecutiveReal( amrex::Gpu::DeviceVector<amrex::Real>& v,amrex::Real begin,amrex::Real increment, int N)
{
#ifdef AMREX_USE_GPU
    // On GPU: Use amrex
    auto data = begin;
    AMREX_FOR_1D( N, i, {data+i*increment;});
#else
    // On CPU: Use std library
    std::iota( begin, N, 0L );
#endif
}
