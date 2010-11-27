// -----------------------------------------------------------------------------
// $Id$
//
//   GPU.h
// 
//   Genetic Programming in OpenCL (gpocl)
//
//   Copyright (C) 2010-2010 Douglas A. Augusto
// 
// This file is part of gpocl
// 
// GPOCL is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
// 
// GPOCL is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
// 
// You should have received a copy of the GNU General Public License along
// with GPOCL; if not, see <http://www.gnu.org/licenses/>.
//
// -----------------------------------------------------------------------------

#ifndef _GPU_h
#define _GPU_h

#include "GP.h"

// -----------------------------------------------------------------------------
// FIXME: Provide a derived class for each strategy!
class GPonGPU: public GP {
public:
   virtual ~GPonGPU()
   {
      std::cerr << "\nCleaning GPonGPU...\n";
      //if( m_X ) delete[] m_X;
   }

   void PrintStrategy() const 
   { 
      std::cout << "GPU ";
      switch( m_params->m_device )
      {
         case Params::DEVICE_GPU_FPI:
            std::cout << "FPI";
            break;
         case Params::DEVICE_GPU_FPC:
            std::cout << "FPC";
            break;
         case Params::DEVICE_GPU_PPCU:
            std::cout << "PPCU";
            break;
         case Params::DEVICE_GPU_PPCE:
            std::cout << "PPCE";
            break;
      }
      std::cout << " (" << m_max_cu << " compute units)";
   }

   void SetKernelArgs()
   {
      GP::SetKernelArgs();

      if( m_params->m_device == Params::DEVICE_GPU_PPCU )
      {
         m_kernel.setArg( 4, sizeof(uint) * MaximumTreeSize(), NULL );
      }
   }

   void CalculateNDRanges() 
   {
      switch( m_params->m_device )
      {
         case Params::DEVICE_GPU_FPI:
            NDRangesFPI();
            break;
         case Params::DEVICE_GPU_FPC:
            NDRangesFPC();
            break;
         case Params::DEVICE_GPU_PPCU:
            NDRangesPPCU();
            break;
         case Params::DEVICE_GPU_PPCE:
            NDRangesPPCE();
            break;
      }
   }

   // -----------
   void NDRangesPPCU()
   {
      if( m_num_points < m_max_wi_size )
         m_num_local_wi = m_num_points;
      else
         m_num_local_wi = m_max_wi_size;

      // FIXME: m_num_global_wi % m_num_local_wi
      // One individual por each group
      m_num_global_wi = m_num_local_wi * m_params->m_population_size;

      // FIXME: Remove these restrictions! (need to change the kernel)
      // For now, m_num_local_wi must be power of two; let's check it:
      assert( ((int)m_num_local_wi & -(int)m_num_local_wi) == (int)m_num_local_wi );
      assert( MaximumTreeSize() <= m_num_local_wi );
      assert( m_num_points % m_num_local_wi == 0 );
   }

   // -----------
   void NDRangesPPCE() 
   {
      // Naïve rule:
      if( m_params->m_population_size <= m_max_wi_size )
      {
         m_num_local_wi = m_params->m_population_size;
         m_num_global_wi= m_params->m_population_size;
      } else
      {
         m_num_local_wi = m_max_wi_size;

         // global size should be evenly divided by m_num_local_wi
         if( m_params->m_population_size % m_num_local_wi == 0 )
            m_num_global_wi = m_params->m_population_size;
         else // round to the next divisible size (the kernel will ensure that
              // no access outside the population range will occur.
            m_num_global_wi = m_params->m_population_size + m_num_local_wi 
                              - (m_params->m_population_size % m_num_local_wi );
      }

      // Rules to better distribute the workload throughout the GPU processors

   }

   void NDRangesFPI() {}
   void NDRangesFPC() {}

   GPonGPU( Params& p ): GP( p, CL_DEVICE_TYPE_GPU )
   {
      // Load common definitions
      LoadKernel( "kernels/common.cl" );

      switch( m_params->m_device )
      {
         // Load the specific kernel according to the actual GPU strategy
         case Params::DEVICE_GPU_FPI:
            std::cout << "Strategy 'fitness-parallel interpreted'\n";
            LoadKernel( "kernels/gpu_fp.cl" );
            break;
         case Params::DEVICE_GPU_FPC:
            std::cout << "Strategy 'fitness-parallel compiled'\n";
            LoadKernel( "kernels/gpu_fp.cl" );
            break;
         case Params::DEVICE_GPU_PPCU:
            std::cout << "Strategy 'population-parallel compute unit'\n";
            LoadKernel( "kernels/gpu_ppcu.cl" );
            break;
         case Params::DEVICE_GPU_PPCE:
            std::cout << "Strategy 'population-parallel compute element'\n";
            LoadKernel( "kernels/gpu_ppce.cl" );
            break;
      }
   }

  void LoadPoints();
};

#endif
