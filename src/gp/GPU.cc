// -----------------------------------------------------------------------------
// $Id$
//
//   GPU.cc
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

#include "GPU.h"

// -----------------------------------------------------------------------------
void GPonGPU::LoadPoints()
{
   std::vector<std::vector<cl_float> > tmp_X;
   GP::LoadPoints( tmp_X );

   // Allocate enough memory (linear) to hold the transposed version
   m_X = new cl_float[ m_num_points * m_x_dim ];

   // Transposition
   // TODO: Acrescentar gráfico explicativo!
   unsigned pos = 0;
   for( unsigned j = 0; j < tmp_X[0].size(); ++j )
      for( unsigned i = 0; i < tmp_X.size(); ++i )
         m_X[pos++] = tmp_X[i][j];
/*
   for( unsigned i = 0; i < m_num_points * m_x_dim; ++i)
      std::cout << m_X[i] << " ";
   for( unsigned i = 0; i < m_num_points; ++i)
      std::cout << m_Y[i] << " ";
 */
}

// -----------------------------------------------------------------------------
void PPCU::CalculateNDRanges() 
{
   if( m_num_points < m_max_wi_size )
      m_num_local_wi = m_num_points;
   else
      m_num_local_wi = m_max_wi_size;

   // One individual por each group
   m_num_global_wi = m_num_local_wi * m_params->m_population_size;

   if( MaximumTreeSize() <= m_num_local_wi )
      m_compile_flags += " -D MAX_TREE_SIZE_IS_LESS_THAN_WGS";

   if( ! util::IsPowerOf2( m_num_local_wi ) )
      m_compile_flags += " -D WGS_IS_NOT_POWER_OF_2";

   m_compile_flags += " -D WGS_NEXT_POWER_OF_2=" 
                      + util::ToString( util::NextPowerOf2( m_num_local_wi ) );

   // FIXME: Remove these restrictions! (need to change the kernel)
   assert( m_num_points % m_num_local_wi == 0 );
}

// -----------------------------------------------------------------------------
void PPCE::CalculateNDRanges() 
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

// -----------------------------------------------------------------------------
void FPI::CalculateNDRanges() 
{
   // Distribute the points (workload) evenly among the compute units
   m_num_local_wi = std::min( (unsigned) std::ceil( m_num_points / m_max_cu ),
         (unsigned) m_max_wi_size );
   // Make m_num_global_wi be divisible by m_num_local_wi
   m_num_global_wi = m_num_points + m_num_local_wi - (m_num_points % m_num_local_wi); 

   if( MaximumTreeSize() <= m_num_local_wi )
      m_compile_flags += "-D MAX_TREE_SIZE_IS_LESS_THAN_WGS";
}

// -----------------------------------------------------------------------------
