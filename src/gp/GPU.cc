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
   if( m_num_points < m_max_local_size )
      m_local_size = m_num_points;
   else
      m_local_size = m_max_local_size;

   // One individual por each group
   m_global_size = m_local_size * m_params->m_population_size;

   if( MaximumTreeSize() <= m_local_size )
      m_compile_flags += " -D MAX_TREE_SIZE_IS_LESS_THAN_LOCAL_SIZE";

   if( ! util::IsPowerOf2( m_local_size ) )
      m_compile_flags += " -D LOCAL_SIZE_IS_NOT_POWER_OF_2";

   m_compile_flags += " -D LOCAL_SIZE_NEXT_POWER_OF_2=" 
                      + util::ToString( util::NextPowerOf2( m_local_size ) );

   if( m_num_points % m_local_size == 0 )
      m_compile_flags += " -D NUM_POINTS_IS_DIVISIBLE_BY_LOCAL_SIZE";
}

// -----------------------------------------------------------------------------
void PPCE::CalculateNDRanges() 
{
   // Naïve rule:
   if( m_params->m_population_size <= m_max_local_size )
   {
      m_local_size = m_params->m_population_size;
      m_global_size= m_params->m_population_size;
   } else
   {
      m_local_size = m_max_local_size;

      // global size should be evenly divided by m_local_size
      if( m_params->m_population_size % m_local_size == 0 )
         m_global_size = m_params->m_population_size;
      else // round to the next divisible size (the kernel will ensure that
         // no access outside the population range will occur.
         m_global_size = m_params->m_population_size + m_local_size 
            - (m_params->m_population_size % m_local_size );
   }

   // Rules to better distribute the workload throughout the GPU processors
}

// -----------------------------------------------------------------------------
void FPI::CalculateNDRanges() 
{
   // Distribute the points (workload) evenly among the compute units
   m_local_size = std::min( (unsigned) std::ceil( m_num_points / (float) m_max_cu ),
         (unsigned) m_max_local_size );
   
   m_global_size = m_num_points;

   if( m_global_size % m_local_size != 0 )
      // Make m_global_size be divisible by m_local_size
      m_global_size += m_local_size - (m_num_points % m_local_size); 

   if( MaximumTreeSize() <= m_local_size )
      m_compile_flags += "-D MAX_TREE_SIZE_IS_LESS_THAN_LOCAL_SIZE";
}

// -----------------------------------------------------------------------------
