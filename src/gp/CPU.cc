// -----------------------------------------------------------------------------
// $Id$
//
//   CPU.cc
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

#include "CPU.h"

// -----------------------------------------------------------------------------
GPonCPU::GPonCPU( Params& p ): GP( p, CL_DEVICE_TYPE_CPU )
{
   // FIXME: (AMD only!) Use the more portable Fission (from cl_ext.h) instead!
   if( m_params->m_cpu_cores > 0 )
   {
      /* FIXME: putenv should accept const char* as argument!
      const std::string env = "CPU_MAX_COMPUTE_UNITS=" + util::ToString( m_params->m_cpu_cores );
      putenv( env.c_str() );
      */
      setenv( "CPU_MAX_COMPUTE_UNITS", util::ToString( m_params->m_cpu_cores ).c_str(), 1 );
   }

   LoadKernel( "kernels/common.cl" );
   LoadKernel( "kernels/cpu.cl" );
}

// -----------------------------------------------------------------------------
void GPonCPU::LoadPoints()
{
   std::vector<std::vector<cl_float> > tmp_X;
   GP::LoadPoints( tmp_X );

   // Allocate enough memory (linear) to hold the linear version
   m_X = new cl_float[ m_num_points * m_x_dim ];

   // Linearization
   unsigned pos = 0;
   for( unsigned i = 0; i < tmp_X.size(); ++i )
      for( unsigned j = 0; j < tmp_X[0].size(); ++j )
         m_X[pos++] = tmp_X[i][j];

   /*
   for( unsigned i = 0; i < m_num_points * m_x_dim; ++i)
      std::cout << m_X[i] << " ";
   std::cout << std::endl;
   for( unsigned i = 0; i < m_num_points; ++i)
      std::cout << m_Y[i] << " ";
   std::cout << std::endl;
   */
}
