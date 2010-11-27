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


