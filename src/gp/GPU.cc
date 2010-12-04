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

#include <iomanip>

// -----------------------------------------------------------------------------
void GPonGPU::LoadPoints()
{
   std::vector<std::vector<cl_float> > tmp_X;
   GP::LoadPoints( tmp_X );

   // Allocate enough memory (linear) to hold the transposed version
   m_X = new cl_float[ m_num_points * m_x_dim ];

   /*
     TRANSPOSITION (for coalesced access on the GPU)

                                 Transformed and linearized data points
                                 +----------++----------+   +----------+
                                 | 1 2     q|| 1 2     q|   | 1 2     q|
           +-------------------> |X X ... X ||X X ... X |...|X X ... X |
           |                     | 1 1     1|| 2 2     2|   | p p     p|
           |                     +----------++----------+   +----------+
           |                                ^             ^
           |    ____________________________|             |
           |   |       ___________________________________|
           |   |      |
         +--++--+   +--+
         | 1|| 1|   | 1|
         |X ||X |...|X |
         | 1|| 2|   | p|
         |  ||  |   |  |
         | 2|| 2|   | q|
         |X ||X |...|X |
         | 1|| 2|   | p|
         |. ||. |   |. |
         |. ||. |   |. |
         |. ||. |   |. |
         | q|| q|   | q|
         |X ||X |...|X |
         | 1|| 2|   | p|
         +--++--+   +--+
      Original data points

    */
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

   // One individual per work-group
   m_global_size = m_local_size * m_params->m_population_size;

   m_compile_flags += " -D LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2=" 
                      + util::ToString( util::NextPowerOf2( m_local_size ) );

   if( MaximumTreeSize() > m_local_size )
      m_compile_flags += " -D PROGRAM_TREE_DOES_NOT_FIT_IN_LOCAL_SIZE";

   if( ! util::IsPowerOf2( m_local_size ) )
      m_compile_flags += " -D LOCAL_SIZE_IS_NOT_POWER_OF_2";

   if( m_num_points % m_local_size != 0 )
      m_compile_flags += " -D NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE";
}

// -----------------------------------------------------------------------------
void PPCE::CalculateNDRanges() 
{
   // Evenly distribute the workload among the compute units (but avoiding local size
   // being more than the maximum allowed.
   m_local_size = std::min( m_max_local_size, 
                            (unsigned) ceil( m_params->m_population_size/(float) m_max_cu ) ) ;

   m_global_size = m_params->m_population_size;

   // It is better to have global size divisible by local size
   if( m_global_size % m_local_size != 0 )
      // Round to the next divisible size (the kernel will ensure that
      // no access outside the population range will occur).
      m_global_size += m_local_size - (m_global_size % m_local_size);
}

// -----------------------------------------------------------------------------
void FPI::CalculateNDRanges() 
{
   // Evenly distribute the workload among the compute units (but avoiding local size
   // being more than the maximum allowed.
   m_local_size = std::min( m_max_local_size, 
                            (unsigned) ceil( m_num_points/(float) m_max_cu ) );

   m_global_size = m_num_points;

   // It is better to have global size divisible by local size
   if( m_global_size % m_local_size != 0 )
   {
      m_compile_flags += " -D NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE";
      // Round to the next divisible size (the kernel will ensure that
      // no access outside the population range will occur).
      m_global_size += m_local_size - (m_global_size % m_local_size); 
   }

   // OpenCL compiler flags
   m_compile_flags += " -D LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2=" 
                      + util::ToString( util::NextPowerOf2( m_local_size ) );

   if( MaximumTreeSize() > m_local_size )
      m_compile_flags += " -D PROGRAM_TREE_DOES_NOT_FIT_IN_LOCAL_SIZE";

   if( ! util::IsPowerOf2( m_local_size ) )
      m_compile_flags += " -D LOCAL_SIZE_IS_NOT_POWER_OF_2";
}

// -----------------------------------------------------------------------------
/** The FPI strategy needs a specialized EvaluatePopulation procedure because
    it is better to evaluate one program per kernel launch instead of the
    entire population as in the CPU, PPCU, and PPCE strategy.

    Using atomic add (for float) would allow us to easily evaluate the entire
    population at once, but atomic operations for float types are not easy/fast
    on the current generation of GPU hardware--and because that not fully
    supported by the current OpenCL specification.
 */ 
bool FPI::EvaluatePopulation( cl_uint* pop )
{
   // Write data to buffer (TODO: can we use mapbuffer here?)
   /* 
      What about creating two mapbuffers (to write directly to the device,
      using CL_MEM_ALLOC_HOST_PTR), one for each population (cur/tmp), and then
      passing them (alternately) when enqueueing the kernel? If mapbuffers mean
      that we can save a copy by writing directly to the device, then could we
      use efficiently this buffer in the host to access the populations, i.e.,
      does mapbuffer keep a copy (synchronized) in the host?
      */
   m_queue.enqueueWriteBuffer( m_buf_pop, CL_TRUE, 0, sizeof( cl_uint ) * 
         ( m_params->m_population_size * MaximumProgramSize() ), pop, NULL, NULL);

#ifdef PROFILING
   cl::Event e_time;
#endif
   
   // Evaluate one program per kernel execution
   for( unsigned p = 0; p < m_params->m_population_size; ++p )
   {
      // Pass the start position of the program we want to be evaluated
      uint program_id = (m_params->m_maximum_tree_size + 1) * p;
      m_kernel.setArg( 5, sizeof(uint), &program_id );

      // ---------- begin kernel execution
      m_queue.enqueueNDRangeKernel( m_kernel, cl::NDRange(), 
                       cl::NDRange( m_global_size ), cl::NDRange( m_local_size )
#ifdef PROFILING
            , NULL, &e_time
#endif
            );
      // ---------- end kernel execution

      // Wait until the kernel has finished
      m_queue.finish();

#ifdef PROFILING
      cl_ulong started, ended, enqueued;
      e_time.getProfilingInfo( CL_PROFILING_COMMAND_START, &started );
      e_time.getProfilingInfo( CL_PROFILING_COMMAND_END, &ended );
      e_time.getProfilingInfo( CL_PROFILING_COMMAND_QUEUED, &enqueued );

      ++m_kernel_calls;

      m_kernel_time += ended - started;
      m_launch_time += started - enqueued;
#endif

      // -----------------------------------------------------------------------
      // Mapping (there is one accumulated error per work-group)
      // The line below maps the contents of 'm_buf_E' into 'partial_errors'.
      cl_float* partial_errors = (cl_float*) m_queue.enqueueMapBuffer( m_buf_E,
            CL_TRUE, CL_MAP_READ, 0, (m_global_size / m_local_size) * sizeof(cl_float) );

      // Reduction on host! (we only need to perform 'work-group' adds)
      m_E[p] = 0.0f;
      for( unsigned i = 0; i < m_global_size/m_local_size; ++i ) m_E[p] += partial_errors[i];

      // Unmapping
      m_queue.enqueueUnmapMemObject( m_buf_E, partial_errors ); 
      // -----------------------------------------------------------------------

      // Check whether we have found a better solution
      if( m_E[p] < m_best_error  ||
        ( util::AlmostEqual( m_E[p], m_best_error ) && 
          ProgramSize( pop, p ) < ProgramSize( m_best_program ) ) )
      {
         m_best_error = m_E[p];
         Clone( Program( pop, p ), m_best_program );

         std::cout << "\nEvolved: [" << std::setprecision(12) << m_best_error << "]\t{" 
            << ProgramSize( m_best_program ) << "}\t";
         PrintProgramPretty( m_best_program );
         std::cout << "\n--------------------------------------------------------------------------------\n";
      }
      // TODO: Pick the best and fill the elitism vector (if any)

      // We should stop the evolution if an error below the specified tolerance is found
      return (m_best_error <= m_params->m_error_tolerance);
   }
}

// -----------------------------------------------------------------------------
