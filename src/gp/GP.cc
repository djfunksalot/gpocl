// -----------------------------------------------------------------------------
// $Id$
//
//   GP.cc
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

#include "GP.h"
#include "../common/util/Util.h"
#include "../common/util/Random.h"

#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <limits>
// FIXME:
#include <iomanip>

Params* GP::m_params = 0;

// -----------------------------------------------------------------------------
GP::GP( Params& p, cl_device_type device_type ): m_device_type( device_type ),
                                                 m_X( 0 ),
                                                 m_E( 0 ),
                                                 m_num_points( 0 ),
                                                 m_y_dim( 1 ), 
                                                 m_x_dim( 0 ),
#ifdef PROFILING
                                                 // Ensure that the performance counters are empty
                                                 m_kernel_time( 0 ),
                                                 m_launch_time( 0 ),
                                                 m_kernel_calls( 0 ),
                                                 m_node_evaluations( 0 ),
#endif
                                                 m_best_error( std::numeric_limits<cl_float>::max() )
{
   m_params = &p;

   // Random seed
   Random::Seed( (p.m_seed == 0 ? time( NULL ) : p.m_seed) );

   // Selection pressure [default = 2]
   if( m_params->m_selection_pressure <= 2 ) 
   {
      m_tournament_size = 2;
   }
   else // maximum = pop size
   {
      m_tournament_size = m_params->m_selection_pressure > m_params->m_population_size ?
         m_params->m_population_size : 
         m_params->m_selection_pressure;
   }

   // Create room for the best individual so far
   m_best_program = new cl_uint[MaximumProgramSize()];
   // Set its size as zero
   m_best_program[0] = 0;
}

// -----------------------------------------------------------------------------
void GP::Evolve()
{
   /*

      Pseudo-code for Evolve:

   1: Create (randomly) the initial population P
   2: Evaluate all individuals (programs) of P
   3: for generation ← 1 to NG do
      4: Copy the best (elitism) individuals of P to the temporary population Ptmp
      5: while |Ptmp| < |P| do
         6: Select and copy from P two fit individuals, p1 e p2
         7: if [probabilistically] crossover then
            8: Recombine p1 and p2, creating the children p1' and p2'
            9: p1 ← p1' ; p2 ← p2'
         10: end if
         11: if [probabilistically] mutation then
            12: Apply mutation operators in p1 and p2, generating p1' and p2'
            13: p1 ← p1' ; p2 ← p2'
         14: end if
         15: Insert p1 and p2 into Ptmp
      16: end while
      17: Evaluate all individuals (programs) of Ptmp
      18: P ← Ptmp; then discard Ptmp
   19: end for
   20: return the best individual so far
   */

#ifndef MAPPING
   m_E = new cl_float[ m_params->m_population_size ];
#endif

   cl_uint* pop_a = new cl_uint[ m_params->m_population_size * MaximumProgramSize() ];
   cl_uint* pop_b = new cl_uint[ m_params->m_population_size * MaximumProgramSize() ];

   cl_uint* cur_pop = pop_a;
   cl_uint* tmp_pop = pop_b;

   // 1:
   std::cout << "\n[Gen 1 of " << m_params->m_number_of_generations << "]...\n";
   InitializePopulation( cur_pop );
   // 2:
   ///EvaluatePopulation( cur_pop, errors );
   EvaluatePopulation( cur_pop );
   //std::cout << "\n";

   // 3:
   for( unsigned gen = 2; gen <= m_params->m_number_of_generations; ++gen )
   {
      std::cout << "\n[Gen " << gen << " of " << m_params->m_number_of_generations << "]...\n";
      // 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16:
      ///Breed( cur_pop, tmp_pop, errors );
      Breed( cur_pop, tmp_pop );
      // 17:
      ///if( EvaluatePopulation( tmp_pop, errors ) ) break;
      if( EvaluatePopulation( tmp_pop ) ) break;

      // 18:
      std::swap( cur_pop, tmp_pop );
   } // 19

   // 20:
   std::cout << "\n> Best program found: [" << m_best_error << "] ";
   PrintProgramPretty( m_best_program );
   std::cout << " [size: " << ProgramSize( m_best_program ) << "]\n";

#ifdef PROFILING
   std::cout << "\n> GPop/s: " << m_node_evaluations / (m_kernel_time/1.0E9) << " | Node evals: " << m_node_evaluations << " | Avg. KET: " << m_kernel_time / (m_kernel_calls * 1.0E6) << "ms | Avg. KLT: " << m_launch_time / (m_kernel_calls * 1.0E6) << "ms | Acc. KET: " << m_kernel_time/1.0E+9 << "s | Acc. KLT: " << m_launch_time/1.0E+9 << "s | Kernel calls: " << m_kernel_calls << "\n";
#endif
   std::cout << "> Strategy: "; PrintStrategy(); std::cout << "\n";

   // Clean up
   delete[] pop_a;
   delete[] pop_b;
   ///delete[] errors;
}

// -----------------------------------------------------------------------------
///void GP::Breed( cl_uint* old_pop, cl_uint* new_pop, const cl_float* errors )
void GP::Breed( cl_uint* old_pop, cl_uint* new_pop )
{
   // Elitism
   for( unsigned i = 0; i < m_params->m_elitism_size; ++i )
   {
      // FIXME: (use the vector of best individuals)
      Clone( m_best_program, Program( new_pop, i ) );

#ifdef PROFILING
      // Update the total number of nodes that are going to be evaluated (to be
      // used to calculate how many GPop/s we could achieve).
      m_node_evaluations += ProgramSize( new_pop, i );
#endif
   }

   // Tournament:
   for( unsigned i = m_params->m_elitism_size; i < m_params->m_population_size; ++i )
   {
      // Genetic operations
      if( Random::Probability( m_params->m_crossover_probability ) )
         // Respectively: mom, dad, and child
         Crossover( Program( old_pop, Tournament( old_pop ) ),
                    Program( old_pop, Tournament( old_pop ) ),
                    Program( new_pop, i ) );
      else
         Clone( Program( old_pop, Tournament( old_pop ) ), Program( new_pop, i ) );

      // Every genetic operator have a chance to go wrong
      if( Random::Probability( m_params->m_mutation_probability ) )
      {
         if( Random::Probability( 0.5 ) )
            NodeMutate( Program( new_pop, i ) );
         else
            SubTreeMutate( Program( new_pop, i ) );
      }

#ifdef PROFILING
      // Update the total number of nodes that are going to be evaluated (to be
      // used to calculate how many GPop/s we could achieve).
      m_node_evaluations += ProgramSize( new_pop, i );
#endif
/*#ifndef NDEBUG
      std::cout << std::endl;
      PrintProgram( Program( old_pop, i ) );
      std::cout << std::endl;
      PrintProgram( Program( new_pop, i ) );
      std::cout << std::endl;
#endif*/
   }
}

void GP::Crossover( const cl_uint* mom, const cl_uint* dad, cl_uint* child ) const
{
   assert( mom != NULL && dad != NULL && child != NULL && child != mom && child != dad );

   unsigned pt_mom;
   unsigned pt_dad;
   unsigned mom_subtree_size;
   unsigned dad_subtree_size;
   unsigned child_program_size;
   register unsigned i,j;

   // Choose cut points (mom/dad) that don't go beyond the maximum tree size
   do
   {
      pt_mom = Random::Int( 1, ProgramSize( mom ) );
      pt_dad = Random::Int( 1, ProgramSize( dad ) );
      mom_subtree_size = TreeSize( mom + pt_mom );
      dad_subtree_size = TreeSize( dad + pt_dad );

      child_program_size = ProgramSize( mom ) - mom_subtree_size + dad_subtree_size;
   } while( child_program_size > MaximumTreeSize() );

   SetProgramSize( child, child_program_size );

   /*
   std::cout << "\n\npt_mom: " << pt_mom << " pt_dad: " << pt_dad;
   std::cout << "\nMom: "; PrintProgram( mom );
   std::cout << "\nDad: "; PrintProgram( dad );
   */
   // Actual crossover
   for( i = 1; i < pt_mom; i++ )
   {
      *(child + i) = *(mom + i);
   }
   for( j = pt_dad; j < pt_dad + dad_subtree_size; j++ )
   {
      *(child + i) = *(dad + j);

      i++;
   }
   for( j = pt_mom + mom_subtree_size; j < ProgramSize( mom ) + 1; j++ )
   {
      *(child + i) = *(mom + j);

      i++;
   }
  // std::cout << "\nSon: "; PrintProgram( child );

   assert( TreeSize( child + 1 ) == ProgramSize( child ) );
}

// -----------------------------------------------------------------------------
void GP::SubTreeMutate( cl_uint* program ) const
{
   // Copy the size (CopyNodeMutate, differently from CopySubTreeMutate doesn't
   // change the actual size of the program)
   assert( program != NULL );
   assert( ProgramSize( program ) <= MaximumTreeSize() );

   //unsigned size = ProgramSize( program );
   // Pos 0 is the program size; pos 1 is the first node and 'program size + 1'
   // is the last node.
   unsigned mutation_pt = Random::Int( 1, ProgramSize( program ) ); // [1, size] (inclusive)

   //                   (mutation point)
   //                          v
   // [ ]     [ ]     [ ]     [*]     [*]     [*]     [ ]    [ ]
   // |      first      |     | mutated subtree |     | second |

   // Create a new random subtree of same size of the original one and put it
   // in the corresponding place in program_dest
   unsigned subtree_size = TreeSize( program + mutation_pt );
   unsigned new_subtree_size = Random::Int( 1, 
            MaximumTreeSize() - ( ProgramSize( program ) - subtree_size ) );
  
   //std::cout << "\n\npt: " << mutation_pt << " ss: " << subtree_size << " nss: " << new_subtree_size;
   //std::cout << "\n1: "; PrintProgram( program );

   // Set the resulting tree size to the newly generated program
   SetProgramSize( program, ProgramSize( program ) + (new_subtree_size - subtree_size) );

   // Move the second fragment (if necessary)
   if( new_subtree_size != subtree_size )
   {
      if( new_subtree_size < subtree_size )
      {
         for( unsigned i = mutation_pt + new_subtree_size; i < ProgramSize( program ) + 1; ++i )
         {
            program[i] = program[i + (subtree_size - new_subtree_size)];
         }
      }
      else
      {
         for( unsigned i = ProgramSize( program ); i >= mutation_pt + new_subtree_size; --i )
         {
            program[i] = program[i - (new_subtree_size - subtree_size)];
         }
      }
   }

   CreateLinearTree( program + mutation_pt, new_subtree_size );
   //std::cout << "\n2: "; PrintProgram( program ); std::cout << std::endl;
}

// -----------------------------------------------------------------------------
void GP::NodeMutate( cl_uint* program ) const
{
   // Copy the size (CopyNodeMutate, differently from CopySubTreeMutate doesn't
   // change the actual size of the program)
   assert( program != NULL );
   assert( ProgramSize( program ) <= MaximumTreeSize() );

   // Pos 0 is the program size; pos 1 is the first node and 'program size + 1'
   // is the last node.
   unsigned mutation_point = Random::Int( 1, ProgramSize( program ) ); // [1, size] (inclusive)

   //                      (mutation point)
   //                             v
   // [ ]     [ ]     [ ]        [*]         [ ]     [ ]     [ ]
   // |      first      |   | mutated pt |   | second          |

   // Mutate the node by a random node of the same arity (remember, this is *node*
   // mutation!).
   program[mutation_point] = m_primitives.RandomNode( ARITY( program[mutation_point] ),
                                                      ARITY( program[mutation_point] ) );
}

void GP::CopySubTreeMutate( const cl_uint* program_orig, cl_uint* program_dest ) const
{
#define CAN_CHANGE_ORIGINAL_SIZE 1

   assert( program_orig != NULL && program_dest != NULL && program_orig != program_dest );
   assert( ProgramSize( program_orig ) <= MaximumTreeSize() );

   // Pos 0 is the program size; pos 1 is the first node and 'program size + 1'
   // is the last node.
   unsigned mutation_point = Random::Int( 1, ProgramSize( program_orig ) ); // [1, size] (inclusive)

   //                   (mutation point)
   //                          v
   // [ ]     [ ]     [ ]     [*]     [*]     [*]     [ ]    [ ]
   // |      first      |     | mutated subtree |     | second |

   // Copy the first fragment but not the size (for now).
   for( unsigned i = 1; i < mutation_point; ++i )
      program_dest[i] = program_orig[i];

   // Create a new random subtree of same size of the original one and put it
   // in the corresponding place in program_dest
   unsigned subtree_size = TreeSize( &program_orig[mutation_point] );
#ifdef CAN_CHANGE_ORIGINAL_SIZE
   unsigned new_subtree_size = Random::Int( 1, 
         MaximumTreeSize() - ( ProgramSize( program_orig ) - subtree_size ) );
#else
   unsigned new_subtree_size = subtree_size;
#endif

   CreateLinearTree( &program_dest[mutation_point], new_subtree_size );

   // Continue to copy the second fragment
   for( unsigned i = mutation_point + subtree_size; i < ProgramSize( program_orig ) + 1; ++i )
      program_dest[i + (new_subtree_size - subtree_size)] = program_orig[i];

   // Finally, set the resulting tree size to the newly generated program
   SetProgramSize( program_dest, ProgramSize( program_orig ) + 
                                (new_subtree_size - subtree_size ) );
}

// -----------------------------------------------------------------------------
void GP::CopyNodeMutate( const cl_uint* program_orig, cl_uint* program_dest ) const
{
   // Copy the size (CopyNodeMutate, differently from CopySubTreeMutate doesn't
   // change the actual size of the program)
   assert( program_orig != NULL && program_dest != NULL && program_orig != program_dest );
   assert( ProgramSize( program_orig ) <= MaximumTreeSize() );

   unsigned size = *program_orig;
   // Pos 0 is the program size; pos 1 is the first node and 'program size + 1'
   // is the last node.
   unsigned mutation_point = Random::Int( 1, size ); // [1, size] (inclusive)

   //                      (mutation point)
   //                             v
   // [ ]     [ ]     [ ]        [*]         [ ]     [ ]     [ ]
   // |      first      |   | mutated pt |   | second          |

   // Copy the size (pos 0) and then the first fragment (until just before the mutation point) 
   for( unsigned i = 0; i < mutation_point; ++i )
      program_dest[i] = program_orig[i];

   // Mutate the node by a random node of the same arity (remember, this is *node*
   // mutation!).
   program_dest[mutation_point] = m_primitives.RandomNode( ARITY( program_orig[mutation_point] ),
                                                          ARITY( program_orig[mutation_point] ) );

   // Continue to copy the second fragment
   for( unsigned i = mutation_point + 1; i < size + 1; ++i )
      program_dest[i] = program_orig[i];
}

// -----------------------------------------------------------------------------
void GP::Clone( cl_uint* program_orig, cl_uint* program_dest ) const
{
   assert( program_orig != NULL && program_dest != NULL );
   assert( *program_orig <= MaximumTreeSize() );

   // The size is the first element
   for( unsigned i = *program_orig + 1; i-- ; )
      *program_dest++ = *program_orig++;
}

// -----------------------------------------------------------------------------
///bool GP::EvaluatePopulation( cl_uint* pop, cl_float* errors )
bool GP::EvaluatePopulation( cl_uint* pop )
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
   m_queue.enqueueWriteBuffer( m_buf_pop, CL_TRUE, 0, 
         sizeof( cl_uint ) * ( m_params->m_population_size * MaximumProgramSize() ),
         pop, NULL, NULL);

#ifdef PROFILING
   cl::Event e_time;
#endif
   
   // ---------- begin kernel execution
   m_queue.enqueueNDRangeKernel( m_kernel, cl::NDRange(), 
                                 cl::NDRange( m_num_global_wi ), cl::NDRange( m_num_local_wi )
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

   // TODO: Do I need to always Map and Unmap?
#ifdef MAPPING
   m_E = (cl_float*) m_queue.enqueueMapBuffer( m_buf_E, CL_TRUE, 
   // FIXME: Fix ppcu kernel (add prefix sum) to support this change
         CL_MAP_READ, 0, m_params->m_population_size * sizeof(cl_float) );
         //CL_MAP_READ, 0, m_num_local_wi * m_params->m_population_size * sizeof(cl_float) );
#else
   m_queue.enqueueReadBuffer( m_buf_E, CL_TRUE, 0,
   // FIXME: Fix ppcu kernel (add prefix sum) to support this change
         m_params->m_population_size * sizeof(cl_float),
         //m_num_local_wi * m_params->m_population_size * sizeof(cl_float),
         m_E, NULL, NULL );
#endif

   // --- Fitness calculation -----------------

   for( unsigned i = 0; i < m_params->m_population_size; ++i )
   {
   // FIXME: Fix ppcu kernel (add prefix sum) to support this change
#if 0
      errors[i] = 0.0f;

//      std::cout << "\nProgram: [" << i << "] ";
 //     PrintProgramPretty( Program( pop, i )  );
  //    std::cout << " (size: " << ProgramSize( Program( pop, i ) ) << ")\n";

      for( unsigned j = 0; j < m_num_local_wi; ++j )
      {
         // Sum of the squared error
   //      std::cout << std::setprecision(16) << "[" << m_E[i * m_num_local_wi + j] << ", " << errors[i] << "]";
         errors[i] += m_E[ i * m_num_local_wi + j ];
      }
    //  std::cout << "\n";

      if( isinf( errors[i] ) || isnan( errors[i] ) )
      {
         // Set the worst error possible
         errors[i] = std::numeric_limits<cl_float>::max();

         continue;
      } else
         errors[i] /= m_num_points; // take the mean


      // Check whether we have found a better solution
      if( errors[i] < m_best_error  ||
         ( util::AlmostEqual( errors[i], m_best_error ) && ProgramSize( pop, i ) < ProgramSize( m_best_program ) ) )
      {
         m_best_error = errors[i];
         Clone( Program( pop, i ), m_best_program );

         std::cout << "Found better: [" << m_best_error << "] ";
         //PrintProgram( m_best_program );
         PrintProgramPretty( m_best_program );
         std::cout << " (size: " << ProgramSize( m_best_program ) << ")\n";
      }
#endif
      // Check whether we have found a better solution
      if( m_E[i] < m_best_error  ||
         ( util::AlmostEqual( m_E[i], m_best_error ) && ProgramSize( pop, i ) < ProgramSize( m_best_program ) ) )
      {
         m_best_error = m_E[i];
         Clone( Program( pop, i ), m_best_program );

         std::cout << "Found better: [" << m_best_error << "] ";
         //PrintProgram( m_best_program );
         PrintProgramPretty( m_best_program );
         std::cout << " (size: " << ProgramSize( m_best_program ) << ")\n";
      }
      // TODO: Pick the best and fill the elitism vector (if any)
   }
#ifdef MAPPING
   m_queue.enqueueUnmapMemObject( m_buf_E, m_E );
#endif

   // We should stop the evolution if an error below the specified tolerance is found
   return (m_best_error <= m_params->m_error_tolerance);
   
}
// -----------------------------------------------------------------------------
///unsigned GP::Tournament( const cl_uint* pop, const cl_float* errors ) const
unsigned GP::Tournament( const cl_uint* pop ) const
{
   unsigned winner = Random::Int( 0, m_params->m_population_size - 1 );
   for( unsigned t = 0; t < m_tournament_size; ++t ) 
   {
      unsigned competitor = Random::Int( 0, m_params->m_population_size - 1 );
      if( m_E[competitor] < m_E[winner]
            || ( util::AlmostEqual( m_E[competitor], m_E[winner] ) &&
               ProgramSize( pop, competitor ) < ProgramSize( pop, winner ) ) )
      {
         winner = competitor;
      }
   }

   return winner;
}

// -----------------------------------------------------------------------------
void GP::OpenCLInit()
{
   // TODO: either (i) use the best device (fastest?) or (ii) use *all* devices!
   cl_int device_to_use = 0;

   std::vector<cl::Platform> platforms;
   cl::Platform::get( &platforms );

   cl_context_properties properties[] = 
   { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
   m_context = cl::Context( m_device_type, properties );

   //std::vector<cl::Device> devices;
   m_device = m_context.getInfo<CL_CONTEXT_DEVICES>()[device_to_use];

   m_max_cu = m_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

   m_max_wg_size = m_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
   m_max_wi_size = m_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];

   std::cerr << "Max CU: " << m_max_cu << " WGS: " << m_max_wg_size << " WIS[0]:" << m_max_wi_size <<  std::endl;

#ifdef PROFILING
   m_queue = cl::CommandQueue( m_context, m_device, CL_QUEUE_PROFILING_ENABLE );
#else
   m_queue = cl::CommandQueue( m_context, m_device, 0 );
#endif

}

// -----------------------------------------------------------------------------
void GP::CreateBuffers()
{
   //TODO: Optimize for CPU (USE_HOST_PTR?)

   // Buffer (memory on the device) of training points
   // TODO: I think m_X can be freed right after cl::Buffer returns. Check that!
   std::cerr << "Trying to allocate " << sizeof( cl_float ) * m_num_points * m_x_dim << " bytes\n";
   m_buf_X = cl::Buffer( m_context,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof( cl_float ) * m_num_points * m_x_dim,
                         m_X );
   std::cerr << "Trying to allocate " << sizeof( cl_float ) * m_num_points << " bytes\n";
   m_buf_Y = cl::Buffer( m_context,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof( cl_float ) * m_num_points, &m_Y[0] );

   // Buffer (memory on the device) of partial errors
   std::cerr << "Trying to allocate " << sizeof( cl_float ) * m_num_local_wi * m_params->m_population_size << " bytes\n";
   m_buf_E = cl::Buffer( m_context,
#ifdef MAPPING
                         CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
#else
                         CL_MEM_WRITE_ONLY,
#endif
                         m_num_local_wi * m_params->m_population_size * sizeof(cl_float) );

  /* 
   Structure of a program (individual)

     |                         |
+----+-----+----+--------------+-------------
|size|arity|type| index/value  |  ...
| 32 |  3  | 7  |     22       |
+----+-----+----+--------------+-------------
     |    first element        | second ...
  */
   // Buffer (memory on the device) of the population
   std::cerr << "Trying to allocate " << sizeof( cl_uint ) * ( m_params->m_population_size * MaximumProgramSize() )  << " bytes\n";
   m_buf_pop = cl::Buffer( m_context,
                           CL_MEM_READ_ONLY,
                           sizeof( cl_uint ) * ( m_params->m_population_size * 
                                               MaximumProgramSize() ) );
}

// -----------------------------------------------------------------------------
void GP::BuildKernel()
{
   /* To avoid redundant switch cases in the kernel, we will add only those cases clauses 
      that correspond to the subset of primitives given by the user. */
   std::string interpreter = "#define INTERPRETER_CORE";
   for( unsigned i = 0; i < m_primitives.m_primitives.size(); ++i )
      if( INDEX( m_primitives.m_primitives[i] ) != Primitives::GPT_VAR ) 
      {
         /* TODO: Check for duplicates! The user may have given duplicated
          * primitives (it is allowed to do that). */
         interpreter += " case " + util::ToString( INDEX( m_primitives.m_primitives[i] ) ) + ":"
            + "PUSH(" + util::ToString( m_primitives.DB[INDEX( m_primitives.m_primitives[i] )].arity ) 
                      + "," + m_primitives.DB[INDEX( m_primitives.m_primitives[i] )].code + ") break;";
            //+ "PUSH(" + m_primitives.DB[INDEX( m_primitives.m_primitives[i] )].code + ");break;";
      }
   interpreter += "\n";

   // program_src = header + kernel
   std::string program_src = 
      "#define WGS " + util::ToString( m_num_local_wi ) + "\n" +
      "#define POP_SIZE " + util::ToString( m_params->m_population_size ) + "\n" +
      "#define NUM_POINTS " + util::ToString( m_num_points ) + "\n"
      "#define X_DIM " + util::ToString( m_x_dim ) + "\n"
      "#define MAX_TREE_SIZE " + util::ToString( MaximumTreeSize() ) + "\n" +
      "#define MAX_FLOAT " + util::ToString( std::numeric_limits<cl_float>::max() ) + "f\n" +
      "#define TOP       ( stack[stack_top] )\n"
      "#define POP       ( stack[stack_top--] )\n"
 //     "#define PUSH( i ) ( stack[++stack_top] = (i) )\n"
      "#define PUSH(arity, exp) stack[stack_top + 1 - arity] = (exp); stack_top = stack_top + 1 - arity;\n"
      "#define ARG(n) (stack[stack_top - n])\n"
      "#define CREATE_STACK( type, size ) type stack[size]; int stack_top = -1;\n"
      "#define NODE program[op]\n"
      + interpreter + m_kernel_src;

  // std::cerr << std::endl;
   //std::cerr << program_src;
  // std::cerr << std::endl;

   //--------------------

   cl::Program::Sources sources( 1, std::make_pair(  program_src.c_str(), program_src.size() ));

   cl::Program program( m_context, sources );

   // TODO: currently not elegant!
   std::vector<cl::Device> devices; devices.push_back( m_device );
   try {
      program.build( devices );
   } catch( cl::Error& e ) {
      if( e.err() == CL_BUILD_PROGRAM_FAILURE )
      {
         std::string str =
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>( m_device );
         std::cerr << "Program Info: " << str << std::endl;
      }

      throw;
   }


   // FIXME:
	std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
	std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
	std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;

   m_kernel = cl::Kernel( program, "evaluate" );
}

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
   // GP::LoadPoints( m_points );
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
   for( unsigned i = 0; i < m_num_points; ++i)
      std::cout << m_Y[i] << " ";
    */
}

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
void GP::LoadKernel( const char* kernel_file )
{
   std::ifstream file( kernel_file );
   m_kernel_src.append( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()) );
}

//////////////////////////// Genetic Operations ///////////////////////////////
// -----------------------------------------------------------------------------
void GP::InitializePopulation( cl_uint* pop )
{
   for( unsigned i = 0; i < m_params->m_population_size; ++i )
   {
      // TODO: how about using a normal distribution (with mean = max_gen_size/2)?
      cl_uint tree_size = Random::Int( 1, MaximumTreeSize() );

      cl_uint* program = Program( pop, i );

      // The first "node" is the program's size
      SetProgramSize( program, tree_size );

      CreateLinearTree( ++program, tree_size );

#ifdef PROFILING
      // Update the total number of nodes that are going to be evaluated (to be
      // used to calculate how many GPop/s we could achieve).
      m_node_evaluations += tree_size;
#endif
   }
}

// -----------------------------------------------------------------------------
void GP::PrintProgram( const cl_uint* program ) const
{
   PrintTree( program + 1 );
}

void GP::PrintProgramPretty( const cl_uint* program, int start, int end ) const
{
   if( (start == -1) || (end == -1) ) { start = 0; end = ProgramSize( program++ ) - 1; }
   
   if( ARITY( *(program + start) ) == 0 )
   {
      PrintNode( program + start );
      return;
   }
   else
   {
      PrintNode( program + start ); std::cout << "( ";
   }

   int i;
   start++;
   while( start <= end )
   {
      i = TreeSize( program + start );
      PrintProgramPretty( program, start, ( i > 1 ) ? start + i - 1 : end );
      start += i; 
      
      /* Put the trailing ")" */
      if( start <= end ) std::cout << ", "; else std::cout << " )";
   }
   return;
}

void GP::PrintNode( const cl_uint* node ) const
{
   switch( INDEX( *node ) )
   {
      case Primitives::GPT_VAR:
         std::cout << "X" << AS_INT( *node ) << "";
         break;
      case Primitives::GPT_EPHEMERAL:
         std::cout << AS_FLOAT( *node ) << "";
         break;
      case Primitives::GPF_IDENTITY:
         std::cout << "";
         break;
      default:
         std::cout << m_primitives.DB[INDEX(*node)].name << "";
   }
}

void GP::PrintTree( const cl_uint* node ) const
{
   int sum = 0;

   do {
      PrintNode( node );
      sum += ARITY( *node++ ) - 1;
   } while( sum != -1 );
}

// -----------------------------------------------------------------------------
void GP::CreateLinearTree( cl_uint* node, unsigned size ) const
{
   assert( size >= 1 );
   assert( node != 0 );

   unsigned open = 1;

   do {
      if( open == size || open > 1 )
         /* 
            [open == size] When the number of open arguments is equal the
            number of left nodes, then the only valid choice is a terminal
            (RandomNode( 0, 0 )), otherwise we would end up with a program
            greater than its size.

            [open > 1] When the number of open arguments is greater than one,
            then we can allow terminals to be chosen because they will not
            prematurely end the program.
          */
         *node = m_primitives.RandomNode( 0, size - open );
      else
         /* This means that 'open == 1' and 'size > 1', so we cannot choose
            a terminal here because we would end up with a shorter program. */
         *node = m_primitives.RandomNode( 1, size - open );

      /* Whenever we put a new operator/operand, the number of open arguments
         decreases. However, if the new operator requires more than one argument
         (arity >= 2) then we end up increasing the current number of open arguments.
       */
      open += ARITY( *node++ ) - 1;
   } while( --size );
}

// -----------------------------------------------------------------------------
void GP::LoadPoints( std::vector<std::vector<cl_float> > & out_x )
{
   using namespace util;

   // We will consider just the first file name given by the user
   std::ifstream points( m_params->m_data_points[0].c_str() );

   if( !points.is_open() ) {
      // Maybe a typo when passing the file name on the command-line
      throw Error( "[" + m_params->m_data_points[0] + "]: file not found." );
	}

   
   
  // m_Y = new cl_float[ 16 ];



   unsigned cur_line = 0; std::string line;
   while( std::getline( points, line ) )
   {
      ++cur_line;

      // Skipping empty lines or lines beginning with '#'
      if( line.empty() || line[0] == '#' ) continue;

      std::stringstream ss( line ); std::string cell; std::vector<cl_float> v;
      while( std::getline( ss, cell, ',' ) )
      {
         cl_float element;
         if( !StringTo( element, cell ) )
            // This means that 'cell' couldn't be converted to a float numeral type. It is
            // better to throw a fatal error here and let the user fix the dirty file.
            throw Error( "[" + m_params->m_data_points[0] + ", line " + ToString( cur_line ) 
                  + "]: could not convert '" + ToString( cell ) + "' to a float point number." );

         // Appending the cell to the temporary float vector 'v'
         v.push_back( element );
      }

      // Setting the dimension of the input variables (X) and checking whether
      // there are lines with different number of variables.
      if( v.size() != m_x_dim + m_y_dim )
      {
         if( v.empty() || m_x_dim != 0 )
            // Ops. We've found a line with a different number of variables!
            throw Error( "[" + m_params->m_data_points[0] + ", line " + ToString( cur_line ) + "]: expected " 
                         + ToString( m_x_dim + m_y_dim ) + "variables but found " + ToString( v.size() ) );
         else
            // For the first time, since m_x_dim is currently not set, we must set
            // its value. So, the actual dimension of X (input) is the one found
            // on the first line.
            m_x_dim = v.size() - m_y_dim;
      }

      // Here we append directly in m_Y because both CPU and GPU we use it (m_Y) throughout
      // the evolutionary process.



  //    m_Y[out_x.size()] = v.back(); v.pop_back();
      m_Y.push_back( v.back() ); v.pop_back();
      


      out_x.push_back( v );
   }

   if( out_x.empty() )
      throw Error( "[" + m_params->m_data_points[0] + "]: no data found." );

   m_num_points = out_x.size();
}
