// -----------------------------------------------------------------------------
// $Id$
//
//   GP.h
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

#ifndef GP_hh
#define GP_hh

///////////// Definitions //////////////

/*
   When defined profiling informations about kernel execution are taken.
*/
#define PROFILING 1

/*
   When MAPPING is defined the output values (predictions) from the kernel
   executions are read via mapping, which is supposed faster than explicit
   copies.
*/
//#define MAPPING 1
////////////////////////////////////////

#define __CL_ENABLE_EXCEPTIONS
//TODO: #define CL_DEVICE_FISSION

#include <CL/cl.hpp>

#include "Params.h"
#include "../common/Exception.h"
#include "../common/util/Util.h"

#include "Primitives.h"

// Functions definition
#include "kernels/common.cl"

#include <string>
#include <iostream>
#include <vector>
#include <cassert>

/*
Definitions:
   Node: an individual cl_uint packing three elements: 
        (1) arity, (2) index, and possibly (3) a value (integer or real).
   Tree: a set of nodes representing a complete (sub)tree.
   Program: the size of a tree and the tree itself

   Example:

   {|   3    || 2, +,  | 0, var, X0 | 0, var, X1 |}
    |_ size _|         |_ 2nd node _|
              |___________ tree _________________|
    |________________ program ___________________|
*/


/*
FIXME:

On the GPU:
- population_size must be divided by work_group_size (usually 256)
- num_points must be divided by work_group_size
*/

// -----------------------------------------------------------------------------
class GP {
public:
   /**
    * @class Error
    *
    * @brief Class for GP's fatal exceptions.
    */
   struct Error: public Exception {
      Error( const std::string& msg ): Exception( "@ GP ", msg ) {};
   };

public:
   // Get platform, context, devices and create queue
   GP( Params& p, cl_device_type );
   virtual ~GP() 
   { 
      if( m_X ) delete[] m_X;
#ifndef MAPPING
      if( m_E ) delete[] m_E;
#endif
      delete[] m_best_program;
   }

   void Run()
   {
      if( m_params->m_print_primitives )
      {
         m_primitives.ShowAvailablePrimitives();

         return;
      }

      // [virtual] Load training points (CPU != GPU)
      LoadPoints();

      // Load primitives
      m_primitives.Load( m_x_dim, m_params->m_maximum_tree_size, m_params->m_primitives );

      // Create context/devices
      // Create queue
      OpenCLInit();

      // [virtual]
      CalculateNDRanges();
      std::cout << "NDRanges: [local: " << m_num_local_wi << ", global: " << m_num_global_wi << "]\n";

      // Create buffers
      CreateBuffers();

      // Create program ("build kernel")
      BuildKernel();

      SetKernelArgs();

      Evolve();
   }

   virtual void PrintStrategy() const = 0;

protected:
   virtual void Evolve();

   unsigned MaximumProgramSize() const { return m_params->m_maximum_tree_size + 1; }
   unsigned MaximumTreeSize() const { return m_params->m_maximum_tree_size; }
   /**
     Return the size of program pointed by 'g'. The size of
     a program is stored at its first position.
    */
   unsigned ProgramSize( const cl_uint* program ) const 
   { 
      assert( *program <= MaximumProgramSize() );
      return *program; 
   };
   unsigned ProgramSize( const cl_uint* pop, unsigned i ) const 
   { 
      return ProgramSize( Program( pop, i ) );
   }
   /**
     Return the i-th program of population 'p'
     */
   cl_uint* Program( cl_uint* pop, unsigned i ) const 
   {
      return pop + (i * (m_params->m_maximum_tree_size + 1) ); 
   }
   /**
     Return the i-th tree of population 'p'
     */
   cl_uint* Tree( cl_uint* pop, unsigned i ) const 
   {
      return Program( pop, i ) + 1; 
   }

   /**
     Return the i-th program of population 'p' (const version)
     */
   const cl_uint* Program( const cl_uint* pop, unsigned i ) const 
   {
      return pop + (i * (m_params->m_maximum_tree_size + 1) ); 
   }
   /**
     Return the i-th tree of population 'p' (const version)
     */
   const cl_uint* Tree( const cl_uint* pop, unsigned i ) const 
   {
      return Program( pop, i ) + 1; 
   }

   void SetProgramSize( cl_uint* program, unsigned size ) const { *program = size; }

   ///bool EvaluatePopulation( cl_uint* pop, cl_float* errors );
   bool EvaluatePopulation( cl_uint* pop  );
   void InitializePopulation( cl_uint* pop );
   ///void Breed( cl_uint* old_pop, cl_uint* new_pop, const cl_float* );
   void Breed( cl_uint* old_pop, cl_uint* new_pop );
   void Clone( cl_uint* program_orig, cl_uint* program_dest ) const;
   ///unsigned Tournament( const cl_uint* pop, const cl_float* errors ) const;
   unsigned Tournament( const cl_uint* pop ) const;
   void Crossover( const cl_uint* mom, const cl_uint* dad, cl_uint* child ) const;
   /**
     Copy the individual @ref program_orig into @ref program_dest but
     with a random subtree mutated--a random subtree of same size is
     created and put in @ref program_dest replacing the corresponding
     subtree in @ref program_orig.
    */
   void CopySubTreeMutate( const cl_uint* program_orig, cl_uint* program_dest ) const;
   /**
     Like CopySubTreeMutate but it does the mutation in loco.
    */
   void SubTreeMutate( cl_uint* program ) const;
   /**
     Copy the individual @ref program_orig into @ref program_dest but
     with a random node mutated.
    */
   void CopyNodeMutate( const cl_uint* program_orig, cl_uint* program_dest ) const;
   /**
     Like CopyNodeMutate but it does the mutation in loco.
    */
   void NodeMutate( cl_uint* program ) const;

   /**
     Create a linear tree of the exactly given size, starting at 'node'.
     */
   void CreateLinearTree( cl_uint* node, unsigned size ) const;

   void PrintProgram( const cl_uint* program ) const;
   void PrintProgramPretty( const cl_uint* program, int start = -1, int end = -1 ) const;
   void PrintTree( const cl_uint* node ) const;
   void PrintNode( const cl_uint* node ) const;

   unsigned TreeSize( const cl_uint* node ) const
   {
      /* We have a valid tree when the sum of the arity minus one equals to -1 */
      unsigned size = 0; int sum = 0;
      do {
         ++size;
         sum += ARITY( *node++ ) - 1;
      } while( sum != -1 );

      return size;
   }

   virtual void LoadPoints() = 0;

   cl_uint* m_best_program;
   cl_float m_best_error;

   Primitives m_primitives;
protected:
   // OpenCL related functions

   void OpenCLInit();
   void CreateBuffers();
   void BuildKernel();
   virtual void SetKernelArgs()
   {
      // Set common kernel arguments
      m_kernel.setArg( 0, m_buf_pop );
      m_kernel.setArg( 1, m_buf_X );
      m_kernel.setArg( 2, m_buf_Y );
      m_kernel.setArg( 3, m_buf_E );
   }

   virtual void CalculateNDRanges() = 0;
protected:
   // OpenCL related data

   cl_device_type m_device_type;
   cl::CommandQueue m_queue;
   cl::Context m_context;
   cl::Device m_device;

   cl::Buffer m_buf_E;
   cl::Buffer m_buf_X;
   cl::Buffer m_buf_Y;
   cl::Buffer m_buf_pop;

   std::string m_kernel_src;
   cl::Kernel m_kernel;

   size_t m_max_cu;
   size_t m_max_wg_size;
   size_t m_max_wi_size;

   size_t m_num_global_wi;
   size_t m_num_local_wi;

   std::string m_compile_flags;
#ifdef PROFILING
   cl_ulong m_kernel_time;
   cl_ulong m_launch_time;
   cl_uint m_kernel_calls;

   unsigned long m_node_evaluations;
#endif
protected:
   cl_float* m_E; /**< Array of partial errors. */
   cl_float* m_X; /**< Linear version of the original data points (will be
                   transposed on the GPU for better access pattern). */
protected:
   /** Load data (training) points in CSV format. The contents are
    put in the given argument (matrix). */
   void LoadPoints( std::vector<std::vector<cl_float> > & );

   //virtual void EvaluatePop() {};
   unsigned m_tournament_size; /**< Tournament size. */


   void LoadKernel( const char* file );
public:
   unsigned m_num_points; /**< Total number of data (training) points. */
   unsigned m_x_dim; /**< Number of input variables. */
   unsigned m_y_dim; /**< Number of output variables. Currently, always = 1. */

   

   std::vector<cl_float> m_Y;
   //cl_float* m_Y;

   static Params* m_params; /**< Pointer to Params class (holds the parameters). */
};

// -----------------------------------------------------------------------------
#endif
