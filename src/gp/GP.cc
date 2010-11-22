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

/*
   There are five valid combinations of the following three flags:
   CL_MEM_ALLOC_HOST_PTR, CL_MEM_COPY_HOST_PTR, and CL_MEM_USE_HOST_PTR.

   The combinations are: (1) No flags specified, (2) CL_MEM_COPY_HOST_PTR, (3)
   CL_MEM_ALLOC_HOST_PTR, (4) CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, and
   (5) CL_MEM_USE_HOST_PTR.

   The first two, (1) No flags specified, and (2) CL_MEM_COPY_HOST_PTR, are
   non-mappable and require clEnqueueReadBuffer, and clEnqueueWriteBuffer to
   typically transfer data to/from the host from/to the device. The next two,
   (3) CL_MEM_ALLOC_HOST_PTR, (4) CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
   are mappable and can use clEnqueueMapBuffer, and clEnqueueUnmapMemObject.
   The last, (5) CL_MEM_USE_HOST_PTR, is also mappable and should use
   clEnqueueMapBuffer, and clEnqueueUnmapMemObject.

   If you are porting an existing application which already allocates its own
   buffers, then the fifth combination, CL_MEM_USE_HOST_PTR, may be your only
   choice. However, this might not be the best performance because the buffer
   may need to be copied from/to host memory to/from device memory. It is
   generally felt that the first, (1) No flags specified, and third, (3)
   CL_MEM_ALLOC_HOST_PTR, combinations are better because they do not require a
   copy and because the buffer can be allocated internally with constraints
   such as alignment, etc. Depending upon the overhead for a "bulk" read/write
   data transfer versus a data transfer for each mapped access can help decided
   between using non-mappable (1) and mappable (3) combinations.

   Naturally this depends upon the performance characteristics of each vendor's
   hardware and software implementation, that is, your mileage may vary
   depending upon the OpenCL that you are using.
   */

#include "GP.h"
#include "../common/util/Util.h"
#include "../common/util/Random.h"

#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <cassert>

Params* GP::m_params = 0;

// -----------------------------------------------------------------------------
GP::GP( Params& p, cl_device_type device_type ): m_device_type( device_type ),
                                                 m_X( 0 ),
                                                 m_predicted_Y( 0 ),
                                                 m_num_points( 0 ),
                                                 m_y_dim( 1 ), 
                                                 m_x_dim( 0 )
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


}

// -----------------------------------------------------------------------------
void GP::Evolve()
{
   /*
   1: Criar aleatoriamente a populacao inicial P
   2: Avaliar todos indivıduos de P
   3: for geracao ← 1 to NG do
      4: Copiar a elite de individuos de P para a populacao temporaria Ptmp
      5: while |Ptmp| < |P| do
         6: Selecionar e copiar de P dois individuos bem adaptados, p1 e p2
         7: if [probabilisticamente] cruzamento then
            8: Cruzar p1 com p2 , gerando os descendentes p1' e p2'
            9: p1 ← p1' ; p2 ← p2'
         10: end if
         11: if [probabilisticamente] mutacao then
            12: Aplicar operadores quaisquer de mutacao em p1 e p2, gerando p1' e p2'
            13: p1 ← p1' ; p2 ← p2'
         14: end if
         15: Avaliar p1 e p2 e inseri-los em Ptmp
      16: end while
      17: P ← Ptmp ; descartar Ptmp
   18: end for
   19: return melhor individuo encontrado
   */

   // TODO: See if we can improve this on the CPU version (mapping memory?)
   m_predicted_Y = new cl_float[ m_num_points * m_params->m_population_size ];

   cl_float* fitness = new cl_float[ m_params->m_population_size ];
   cl_uint* pop_a = new cl_uint[ m_params->m_population_size * (m_params->m_maximum_genome_size + 1) ];
   cl_uint* pop_b = new cl_uint[ m_params->m_population_size * (m_params->m_maximum_genome_size + 1) ];

   cl_uint* cur_pop = pop_a;
   cl_uint* tmp_pop = pop_b;

   std::cout << "Evolving initial generation... ";
   InitializePopulation( cur_pop );
   EvaluatePopulation( cur_pop, fitness );
   std::cout << "done.\n";

   for( unsigned gen = 1; gen < m_params->m_number_of_generations; ++gen )
   {
      std::cout << "Evolving generation " << gen << "... ";

      Breed( cur_pop, tmp_pop );

      EvaluatePopulation( tmp_pop, fitness );
      std::cout << "done.\n";

      std::swap( cur_pop, tmp_pop );
   }

   delete[] pop_a;
   delete[] pop_b;
   delete[] fitness;
}

// -----------------------------------------------------------------------------
void GP::Breed( cl_uint* old_pop, cl_uint* new_pop )
{
   // Elitism
   for( unsigned i = 0; i < m_params->m_elitism_size; ++i )
   {
      // FIXME: (use the vector of best individuals)
      Clone( &old_pop[i * (m_params->m_maximum_genome_size + 1)],
             &new_pop[i * (m_params->m_maximum_genome_size + 1)] );

      // TODO: remove:
      PrintGenome( &new_pop[i * (m_params->m_maximum_genome_size + 1)] );
   }

   // Genetic operations
}

// -----------------------------------------------------------------------------
void GP::Clone( cl_uint* genome_orig, cl_uint* genome_dest ) const
{
   assert( genome_orig != NULL && genome_dest != NULL );
   assert( *genome_orig <= m_params->m_maximum_genome_size );

   // The size is the first element
   for( unsigned i = *genome_orig + 1; i-- ; )
      *genome_dest++ = *genome_orig++;
}

// -----------------------------------------------------------------------------
void GP::EvaluatePopulation( cl_uint* pop, cl_float* fitness )
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
         sizeof( cl_uint ) * ( m_params->m_population_size * ( m_params->m_maximum_genome_size + 1 ) ),
         pop, NULL, NULL);

   m_queue.enqueueNDRangeKernel( m_kernel, cl::NDRange(), cl::NDRange( m_num_global_wi ), 
                                 cl::NDRange( m_num_local_wi ) );

   m_queue.finish();

   // TODO: How about enqueueMapBuffer? (it can be faster)
   m_queue.enqueueReadBuffer( m_buf_predicted_Y, CL_TRUE, 0,
         m_num_points * m_params->m_population_size * sizeof(cl_float),
         m_predicted_Y, NULL, NULL );

   // --- Fitness calculation -----------------
   // TODO: Do this on the GPU!!
   for( unsigned i = 0; i < m_params->m_population_size; ++i )
   {
      fitness[i] = 0.0f;
      // TODO: check for nan/infinity
      // sum of the squared error
      for( unsigned j = 0; j < m_num_points; ++j )
         //fitness[i] += std::pow( m_predicted_Y[i * m_num_points + j] - m_Y[j], 2 );
         fitness[i] += std::abs( m_predicted_Y[i * m_num_points + j] - m_Y[j] );

      std::cout << "Fitness program " << i << ": " << fitness[i] << std::endl;
      // TODO: Pick the best and fill the elitism vector (if any)
   }
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

   m_queue = cl::CommandQueue( m_context, m_device, 0 );
}

// -----------------------------------------------------------------------------
void GP::CreateBuffers()
{
   //TODO: Optimize for CPU (USE_HOST_PTR?)

   // Buffer (memory on the device) of training points
   // TODO: I think m_X can be freed right after cl::Buffer returns. Check that!
   m_buf_X = cl::Buffer( m_context,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof( cl_float ) * m_num_points * m_x_dim,
                         m_X );

   // Buffer (memory on the device) of predicted values
   m_buf_predicted_Y = cl::Buffer( m_context,
                                   CL_MEM_WRITE_ONLY,
                                   m_num_points * m_params->m_population_size * sizeof(cl_float) );

  /* 
   Structure of a genome (individual)

     |                         |
+----+-----+----+--------------+-------------
|size|arity|type| index/value  |  ...
| 32 |  3  | 7  |     22       |
+----+-----+----+--------------+-------------
     |    first element        | second ...
  */
   // Buffer (memory on the device) of the population
   m_buf_pop = cl::Buffer( m_context,
                          CL_MEM_READ_ONLY,
                          sizeof( cl_uint ) * ( m_params->m_population_size * 
                                                ( m_params->m_maximum_genome_size + 1 ) ) );
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
            + "PUSH(" + m_primitives.DB[INDEX( m_primitives.m_primitives[i] )].code + ");break;";
      }
   interpreter += "\n";

   // program_src = header + kernel
   std::string program_src = 
      "#define MAX_GENOME_SIZE " + util::ToString( m_params->m_maximum_genome_size ) + "\n" +
      "#define NUM_POINTS " + util::ToString( m_num_points ) + "\n"
      "#define TOP       ( stack[stack_top] )\n"
      "#define POP       ( stack[stack_top--] )\n"
      "#define PUSH( i ) ( stack[++stack_top] = (i) )\n"
      "#define CREATE_STACK( type, size ) type stack[size]; int stack_top = -1;\n"
      "#define GENE genome[op]\n"
      + interpreter + m_kernel_src;

   // FIXME:
   std::cerr << std::endl;
   std::cerr << program_src;
   std::cerr << std::endl;

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

   // Set kernel arguments
   m_kernel.setArg( 0, m_buf_pop );
   m_kernel.setArg( 1, m_buf_X );
   m_kernel.setArg( 2, m_buf_predicted_Y );
   m_kernel.setArg( 3, sizeof(uint) * m_params->m_maximum_genome_size, NULL );
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
   // FIXME:
   LoadKernel( "kernels/gpu_ppcu.cl" );
   //LoadKernel( "kernels/cpu.cl" );
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
   for( cl_uint i = 0; i < m_params->m_population_size; ++i )
   {
      // TODO: how about using a normal distribution (with mean = max_gen_size/2)?
      cl_uint size = Random::Int( 1, m_params->m_maximum_genome_size );

      cl_uint idx = i * (m_params->m_maximum_genome_size + 1);
      // The first "gene" is the genome's size
      pop[idx++] = size;

      CreateLinearTree( pop + idx, size );

      // TODO: remove:
      PrintGenome( pop + idx - 1 );

   }
}

// -----------------------------------------------------------------------------
void GP::PrintGenome( const cl_uint* genome ) const
{
   int sum = *genome;

   for( unsigned i = *genome; i-- ; )
   {
      sum -= ARITY( *(++genome) );
     // std::cout << "[" << ARITY(*genome) << "," << INDEX(*genome) << "," 
       //         << (INDEX(*genome) == Primitives::GPT_EPHEMERAL ? AS_FLOAT(*genome) : AS_INT(*genome)) << "]";

      switch( INDEX(*genome) )
      {
         case Primitives::GPT_VAR:
            std::cout << "X" << AS_INT(*genome) << " ";
            break;
         case Primitives::GPT_EPHEMERAL:
            std::cout << AS_FLOAT(*genome) << " ";
            break;
         case Primitives::GPF_IDENTITY:
            break;
         default:
            std::cout << m_primitives.DB[INDEX(*genome)].symbol << " ";
      }
   }

   //std::cerr << "\nSize: " << size << " Sum: " << sum << std::endl;
   std::cout << " (CRC: " << sum << ")" << std::endl; 
   if( sum != 1 ) throw Error( "CRC != 1" );
}

// -----------------------------------------------------------------------------
void GP::CreateLinearTree( cl_uint* genome, unsigned left )
{
   assert( left >= 1 );
   assert( genome != 0 );

   unsigned open = 1;

   do {
      if( open == left || open > 1 )
         /* 
            [open == left] When the number of open arguments is equal the
            number of left genes, then the only valid choice is a terminal
            (RandomGene( 0, 0 )), otherwise we would end up with a genome
            greater than its size.

            [open > 1] When the number of open arguments is greater than one,
            then we can allow terminals to be chosen because they will not
            prematurely end the genome.
          */
         *genome = m_primitives.RandomGene( 0, left - open );
      else
         /* This means that 'open == 1' and 'left > 1', so we cannot choose
            a terminal here because we would end up with a shorter genome. */
         *genome = m_primitives.RandomGene( 1, left - open );

      /* Whenever we put a new operator/operand, the number of open arguments
         decreases. However, if the new operator requires more than one argument
         (arity >= 2) then we end up increasing the current number of open arguments.
       */
      open += ARITY( *genome++ ) - 1;
   } while( --left );
}

// -----------------------------------------------------------------------------
//void GP::LoadPoints( std::vector<std::vector<float> > & out_x, std::vector<float> & out_y )
void GP::LoadPoints( std::vector<std::vector<cl_float> > & out_x )
{
   using namespace util;

   // We will consider just the first file name given by the user
   std::ifstream points( m_params->m_data_points[0].c_str() );

   if( !points.is_open() ) {
      // Maybe a typo when passing the file name on the command-line
      throw Error( "[" + m_params->m_data_points[0] + "]: file not found." );
	}

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
      m_Y.push_back( v.back() ); v.pop_back();
      
      out_x.push_back( v );
   }

   if( out_x.empty() )
      throw Error( "[" + m_params->m_data_points[0] + "]: no data found." );

   m_num_points = out_x.size();
}
