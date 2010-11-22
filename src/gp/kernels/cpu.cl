#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void evaluate( __global const uint* pop, __global const float* X, __global float* pred_Y, __local uint* genome )
{
   // TODO: remove __local uint* genome

   __local unsigned int genome_size;
   CREATE_STACK( float, MAX_GENOME_SIZE )

   //uint i_id = get_local_id( 0 );
   uint g_id = get_group_id( 0 );
//   uint wg_size = get_local_size( 0 );

   // Get the actual genome's size
   genome_size = pop[(MAX_GENOME_SIZE + 1) * g_id];

   // TODO: use directly pop instead of genome
   for( uint i = 0; i < genome_size; ++i )
      genome[i] = pop[(MAX_GENOME_SIZE + 1) * g_id + i + 1];

   //printf( "[Genome size: %d]", genome_size );
   for( uint iter = 0; iter < NUM_POINTS; ++iter )
   {
      // -------------------------------
      // Calls the interpreter (C macro)
      // -------------------------------
      for( int op = genome_size; op-- ; )
{
   //printf( "[%d][%d](%d,", op, INDEX( genome[op] ), stack_top );

         switch( INDEX( genome[op] ) )
         {
            INTERPRETER_CORE
            default:
               PUSH( 0, X[iter * X_DIM + AS_INT( genome[op] )] );
         }
   //printf( "%d,%f)", stack_top, TOP );
}
//printf( ": %f\n", TOP );
      // -------------------------------

      pred_Y[NUM_POINTS * g_id + iter] = POP;
   }
}
