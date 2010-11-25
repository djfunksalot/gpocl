//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
                        __global float* E, __local uint* program )
{
   // TODO: remove __local uint* program

   __local unsigned int program_size;
   CREATE_STACK( float, MAX_TREE_SIZE )

   uint g_id = get_group_id( 0 );

   float partial_error = 0.0f;

   // Get the actual program's size
   program_size = pop[(MAX_TREE_SIZE + 1) * g_id];

   // TODO: use directly pop instead of program
   for( uint i = 0; i < program_size; ++i )
      program[i] = pop[(MAX_TREE_SIZE + 1) * g_id + i + 1];

   //printf( "[Genome size: %d]", program_size );
   for( uint iter = 0; iter < NUM_POINTS; ++iter )
   {
      // -------------------------------
      // Calls the interpreter (C macro)
      // -------------------------------
      for( int op = program_size; op-- ; )
{
   //printf( "[%d][%d](%d,", op, INDEX( program[op] ), stack_top );

         switch( INDEX( program[op] ) )
         {
            INTERPRETER_CORE
            default:
               PUSH( 0, X[iter * X_DIM + AS_INT( program[op] )] );
         }
   //printf( "%d,%f)", stack_top, TOP );
}
//printf( ": %f\n", TOP );
      // -------------------------------

      partial_error += pown( POP - Y[ iter + g_id ], 2 );
   }

   E[ g_id ] = partial_error;
}
