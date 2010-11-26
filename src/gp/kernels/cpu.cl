//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
                        __global float* E )
{
   CREATE_STACK( float, MAX_TREE_SIZE )

   uint g_id = get_group_id( 0 );

   // Get the actual program's size
   uint program_size = pop[(MAX_TREE_SIZE + 1) * g_id];
   // Make program points to the actual program being evaluated
   __global const uint* program = &pop[(MAX_TREE_SIZE + 1) * g_id + 1];

   float error = 0.0f;

   //printf( "[Genome size: %d]", program_size );
   for( uint iter = 0; iter < NUM_POINTS; ++iter )
   {
      // -------------------------------
      // Calls the interpreter (C macro)
      // -------------------------------
      for( int op = program_size; op-- ; )
      {
         switch( INDEX( program[op] ) )
         {
            INTERPRETER_CORE
            default:
               PUSH( 0, X[iter * X_DIM + AS_INT( program[op] )] );
         }
      }
      error += pown( POP - Y[ iter ], 2 );
   }

   // Check for inifity/NaN and then normalize the error (dividing by NUM_POINTS)
   E[ g_id ] = ( isinf( error ) || isnan( error ) ) ? MAX_FLOAT : error / (float) NUM_POINTS;
}
