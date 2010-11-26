__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
                        __global float* E )
{
   CREATE_STACK( float, MAX_TREE_SIZE );

   uint i_gid = get_global_id( 0 );

   uint program_size;
   __global const uint* program;

   float error = 0.0f;

   // This is needed because global dimension can be greater than POP_SIZE (on
   // cases where POP_SIZE needs to be evenly divided by local work size)
   if( i_gid < POP_SIZE )
   {
      // Get the actual program's size
      program_size = pop[(MAX_TREE_SIZE + 1) * i_gid];
      // Make program points to the actual program being evaluated
      program =     &pop[(MAX_TREE_SIZE + 1) * i_gid + 1];

      for( uint iter = 0; iter < NUM_POINTS; ++iter )
      {
         // -------------------------------
         // Calls the interpreter (C macro)
         // -------------------------------
         for( int op = program_size; op-- ; )
            switch( INDEX( program[op] ) )
            {
               INTERPRETER_CORE
               default:
                  PUSH( 0, X[iter + NUM_POINTS * AS_INT( program[op] )] );
            }

         // -------------------------------

         error += pown( POP - Y[ iter ], 2 );
      }

      // Check for inifity/NaN and then normalize the error (dividing by NUM_POINTS)
      E[ i_gid ] = ( isinf( error ) || isnan( error ) ) ? MAX_FLOAT : error / (float) NUM_POINTS;
   }
}
