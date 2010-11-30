__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
                        __global float* E )
{
   CREATE_STACK

   uint gl_id = get_global_id( 0 );

   uint program_size;
   __global const uint* program;

   float error = 0.0f;

   // This is needed because global dimension can be greater than POP_SIZE (on
   // cases where POP_SIZE needs to be evenly divided by local work size)
   if( gl_id < POP_SIZE )
   {
      // Get the actual program's size
      program_size = pop[(MAX_TREE_SIZE + 1) * gl_id];
      // Make program points to the actual program being evaluated
      program =     &pop[(MAX_TREE_SIZE + 1) * gl_id + 1];

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
                  // Coalesced access pattern
                  PUSH( 0, X[iter + NUM_POINTS * AS_INT( program[op] )] );
            }

         // -------------------------------

         error += ERROR_METRIC( POP, Y[iter] );

         // Avoid further calculations if the current one has overflown the
         // float (i.e., it is inf or NaN).
         if( isinf( error ) || isnan( error ) ) { error = MAX_FLOAT; break; }

      }

      // Store on the global memory (to be read by the host)
      E[gl_id] = error;
   }
}
