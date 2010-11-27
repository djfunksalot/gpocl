__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
                        __global float* E )
{
   CREATE_STACK( float, MAX_TREE_SIZE );

   uint gl_id = get_global_id( 0 );
   uint program_size;
   __global const uint* program;
   float error;

   if( gl_id < NUM_POINTS )
   { 
      for( unsigned i = 0; i < POP_SIZE; ++i )
      {
         // TODO: use local memory
         // Get the actual program's size
         program_size = pop[(MAX_TREE_SIZE + 1) * i];
         // Make program points to the actual program being evaluated
         program =     &pop[(MAX_TREE_SIZE + 1) * i + 1];

         error = 0.0f;
         for( int op = program_size; op-- ; )
            switch( INDEX( program[op] ) )
            {
               INTERPRETER_CORE
               default:
                  PUSH( 0, X[NUM_POINTS * AS_INT( program[op] ) + gl_id] );
            }

         // -------------------------------

         error = pown( POP - Y[ gl_id ], 2 );

         // Fetch the errors from all the others work-items and store the sum in E[i]
         // FIXME (just to test, storing error only for point 0)
         if( gl_id == 0 ) E[i] = isnormal( error ) ? error : MAX_FLOAT;
      }
   }
}
