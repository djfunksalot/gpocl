//__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
 //                       __global float* E )
__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
                        __global float* E, __local uint* program )
{
   __local uint program_size;
   CREATE_STACK

   uint gl_id = get_global_id( 0 );
   uint lo_id = get_local_id( 0 );

   //uint program_size;
  // __global const uint* program;
   float error;

   for( unsigned p = 0; p < POP_SIZE; ++p )
   {
      // TODO: use local memory
      // Get the actual program's size
      if( lo_id == 0 ) program_size = pop[(MAX_TREE_SIZE + 1) * p];

      barrier(CLK_LOCAL_MEM_FENCE);

#ifdef MAX_TREE_SIZE_IS_LESS_THAN_WGS
      if( lo_id < program_size ) program[lo_id] = pop[(MAX_TREE_SIZE + 1) * p + lo_id + 1];
#else   
      // Too few workers for the program_size, thus we need to do the work in rounds.
      uint rounds = ceil( program_size / (float) WGS );
      for( uint i = 0; i < rounds; ++i )
      {
         if( i * rounds + lo_id < program_size )
            program[i * rounds + lo_id] = pop[(MAX_TREE_SIZE + 1) * p + (i * rounds + lo_id) + 1];
      } 
#endif

      barrier(CLK_LOCAL_MEM_FENCE);


      if( gl_id < NUM_POINTS )
      { 

         /*
         // Get the actual program's size
         program_size = pop[(MAX_TREE_SIZE + 1) * p];
         // Make program points to the actual program being evaluated
         program =     &pop[(MAX_TREE_SIZE + 1) * p + 1];
          */

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

         // Fetch the errors from all the others work-items and store the sum in E[p]
         // FIXME (just to test, storing error only for point 0)
         if( gl_id == 0 ) E[p] = isnormal( error ) ? error : MAX_FLOAT;
      }
   }
}
