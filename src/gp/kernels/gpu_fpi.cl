__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
      __global float* E, __local uint* program )
{
   __local float PE[LOCAL_SIZE];
   __local uint program_size;

   CREATE_STACK

   uint gl_id = get_global_id( 0 );
   uint gr_id = get_group_id( 0 );
   uint lo_id = get_local_id( 0 );

   float error;

   for( unsigned p = 0; p < POP_SIZE; ++p )
   {
      if( lo_id == 0 ) program_size = pop[(MAX_TREE_SIZE + 1) * p];

      barrier(CLK_LOCAL_MEM_FENCE);

#ifdef PROGRAM_TREE_FITS_IN_LOCAL_SIZE
      if( lo_id < program_size ) program[lo_id] = pop[(MAX_TREE_SIZE + 1) * p + lo_id + 1];
#else   
      // Too few workers for the program_size, thus we need to do the work iteratively
      for( uint i = 0; i < ceil( program_size / (float) LOCAL_SIZE ); ++i )
      {
         uint index = i * LOCAL_SIZE + lo_id;

         if( index < program_size )
            program[index] =   pop[(MAX_TREE_SIZE + 1) * p + index + 1];
      } 
#endif

      barrier(CLK_LOCAL_MEM_FENCE);

      PE[lo_id] = 0.0f;

      if( gl_id < NUM_POINTS )
      { 
         for( int op = program_size; op-- ; )
            switch( INDEX( program[op] ) )
            {
               INTERPRETER_CORE
               default:
                  PUSH( 0, X[NUM_POINTS * AS_INT( program[op] ) + gl_id] );
            }

         // -------------------------------

         PE[lo_id] = ERROR_METRIC( POP, Y[gl_id] );
      }

      // Parallel reduction

      for( uint s = LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 / 2; s > 0; s >>= 1 ) 
      {
         barrier(CLK_LOCAL_MEM_FENCE);

#ifdef LOCAL_SIZE_IS_POWER_OF_2
         if( lo_id < s )
#else
            /* LOCAL_SIZE is not power of 2, so we need to perform an additional
             * check to ensure that no access beyond PE's range will occur. */ 
            if( (lo_id < s) && (lo_id + s < LOCAL_SIZE) )
#endif 
               PE[lo_id] += PE[lo_id + s];
      }

      // We will store this group reduction into the global memory.
      // FIXME: We need to expand E and perform reduction in the global memory
      // (i.e. reduce the partial sums of each CU)
      if( lo_id == 0 ) 
         E[gr_id] = ( isinf( PE[0] ) || isnan( PE[0] ) ) ? MAX_FLOAT : PE[0];
   }
}
