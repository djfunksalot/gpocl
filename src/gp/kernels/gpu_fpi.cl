__kernel void evaluate( __global const uint* pop, __global const float* X, 
#ifdef Y_DOES_NOT_FIT_IN_CONSTANT_BUFFER
      __global const 
#else
      __constant 
#endif
      float* Y, __global float* E, __local uint* program, uint program_id )
{
   __local float PE[LOCAL_SIZE];
   __local uint program_size;

   CREATE_STACK

   uint gl_id = get_global_id( 0 );
   uint gr_id = get_group_id( 0 );
   uint lo_id = get_local_id( 0 );

   if( lo_id == 0 ) program_size = pop[program_id];

   barrier(CLK_LOCAL_MEM_FENCE);

#ifndef PROGRAM_TREE_DOES_NOT_FIT_IN_LOCAL_SIZE
   if( lo_id < program_size ) program[lo_id] = pop[program_id + lo_id + 1];
#else   
   // Too few workers for the program_size, thus we need to do the work iteratively
   for( uint i = 0; i < ceil( program_size / (float) LOCAL_SIZE ); ++i )
   {
      uint index = i * LOCAL_SIZE + lo_id;

      if( index < program_size ) program[index] =   pop[program_id + index + 1];
   } 
#endif

   barrier(CLK_LOCAL_MEM_FENCE);

   PE[lo_id] = 0.0f;

#ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   if( gl_id < NUM_POINTS )
   { 
#endif
      for( int op = program_size; op-- ; )
         switch( INDEX( program[op] ) )
         {
            INTERPRETER_CORE
            default:
               PUSH_0( X[NUM_POINTS * AS_INT( program[op] ) + gl_id] );
         }

      // -------------------------------

      PE[lo_id] = ERROR_METRIC( POP, Y[gl_id] );
#ifdef NUM_POINTS_IS_NOT_DIVISIBLE_BY_LOCAL_SIZE
   }
#endif
   // Parallel reduction

   for( uint s = LOCAL_SIZE_ROUNDED_UP_TO_POWER_OF_2 / 2; s > 0; s >>= 1 ) 
   {
      barrier(CLK_LOCAL_MEM_FENCE);

#ifndef LOCAL_SIZE_IS_NOT_POWER_OF_2
      if( lo_id < s )
#else
         /* LOCAL_SIZE is not power of 2, so we need to perform an additional
          * check to ensure that no access beyond PE's range will occur. */ 
         if( (lo_id < s) && (lo_id + s < LOCAL_SIZE) )
#endif 
            PE[lo_id] += PE[lo_id + s];
   }

   /* We will store this group reduction into the global memory.
      On the host we will sum the error accumulated in each compute unit. It
      would be nice to do this 'host reduction' on the GPU, but for that we
      need "atomic add" for float types, which for the time there isn't a clean
      way to use.
      */
   if( lo_id == 0 ) 
      E[gr_id] = ( isinf( PE[0] ) || isnan( PE[0] ) ) ? MAX_FLOAT : PE[0];
}
