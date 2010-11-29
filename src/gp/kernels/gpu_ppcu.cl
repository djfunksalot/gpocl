//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
                        __global float* E, __local uint* program )
{
   __local float PE[LOCAL_SIZE];

   __local unsigned int program_size;

   CREATE_STACK

   uint lo_id = get_local_id( 0 );
   uint gr_id = get_group_id( 0 );

   // Get the actual program's size
   if( lo_id == 0 ) program_size = pop[(MAX_TREE_SIZE + 1) * gr_id];

   barrier(CLK_LOCAL_MEM_FENCE);

#ifdef MAX_TREE_SIZE_IS_LESS_THAN_LOCAL_SIZE
   if( lo_id < program_size ) program[lo_id] = pop[(MAX_TREE_SIZE + 1) * gr_id + lo_id + 1];
#else   
   // Too few workers for the program_size, thus we need to do the work in rounds.
   uint rounds = ceil( program_size / (float) LOCAL_SIZE );
   for( uint i = 0; i < rounds; ++i )
   {
      if( i * rounds + lo_id < program_size )
         program[i * rounds + lo_id] = pop[(MAX_TREE_SIZE + 1) * gr_id + (i * rounds + lo_id) + 1];
   } 
#endif

   barrier(CLK_LOCAL_MEM_FENCE);

   PE[lo_id] = 0.0f;

#ifdef NUM_POINTS_IS_DIVISIBLE_BY_LOCAL_SIZE
   /* When we know that NUM_POINTS is divisible by LOCAL_SIZE then we can avoid a
      comparison in each iteration due to the guarantee of not having work-items
      accessing beyond the available amount of points. */
   for( uint iter = 0; iter < NUM_POINTS/LOCAL_SIZE; ++iter )
   {
#else
   for( uint iter = 0; iter < ceil( NUM_POINTS / (float) LOCAL_SIZE); ++iter )
   {
      if( iter * LOCAL_SIZE + lo_id < NUM_POINTS )
      {
#endif
         // -------------------------------
         // Calls the interpreter (C macro)
         // -------------------------------
         for( int op = program_size; op-- ; )
            switch( INDEX( program[op] ) )
            {
               INTERPRETER_CORE
               default:
                  PUSH( 0, X[iter * LOCAL_SIZE + NUM_POINTS * AS_INT( program[op] ) + lo_id] );
            }

         // -------------------------------

         PE[lo_id] += pown( POP - Y[ iter * LOCAL_SIZE + lo_id ], 2 );

         // Avoid further calculations if the current one has overflown the float
         // (i.e., it is inf or NaN).
         if( isinf( PE[lo_id] ) || isnan( PE[lo_id] ) ) break;
#ifndef NUM_POINTS_IS_DIVISIBLE_BY_LOCAL_SIZE
      }
#endif
   }

   /*
      Parallel way to perform reduction:

      | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     LOCAL_SIZE = 8

        |___|   |___|   |___|   |___|       d = 2
        
      | 0 |   | 2 |   | 4 |   | 6 |      

        |_______|       |_______|           d = 4

      | 0 |           | 4 |               

        |_______________|                   d = 8

      | 0 |

        |-----> total sum is stored on the first work-item
    */
         
   for( uint d = 2; d <= LOCAL_SIZE_NEXT_POWER_OF_2; d *= 2 )
   {
      barrier(CLK_LOCAL_MEM_FENCE);

      if( lo_id % d == 0 ) 
      {
#ifdef LOCAL_SIZE_IS_NOT_POWER_OF_2
         /* If LOCAL_SIZE is not power of two (and we are iterating up to the next power
            of two value) then we need to ensure that we will not read past LOCAL_SIZE. */
         if( lo_id + d/2 < LOCAL_SIZE )
#endif
         PE[lo_id] += PE[lo_id + d/2];
      }
   }

   // Store on the global memory (to be read by the host)
   if( lo_id == 0 ) 
      // Check for infinity/NaN
      E[gr_id] = ( isinf( PE[0] ) || isnan( PE[0] ) ) ? MAX_FLOAT : PE[0];
}
