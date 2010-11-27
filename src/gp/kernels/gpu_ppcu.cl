//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
                        __global float* E, __local uint* program )
{
   __local float PE[WGS];

   __local unsigned int program_size;
   CREATE_STACK( float, MAX_TREE_SIZE );

   uint lo_id = get_local_id( 0 );
   uint gr_id = get_group_id( 0 );

   // Get the actual program's size
   if( lo_id == 0 ) program_size = pop[(MAX_TREE_SIZE + 1) * gr_id];

   barrier(CLK_LOCAL_MEM_FENCE);

   // FIXME: manage cases where program_size > work_group_size (each work-item
   // will need to handle more than one index.
   ///if( lo_id < program_size ) program[lo_id] = pop[(MAX_TREE_SIZE + 1) * gr_id + lo_id + 1];
   
   uint rounds = ceil( program_size / (float) WGS );
   for( uint i = 0; i < rounds; ++i )
   {
      if( i * rounds + lo_id < program_size )
         program[i * rounds + lo_id] = pop[(MAX_TREE_SIZE + 1) * gr_id + (i * rounds + lo_id) + 1];
   } 

   barrier(CLK_LOCAL_MEM_FENCE);

   PE[lo_id] = 0.0f;

   // FIXME: handle cases where num_points is not divided by WGS
   for( uint iter = 0; iter < NUM_POINTS/WGS; ++iter )
   {
      // -------------------------------
      // Calls the interpreter (C macro)
      // -------------------------------
      for( int op = program_size; op-- ; )
         switch( INDEX( program[op] ) )
         {
            INTERPRETER_CORE
            default:
               PUSH( 0, X[iter * WGS + NUM_POINTS * AS_INT( program[op] ) + lo_id] );
         }

      // -------------------------------

      PE[lo_id] += pown( POP - Y[ iter * WGS + lo_id ], 2 );

      // Avoid further calculations if the current one has overflown the float
      // (i.e., it is inf or NaN).
      if( ! isnormal( PE[lo_id] ) ) break;
   }

   /*
      Parallel way to perform reduction:

      | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     WGS = 8

        |___|   |___|   |___|   |___|       d = 2
        
      | 0 |   | 2 |   | 4 |   | 6 |      

        |_______|       |_______|           d = 4

      | 0 |           | 4 |               

        |_______________|                   d = 8

      | 0 |

        |-----> total sum is stored on the first work-item
    */
         
   // FIXME: ensure WGS is power of 2 (p.e., when NUM_POINTS < Max WGS, then
   // WGS = NUM_POINTS )
   for( uint d = 2; d <= WGS; d *= 2 )
   {
      barrier(CLK_LOCAL_MEM_FENCE);

      if( lo_id % d == 0 ) PE[lo_id] += PE[lo_id + d/2];
   }

   // Store on the global memory (to be read by the host)
   if( lo_id == 0 ) 
      // Check for infinity/NaN
      E[gr_id] = isnormal( PE[0] ) ? PE[0] : MAX_FLOAT;
}
