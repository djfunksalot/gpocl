//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
                        __global float* E, __local uint* program )
{
   __local float PE[WGS];

   __local unsigned int program_size;
   CREATE_STACK( float, MAX_TREE_SIZE );

   uint i_id = get_local_id( 0 );
   uint g_id = get_group_id( 0 );
   uint wg_size = get_local_size( 0 );

   // Get the actual program's size
   if( i_id == 0 ) program_size = pop[(MAX_TREE_SIZE + 1) * g_id];

   barrier(CLK_LOCAL_MEM_FENCE);

   // FIXME: manage cases where program_size > work_group_size (each work-item
   // will need to handle more than one index.
   if( i_id < program_size ) program[i_id] = pop[(MAX_TREE_SIZE + 1) * g_id + i_id + 1];

   barrier(CLK_LOCAL_MEM_FENCE);

   PE[i_id] = 0.0f;

   // FIXME: handle cases where num_points is not divided by wg_size
   for( uint iter = 0; iter < NUM_POINTS/wg_size; ++iter )
   {
      // -------------------------------
      // Calls the interpreter (C macro)
      // -------------------------------
      for( int op = program_size; op-- ; )
         switch( INDEX( program[op] ) )
         {
            INTERPRETER_CORE
            default:
               PUSH( 0, X[iter * wg_size + NUM_POINTS * AS_INT( program[op] ) + i_id] );
         }

      // -------------------------------

      PE[i_id] += pown( POP - Y[ iter * wg_size + i_id ], 2 );
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
   for( uint d = 2; d<= WGS; d *= 2 )
   {
      barrier(CLK_LOCAL_MEM_FENCE);

      if( i_id % d == 0 ) PE[i_id] += PE[i_id + d/2];
   }

   // Store on the global memory (to be read by the host)
   if( i_id == 0 ) E[g_id] = PE[0];
}
