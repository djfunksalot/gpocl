//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void evaluate( __global const uint* pop, __global const float* X, __global const float* Y,
                        __global float* E, __local uint* program )
{
   __local unsigned int program_size;
   CREATE_STACK( float, MAX_TREE_SIZE );

   uint i_id = get_local_id( 0 );
   uint g_id = get_group_id( 0 );
   uint wg_size = get_local_size( 0 );

   float partial_error = 0.0f;

   // Get the actual program's size
   if( i_id == 0 ) program_size = pop[(MAX_TREE_SIZE + 1) * g_id];

   barrier(CLK_LOCAL_MEM_FENCE);

   // FIXME: manage cases where program_size > work_group_size (each work-item
   // will need to handle more than one index.
   // TODO: Check whether or not *all* work-items for this group are complete
   // when the code reaches the barrier bellow.
   if( i_id < program_size ) program[i_id] = pop[(MAX_TREE_SIZE + 1) * g_id + i_id + 1];

   barrier(CLK_LOCAL_MEM_FENCE);

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

//      if( g_id == 5 && i_id == 0 ) 
 //        printf( "[%f]", Y[ iter * wg_size + i_id ]);
         //printf( "[%f - %f: %f]", TOP, Y[ iter * wg_size + i_id ], partial_error );
      partial_error += pown( POP - Y[ iter * wg_size + i_id ], 2 );
    //  partial_error += ( TOP - Y[ iter * wg_size + i_id ] ) *
     //                  ( TOP - Y[ iter * wg_size + i_id ] ); POP;
    //  partial_error += fabs( POP - Y[ iter * wg_size + i_id ] );
   }

   E[ g_id * wg_size + i_id ] = partial_error;

//   if( g_id == 5 )
      //printf( "[%f]", partial_error );
   // TODO: Prefix sum, i.e., calculate the fitness!
}
