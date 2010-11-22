__kernel void evaluate( __global const uint* pop, __global const float* X, __global float* pred_Y, __local uint* genome )
{
   __local unsigned int genome_size;
   CREATE_STACK( float, MAX_GENOME_SIZE );

   uint i_id = get_local_id( 0 );
   uint g_id = get_group_id( 0 );
   uint wg_size = get_local_size( 0 );

   // Get the actual genome's size
   if( i_id == 0 ) genome_size = pop[(MAX_GENOME_SIZE + 1) * g_id];

   barrier(CLK_LOCAL_MEM_FENCE);

   // FIXME: manage cases where genome_size > work_group_size (each work-item
   // will need to handle more than one index.
   // TODO: Check whether or not *all* work-items for this group are complete
   // when the code reaches the barrier bellow.
   if( i_id < genome_size ) genome[i_id] = pop[(MAX_GENOME_SIZE + 1) * g_id + i_id + 1];

   barrier(CLK_LOCAL_MEM_FENCE);


   // FIXME: handle cases where num_points is not divided by wg_size
   // TODO: Check which version below is correct:
   //for( iter = 0; iter < NUM_POINTS/wg_size - 1; ++iter )
   for( uint iter = 0; iter < NUM_POINTS/wg_size; ++iter )
   {
      // -------------------------------
      // Calls the interpreter (C macro)
      // -------------------------------
      for( int op = genome_size; op-- ; )
         switch( INDEX( genome[op] ) )
         {
            INTERPRETER_CORE
            default:
               PUSH( 0, X[iter * wg_size + NUM_POINTS * AS_INT( genome[op] ) + i_id] );
         }

      // -------------------------------

      pred_Y[NUM_POINTS * g_id + iter * wg_size + i_id] = POP;
   }

   // TODO: Prefix sum, i.e., calculate the fitness!
}
