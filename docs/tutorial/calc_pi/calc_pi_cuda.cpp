///////////////////////////////////////////////////////////////////////////////
//
// file: calc_pi_cuda.cpp
// 
// Source file for calc_pi_cuda library example.
//
// Adapted from:
// https://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples-usingmpi/simplempi/cpi_c.html
///////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
///
/// file: calc_pi_cuda.cpp
///
//-----------------------------------------------------------------------------

const int block_size = 512;

// -- cuda kernel to calculate pi via simple integration  -- //
__global__ void calc_pi_kernel(int num_intervals,
                               double *pi)
{
    // local thread id
    int tid = threadIdx.x;
    // calc id
    int i   = blockIdx.x*blockDim.x + threadIdx.x;
    
    __shared__ double sum[block_size];
    
    double h   = 1.0 / (double) num_intervals;
    
    // calc sum contrib in parallel 
    double x = h * ((double)i - 0.5);
    double thread_sum = 4.0 / (1.0 + x*x);
    
    // save to shared memory, last block may pad with 0’s
    sum[tid] = (i < num_intervals) ? thread_sum : 0.0; 
    __syncthreads();
    
    // Build summation tree over elements
    for(int s=blockDim.x/2; s>0; s=s/2)
    {
        if(tid < s)
        {
            sum[tid] += sum[tid + s];
        }
        __syncthreads();
    }
    // Thread 0 adds the partial sum to the total sum
    if( tid == 0 )
    {
        // TODO: atomicAdd w/ doubles requires compute_60 or greater,
        //   we may need another path ...
        atomicAdd(pi, sum[tid]);
    }
}

// -- helper for calcing number of blocks to launch -- //
int iDivUp(int a, int b)
{ 
    return (a % b != 0) ? (a / b + 1) : (a / b); 
}

// -- calculate pi via simple integration  -- //
double calc_pi_cuda(int num_intervals)
{
    int num_threads = block_size;
    int num_blocks  = iDivUp(num_intervals, block_size);
    
    double  h_pi = 0.0;
    double *d_pi = NULL;
    
    cudaMalloc((void**)&d_pi, sizeof(double));
    
    calc_pi_kernel<<<num_blocks,num_threads>>>(num_intervals, d_pi);
    
    cudaMemcpy(&h_pi, d_pi,
               sizeof(double),
               cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize(); 
    
    cudaFree(d_pi);
    
    // final scaling
    return h_pi / (double) num_intervals;    
}



