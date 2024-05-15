#include <cstdio>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)
#define INDEX3D(a, b, c, db, dc) (((a) * (db) * (dc) + (b) * (dc) + (c)))

__global__ void leg_fwd_kernel(const float *x, float *leg, int batch_size, int in_feats, int degree, int numThreads){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numThreads) {
        int irow = idx / in_feats;
        int icol = idx % in_feats;
        
        float x_val = x[idx];
        leg[INDEX3D(1, irow, icol, batch_size, in_feats)] = x_val;
        float leg_val_z = x_val; // index = i - 1
        float leg_val_zz = 1;    // index = i - 2

        for(int d = 2; d < degree + 1; d++){
            float df = static_cast<float>(d);
            float denom_inv = 1.f / (df + 1);
            float new_leg_val = ((2 * df + 1) * x_val * leg_val_z - df * x_val * leg_val_zz) * denom_inv;
            leg[INDEX3D(d, irow, icol, batch_size, in_feats)] = new_leg_val;


            // finally
            leg_val_zz = leg_val_z;
            leg_val_z = new_leg_val;
        }
    }
}


__global__ void leg_bwd_kernel(const float* gout, const float *x, const float *leg, float* grad_x,
                                 int batch_size, int in_feats, int degree, int numThreads){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numThreads) {

        int irow = idx / in_feats;
        int icol = idx % in_feats;

        float b0 = 0, b1 = 1;
        float b_z = b1, b_zz = b0; // b(i-1) and b(i-2)

        float x_val = x[idx];

        // grad wrt d=0 is equal to zero
        // here is grad wrt d=1
        float grad_x_val = gout[INDEX3D(1, irow, icol, batch_size, in_feats)];

        for(int d = 2; d < degree + 1; d++){

            float df = static_cast<float>(d);
            float denom_inv = 1.f / (df + 1);

            // 2a(i-1)
            float b = ((2 * df + 1) * (leg[INDEX3D(d - 1, irow, icol, batch_size, in_feats)] + x_val * b_z) - df * b_zz) * denom_inv;

            grad_x_val += gout[INDEX3D(d, irow, icol, batch_size, in_feats)] * b;

            // finally
            b_zz = b_z;
            b_z = b;
        }

        grad_x[idx] = grad_x_val;


    }
}



void leg_launcher(const float *x, float *leg, int batch_size, int in_feats, int degree){
    int numThreads = batch_size * in_feats;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    leg_fwd_kernel<<<blockSize, threadSize>>>(x, leg, batch_size, in_feats, degree, numThreads);
}

void leg_bwd_launcher(const float *gout, const float *x, const float *leg, float *grad_x,
                        int batch_size, int in_feats, int degree){
    int numThreads = batch_size * in_feats;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    leg_bwd_kernel<<<blockSize, threadSize>>>(gout, x, leg, grad_x, batch_size, in_feats, degree, numThreads);
}


