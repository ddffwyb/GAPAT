#include "kernel.h"

int main() {
    float* signal_backproj = (float*)malloc(sizeof(float) * NUM_SIGNAL);
    float* detector_location = (float*)malloc(sizeof(float) * NUM_DETECTOR * 3);
    float* signal_recon = (float*)malloc(sizeof(float) * NUM_RECON);
    MatRead("data/signal_backproj.mat", "signal_backproj", NUM_SIGNAL,
            signal_backproj);
    MatRead("data/detector_location.mat", "detector_location", NUM_DETECTOR * 3,
            detector_location);
    memset(signal_recon, 0, sizeof(float) * NUM_RECON);
    ReconWithCuda(signal_backproj, detector_location, signal_recon);
    MatWrite("data/signal_recon.mat", "signal_recon", NUM_RECON, signal_recon);
    free(signal_backproj);
    free(detector_location);
    free(signal_recon);
    return 0;
}

__global__ void ReconKernel(const float* signal_backproj,
                            const float* detector_location, const size_t n,
                            float* signal_recon) {
    size_t k = blockDim.x * blockIdx.x + threadIdx.x;
    size_t x_idx = k / (NUM_RECON_Y * NUM_RECON_Z);
    size_t y_idx = (k % (NUM_RECON_Y * NUM_RECON_Z)) / NUM_RECON_Z;
    size_t z_idx = (k % (NUM_RECON_Y * NUM_RECON_Z)) % NUM_RECON_Z;
    if (k < NUM_RECON) {
        float dx = X_START + x_idx * RES_X - detector_location[n];
        float dy =
            Y_START + y_idx * RES_Y - detector_location[n + NUM_DETECTOR];
        float dz =
            Z_START + z_idx * RES_Z - detector_location[n + NUM_DETECTOR * 2];
        float d = sqrt(dx * dx + dy * dy + dz * dz);
        float angle_cos = dz / d;
        size_t idx = d / VS * FS;
        if ((angle_cos > ANGLE_COS_LIMIT) && (idx < NUM_TIME)) {
            signal_recon[k] +=
                signal_backproj[n + NUM_DETECTOR * idx] * angle_cos / d / d;
        }
    }
}

void ReconWithCuda(const float* signal_backproj, const float* detector_location,
                   float* signal_recon) {
    float* d_signal_backproj;
    float* d_detector_location;
    float* d_signal_recon;
    cudaMalloc(&d_signal_backproj, sizeof(float) * NUM_SIGNAL);
    cudaMalloc(&d_detector_location, sizeof(float) * NUM_DETECTOR * 3);
    cudaMalloc(&d_signal_recon, sizeof(float) * NUM_RECON);
    cudaMemcpy(d_signal_backproj, signal_backproj, sizeof(float) * NUM_SIGNAL,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_detector_location, detector_location,
               sizeof(float) * NUM_DETECTOR * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_signal_recon, signal_recon, sizeof(float) * NUM_RECON,
               cudaMemcpyHostToDevice);
    for (size_t n = 0; n < NUM_DETECTOR; ++n) {
        ReconKernel<<<NUM_BLOCK, NUM_THREAD>>>(
            d_signal_backproj, d_detector_location, n, d_signal_recon);
    }
    cudaMemcpy(signal_recon, d_signal_recon, sizeof(float) * NUM_RECON,
               cudaMemcpyDeviceToHost);
    cudaFree(d_signal_backproj);
    cudaFree(d_detector_location);
    cudaFree(d_signal_recon);
}

void MatRead(const char* file, const char* variable, const size_t size,
             float* out) {
    MATFile* pMatFile;
    mxArray* pMxArray;
    pMatFile = matOpen(file, "r");
    if (pMatFile == NULL) {
        printf("Error opening file %s\n", file);
        return;
    }
    pMxArray = matGetVariable(pMatFile, variable);
    if (pMxArray == NULL) {
        printf("Error opening variable %s\n", variable);
        return;
    }
    memcpy((void*)out, (void*)mxGetPr(pMxArray), sizeof(float) * size);
    mxDestroyArray(pMxArray);
    matClose(pMatFile);
}

void MatWrite(const char* file, const char* variable, const size_t size,
              float* in) {
    MATFile* pMatFile;
    mxArray* pMxArray;
    pMatFile = matOpen(file, "w");
    if (pMatFile == NULL) {
        printf("Error opening file %s\n", file);
        return;
    }
    const mwSize dims[2] = {1, size};
    pMxArray = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
    if (pMxArray == NULL) {
        printf("Error opening variable %s\n", variable);
        return;
    }
    memcpy((void*)mxGetPr(pMxArray), (void*)in, sizeof(float) * size);
    matPutVariableAsGlobal(pMatFile, variable, pMxArray);
    mxDestroyArray(pMxArray);
    matClose(pMatFile);
}
