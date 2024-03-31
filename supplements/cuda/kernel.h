#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mat.h"

// PLEASE ENSURE THAT THE FOLLOWING DEFINITIONS MATCH THE DEFINITIONS IN MAIN.M
constexpr size_t NUM_DETECTOR_X = 1380;
constexpr size_t NUM_DETECTOR_Y = 256;
constexpr size_t NUM_TIME = 2048;

constexpr float VS = 1500.0;
constexpr float FS = 40e6;
constexpr float ANGLE_COS_LIMIT = 0.5;

constexpr float DETECTOR_INTERVAL_X = 0.1e-3;
constexpr float DETECTOR_INTERVAL_Y = 0.5e-3;

constexpr float RES_X = 0.2e-3;
constexpr float RES_Y = 0.2e-3;
constexpr float RES_Z = 0.2e-3;

constexpr float X_START = -10e-3;
constexpr float X_END = 110e-3;
constexpr float Y_START = -10e-3;
constexpr float Y_END = 110e-3;
constexpr float Z_START = 10e-3;
constexpr float Z_END = 50e-3;
// -----------------------------------------------------------------------------

constexpr size_t NUM_RECON_X = (X_END - X_START) / RES_X;
constexpr size_t NUM_RECON_Y = (Y_END - Y_START) / RES_Y;
constexpr size_t NUM_RECON_Z = (Z_END - Z_START) / RES_Z;
constexpr size_t NUM_RECON = NUM_RECON_X * NUM_RECON_Y * NUM_RECON_Z;
constexpr size_t NUM_DETECTOR = NUM_DETECTOR_X * NUM_DETECTOR_Y;
constexpr size_t NUM_SIGNAL = NUM_DETECTOR * NUM_TIME;

constexpr size_t NUM_THREAD = 256;
constexpr size_t NUM_BLOCK = (NUM_RECON + NUM_THREAD - 1) / NUM_THREAD;

__global__ void ReconKernel(const float* signal_backproj,
                            const float* detector_location, const size_t n,
                            float* signal_recon);

void ReconWithCuda(const float* signal_backproj, const float* detector_location,
                   float* signal_recon);

void MatRead(const char* file, const char* variable, const size_t size,
             float* out);

void MatWrite(const char* file, const char* variable, const size_t size,
              float* in);
