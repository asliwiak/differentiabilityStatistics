//written by Qiqi Wang
// density/bakers_density.cpp
#include<cstdio>
#include<cassert>
#include<algorithm>

#include"histogram.h"
#include"lorenz.h"

struct ReturnXZ {
    __device__ __forceinline__
    static ValueWeight eval(double xyz[3]) {
        return ValueWeight{xyz[0], xyz[2], 1.0};
    }
};

int main() {
    uint32_t iDevice;
    assert(fread(&iDevice, sizeof(uint32_t), 1, stdin) == 1);
    cudaSetDevice(iDevice);
    fprintf(stderr, "Set To Device %u\n", iDevice);

    uint32_t randSeed;
    assert(fread(&randSeed, sizeof(uint32_t), 1, stdin) == 1);
    srand(randSeed);

    const int nx = 3840, nz = 2160;
    // Counter<3> counter(nx, nz, -20.0, 0., 40./nx, 50./nz); // 28
    Counter<3> counter(nx, nz, -40.0, 0., 80./nx, 135./nz); // 70

    float parameters[5];
    assert(fread(parameters, sizeof(float), 5, stdin) == 5);
    Lorenz lorenz{parameters[0], parameters[1], parameters[2], parameters[3]};
    ReturnXZ obj;
    fprintf(stderr, "Ready with parameters (%f %f %f) with dt %f for %f steps\n",
            parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]);

    const uint64_t totalPoints = 500000000;
    counter.init(lorenz, totalPoints);
    counter.run(lorenz, obj, 1000, false);
    float * existingCounts = (float*)malloc(sizeof(float) * nx * nz);
    for (uint32_t i = 0; i < static_cast<int>(parameters[4]); ++i) {
        counter.run(lorenz, obj, 100, true);
        uint64_t totalCounts = 0;
        for (uint32_t j = 0; j < nx * nz; ++j)
            totalCounts += uint64_t(round(counter.counts[j]));
        fprintf(stderr, "Step %u: %lu/%lu\n", i, totalCounts, totalPoints);

        char fname[256];
        sprintf(fname, "data/device_%d_step_%d.bin", iDevice, i);
        FILE * f;
        if (f = fopen(fname, "rb")) {
            fread(existingCounts, sizeof(float), nx*nz, f);
            fclose(f);
            for (uint32_t j = 0; j < nx * nz; ++j) {
                counter.counts[j] += existingCounts[j];
            }
        }
        f = fopen(fname, "wb");
        fwrite(counter.counts, sizeof(float), nx * nz, f);
        fclose(f);
        memset(counter.counts, 0, sizeof(float) * counter.nx * counter.ny);
    }
    free(existingCounts);
    return 0;
}
