written by Qiqi Wang
#include<cstdio>
#include<cassert>
#include<algorithm>

#include"histogram.h"
#include"hybrid_map.h"

struct AccumG {
    __device__ __forceinline__
    static ValueWeight eval(double xg[2]) {
        return ValueWeight{xg[0], xg[1]};
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

    const int nx = 2048;
    Counter<2> counter(nx, 0., 1./nx);

    float parameters[2];
    assert(fread(parameters, sizeof(float), 2, stdin) == 2);
    HybridMap map{parameters[0], parameters[1]};
    AccumG obj;
    fprintf(stderr, "Ready with parameters %f %f\n",
            parameters[0], parameters[1]);

    uint32_t nIters;
    assert(fread(&nIters, sizeof(uint32_t), 1, stdin) == 1);

    for (uint32_t iIter = 0; iIter < nIters; iIter+=4) {
        counter.init(80000);
        counter.run(map, obj, 256, false);

        if (iIter % 128 == 0) {
            fprintf(stderr, "%u/%u iterations\n", iIter, nIters);
        }

        counter.run(map, obj, std::min(4u, nIters - iIter) * 1024, true);
    }

    fwrite(counter.counts, sizeof(double), nx, stdout);
    return 0;
}
