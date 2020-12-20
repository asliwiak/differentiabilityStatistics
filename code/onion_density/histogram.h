//written by Qiqi Wang
// density/density.h
#pragma once
#include<cinttypes>

struct ValueWeight {
    double value, weight;
};

__device__ __forceinline__
void pIncrement(
    float* tmpCounts,
    double x0, double dx, uint32_t nx, double x, float w
) {
    uint32_t i = uint32_t((x - x0) / dx);

    if (i < nx) {
        atomicAdd(tmpCounts + i, w);
    }
}

template<bool accumulate, class Map, class ObjFunc, uint32_t mapDim>
__global__
void pRun(
    Map map, ObjFunc obj, uint32_t nIters,
    uint32_t nPoints, double (*points)[mapDim],
    float* tmpCounts, double x0, double dx, uint32_t nx
) {
    uint32_t ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < nPoints) {
        for (uint32_t iIter = 0; iIter < nIters; ++iIter) {
            map.map(points[ind]);
            ValueWeight valWgt = obj.eval(points[ind]);

	    //if (fabs(valWgt.weight) < 0.5 )
		//    continue;

            if (accumulate) {
                pIncrement(tmpCounts, x0, dx, nx, valWgt.value, valWgt.weight);
            }
        }
    }
}

template<uint32_t mapDim>
struct Counter {
  public:
    const uint32_t nx;
    const double x0, dx;
    double* counts;

  private:
    float* tmpCounts;
    float* cpuCounts;

    double (*points)[mapDim];
    uint32_t nPoints;

    void pAddToCpuCounterAndClear() {
        cudaMemcpy(cpuCounts, tmpCounts, sizeof(float) * nx,
                   cudaMemcpyDeviceToHost);
        cudaMemset(tmpCounts, 0, sizeof(float) * nx);

        for (uint32_t i = 0; i < nx; ++i) {
            counts[i] += (double)cpuCounts[i];
        }
    }

  public:
    Counter(uint32_t nx, double x0, double dx)
        : nx(nx), x0(x0), dx(dx) {
        cudaError_t err = cudaMalloc(&tmpCounts, sizeof(float) * nx);
        cpuCounts = new float[nx];
        counts = new double[nx];
        memset(counts, 0, sizeof(double) * nx);
        nPoints = 0;
    }

    void init(uint32_t nPoints) {
        cudaMemset(tmpCounts, 0, sizeof(float) * nx * nx);

        if (this->nPoints) {
            cudaFree(points);
        }

        this->nPoints = nPoints;
        cudaMalloc(&points, sizeof(double) * nPoints * mapDim);

        double* cpuPoints = new double[nPoints * mapDim];

        for (uint32_t i = 0; i < nPoints * mapDim; ++i) {
            cpuPoints[i] = (rand() / double(RAND_MAX) + rand())
                           / double(RAND_MAX);
        }

        cudaMemcpy(points, cpuPoints, sizeof(double) * nPoints * mapDim,
                   cudaMemcpyHostToDevice);
        delete[] cpuPoints;
    }

    template<class Map, class ObjFunc>
    void run(Map map, ObjFunc obj, uint32_t iters, bool accumulate) {
        if (accumulate) {
            pRun<true, Map, ObjFunc, mapDim><<<ceil(nPoints / 64.), 64>>>(
                map, obj, iters, nPoints, points, tmpCounts, x0, dx, nx);

            pAddToCpuCounterAndClear();
        } else {
            pRun<false, Map, ObjFunc, mapDim><<<ceil(nPoints / 64.), 64>>>(
                map, obj, iters, nPoints, points, tmpCounts, x0, dx, nx);
        }

        cudaDeviceSynchronize();
    }

    ~Counter() {
        cudaFree(tmpCounts);
        delete[] cpuCounts;
        delete[] counts;
    }
};
