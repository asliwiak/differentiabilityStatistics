//written by Qiqi Wang
// density.h
#pragma once
#include<cinttypes>

struct ValueWeight {
    double x, y, weight;
};

__device__ __forceinline__
void pIncrement(
    uint32_t * tmpCounts,
    double x0, double y0,
    double dx, double dy,
    uint32_t nx, uint32_t ny,
    double x, double y
) {
    if (x < x0 or y < y0)
        return;

    uint32_t i = uint32_t((x - x0) / dx);
    uint32_t j = uint32_t((y - y0) / dy);

    if (i < nx and j < ny) {
        uint32_t ind = i * ny + j;
        atomicAdd(tmpCounts + ind, 1);
    }
}

template<bool accumulate, class Map, class ObjFunc, uint32_t mapDim>
__global__
void pRun(
    Map map, ObjFunc obj, uint32_t nIters,
    uint32_t nPoints, double (*points)[mapDim],
    uint32_t * tmpCounts,
    double x0, double y0,
    double dx, double dy,
    uint32_t nx, uint32_t ny
) {
    uint32_t ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < nPoints) {
        for (uint32_t iIter = 0; iIter < nIters; ++iIter) {
            map.map(points[ind]);
            ValueWeight valWgt = obj.eval(points[ind]);

            if (accumulate) {
                pIncrement(tmpCounts, x0, y0, dx, dy, nx, ny,
                           valWgt.x, valWgt.y);
            }
        }
    }
}

template<uint32_t mapDim>
struct Counter {
public:
    const uint32_t nx, ny;
    const double x0, y0, dx, dy;
    float* counts;

private:
    uint32_t* tmpCounts;
    uint32_t* cpuCounts;

    double (*points)[mapDim];
    uint32_t nPoints;

    void pAddToCpuCounterAndClear() {
        cudaMemcpy(cpuCounts, tmpCounts, sizeof(uint32_t) * nx * ny,
                   cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < nx * ny; ++i) {
            counts[i] += (float)cpuCounts[i];
        }
        cudaMemset(tmpCounts, 0, sizeof(uint32_t) * nx * ny);
    }

public:
    Counter(uint32_t nx, uint32_t ny,
            double x0, double y0,
            double dx, double dy
           ) : nx(nx), ny(ny), x0(x0), y0(y0), dx(dx), dy(dy) {
        cudaMalloc(&tmpCounts, sizeof(float) * nx * nx);
        cpuCounts = new uint32_t[nx * nx];
        counts = new float[nx * nx];
        memset(counts, 0, sizeof(float) * nx * nx);
        nPoints = 0;
    }

    template<class Map>
    void init(Map map, uint32_t nPoints) {
        cudaMemset(tmpCounts, 0, sizeof(float) * nx * nx);

        if (this->nPoints) {
            cudaFree(points);
        }
        this->nPoints = nPoints;
        cudaMalloc(&points, sizeof(double) * nPoints * mapDim);

        double *cpuPoints = new double[nPoints * mapDim];

        for (uint32_t i = 0; i < nPoints; ++i) {
            map.init(cpuPoints + (i * mapDim));
        }

        cudaMemcpy(points, cpuPoints, sizeof(double) * nPoints * mapDim,
                   cudaMemcpyHostToDevice);
        delete[] cpuPoints;
    }

    template<class Map, class ObjFunc>
    void run(Map map, ObjFunc obj, uint32_t iters, bool accumulate) {
        if (accumulate) {
            pRun<true, Map, ObjFunc, mapDim><<<ceil(nPoints / 64.), 64>>>(
                map, obj, iters, nPoints, points,
                tmpCounts, x0, y0, dx, dy, nx, ny);
            pAddToCpuCounterAndClear();
        } else {
            pRun<false, Map, ObjFunc, mapDim><<<ceil(nPoints / 64.), 64>>>(
                map, obj, iters, nPoints, points,
                tmpCounts, x0, y0, dx, dy, nx, ny);
        }

        cudaDeviceSynchronize();
    }

    ~Counter() {
        cudaFree(tmpCounts);
        delete[] cpuCounts;
        delete[] counts;
    }
};
