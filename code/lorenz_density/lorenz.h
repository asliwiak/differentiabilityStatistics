//written by Qiqi Wang
struct Lorenz {
    const double sigma, rho, beta, dt;

    void init(double xyz[3]) {
        for (int i = 0; i < 3; ++i)
            xyz[i] = (rand() / double(RAND_MAX) + rand()) / double(RAND_MAX) / 50;
        xyz[2] += 28;
    }

    __device__ __forceinline__
    void ddt(double ddtxyz[3], const double xyz[3]) {
        ddtxyz[0] = sigma * (xyz[1] - xyz[0]);
        ddtxyz[1] = xyz[0] * (rho - xyz[2]) - xyz[1];
        ddtxyz[2] = xyz[0] * xyz[1] - beta * xyz[2];
    }

    __device__ __forceinline__
    void map(double xyz[3]) {
        for (uint8_t istep = 0; istep < 50; ++istep) {
            double ddtxyz[3];
            ddt(ddtxyz, xyz);
            double xyzMid[3];

            for (uint8_t i = 0; i < 3; ++i)
                xyzMid[i] = xyz[i] + 0.5 * dt * ddtxyz[i];
    
            ddt(ddtxyz, xyzMid);
    
            for (uint8_t i = 0; i < 3; ++i)
                xyz[i] += dt * ddtxyz[i];
        }
    }
};
