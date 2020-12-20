//written by Adam Sliwiak
struct HybridMap {
    const double power, height;

    __device__ __forceinline__
    void map(double xg[2]) {
        
        double fun = fabs(1.0 - 2 * xg[0]);
        double xNext = height * sqrt(1.0 - pow(fun, power));
	//double xNext = height * pow(1.0 - pow(fun, power), 2);
        double sign = (xg[0] <= 0.5) ? 1.0 : -1.0;
        double dphidx = sign * height * power * pow(fun, power - 1) / sqrt(1.0 - pow(fun, power));
	//double dphidx = sign * 4 * height * power * pow(fun, power - 1) * (1.0 - pow(fun, power));
        double d2phidx2 = -2.0 * height * power / (1.0 - pow(fun, power)) * ((power - 1) * pow(fun, power - 2) * sqrt(1.0 - pow(fun, power)) + power/2 * pow(fun, 2 * power - 2) / sqrt(1 - pow(fun, power)));
        //double d2phidx2 = 8 * power * power * height * pow(fun, 2 * power - 2) - 8 * (power - 1) * power * height * (1.0 - pow(fun, power)) * pow(fun, power - 2);
	double gNext = xg[1] / dphidx - d2phidx2 / dphidx / dphidx;

        xg[0] = xNext;
        xg[1] = gNext;
    }

    __device__ __forceinline__
    double f_power(const double xg[2]) {
        
        double fun = fabs(1.0 - 2 * xg[0]);
        return -height/2 / sqrt(1.0 - pow(fun, power)) * pow(fun, power) * log(fun);
	//return -2 * height * pow(fun, power) * (1.0 - pow(fun, power)) * log(fun);
    }

    __device__ __forceinline__
    double f_power_grad(const double xg[2]) {
        
        double fun = fabs(1.0 - 2 * xg[0]);
        double sign = (xg[0] <= 0.5) ? 1.0 : -1.0;
        double d2phi_dpower_dx = sign * height/2 * pow(fun, power - 1) / pow(1.0 - pow(fun, power), 1.5) * (-2 * pow(fun, power) - power * pow(fun, power) * log(fun) + 2 * power * log(fun) + 2);
	/*double d2phi_dpower_dx = sign * 4 * height * (1.0 - pow(fun, power)) * pow(fun, power - 1)
		               + sign * 4 * height * power * (1.0 - pow(fun, power)) * pow(fun, power - 1) * log(fun)
			       - sign * 4 * height * power * pow(fun, 2 * power - 1) * log(fun);*/
	double dphidx = sign * height * power * pow(fun, power - 1) / sqrt(1.0 - pow(fun, power));
        //double dphidx = sign * 4 * height * power * pow(fun, power - 1) * (1.0 - pow(fun, power));
	return d2phi_dpower_dx / dphidx;
    }
/*
    __device__ __forceinline__
    double ts_operator(const double xg[2]) {

        double fun = fabs(1.0 - 2 * xg[0]);
        double sign = (xg[0] <= 0.5) ? 1.0 : -1.0;
        double dphidx = sign * height * power * pow(fun, power - 1) / sqrt(1.0 - pow(fun, power));
        double d2phidx2 = -2.0 * height * power / (1.0 - pow(fun, power)) * ((power - 1) * pow(fun, power - 2) * sqrt(1.0 - pow(fun, power)) + power/2 * pow(fun, 2 * power - 2) / sqrt(1 - pow(fun, power)));    
       return - xg[1] / dphidx / fabs(dphidx) + d2phidx2 / pow(fabs(dphidx), 3); 

   }*/    
};
