#include <cmath> // For mathematical functions like cos, sin, sinh, etc.

extern "C" {
    void compute_4vectors(
        const double* pt, const double* eta, const double* phi, const double* mass,
        double* px, double* py, double* pz, double* energy,
        int size
    ) {
        for (int i = 0; i < size; i++) {
            px[i] = pt[i] * cos(phi[i]);
            py[i] = pt[i] * sin(phi[i]);
            pz[i] = pt[i] * sinh(eta[i]);
            energy[i] = sqrt(px[i] * px[i] + py[i] * py[i] + pz[i] * pz[i] + mass[i] * mass[i]);
        }
    }
}

