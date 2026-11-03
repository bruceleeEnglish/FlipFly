// Projection-scaling microbenchmark for §5.2.1.
//
// Times two ways of applying a per-query projection to a random
// float32 vector of dimension D, for D in {1024, 2048, ..., 16384}:
//
//   FWHT path (SRHT / SRHT+PCA):
//       y = (1/sqrt(D)) * H_D * D_rad * x
//       implementation copied verbatim from
//       `DCO/L2/SRHTADSampling.hpp::fwht_inplace`, so the timing
//       reflects exactly the code path used by the search benchmark.
//
//   Dense matvec (ADSampling / DDCres / DADE):
//       y = H_dense * x
//       implementation matches `DCO/L2/ADSampling.hpp::query_transform`
//       (Eigen MatrixXf * VectorXf, which lowers to the BLAS-style
//       GEMV used at runtime).
//
// Output:  CSV on stdout with columns `dim, fwht_ms, matvec_ms`,
//          one row per measured dimension; the reported time is the
//          median across `n_repeats` calls after `n_warmup` warmup
//          iterations.
//
// Usage:
//      ./bench_projection [n_repeats=200] [n_warmup=20]  > result.csv

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Dense>


// ----- exact copy of SRHTADSampling::fwht_inplace -------------------------- //
static void fwht_inplace(float *x, size_t D) {
    for (size_t h = 1; h < D; h <<= 1) {
        for (size_t i = 0; i < D; i += (h << 1)) {
            for (size_t j = i; j < i + h; ++j) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}


static double median_of(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}


static double time_fwht(size_t D, int n_repeats, int n_warmup) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> bit(0, 1);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> rad(D);
    for (size_t i = 0; i < D; ++i) rad[i] = bit(rng) ? 1.0f : -1.0f;

    std::vector<float> x_buf(D);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto one_pass = [&]() {
        // Step 1: D_rad * x
        for (size_t i = 0; i < D; ++i) x_buf[i] *= rad[i];
        // Step 2: H * (...)  via in-place butterfly FWHT
        fwht_inplace(x_buf.data(), D);
        // Step 3: scale by 1/sqrt(D)
        for (size_t i = 0; i < D; ++i) x_buf[i] *= scale;
    };

    for (int w = 0; w < n_warmup; ++w) {
        for (size_t i = 0; i < D; ++i) x_buf[i] = nd(rng);
        one_pass();
    }

    std::vector<double> times;
    times.reserve(n_repeats);
    for (int r = 0; r < n_repeats; ++r) {
        for (size_t i = 0; i < D; ++i) x_buf[i] = nd(rng);
        auto t0 = std::chrono::steady_clock::now();
        one_pass();
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return median_of(times);
}


static double time_matvec(size_t D, int n_repeats, int n_warmup) {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    Eigen::MatrixXf H(D, D);
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) {
            H(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = nd(rng);
        }
    }

    // ADSampling::query_transform actually does a 1xD * DxD product, but
    // a Dx1 = DxD * Dx1 (i.e. VectorXf) has identical flops and exercises
    // the same Eigen GEMV path. We use VectorXf here for clarity.
    Eigen::VectorXf x(D);
    Eigen::VectorXf y(D);

    auto one_pass = [&]() {
        y.noalias() = H * x;
    };

    for (int w = 0; w < n_warmup; ++w) {
        for (size_t i = 0; i < D; ++i)
            x(static_cast<Eigen::Index>(i)) = nd(rng);
        one_pass();
    }

    std::vector<double> times;
    times.reserve(n_repeats);
    for (int r = 0; r < n_repeats; ++r) {
        for (size_t i = 0; i < D; ++i)
            x(static_cast<Eigen::Index>(i)) = nd(rng);
        auto t0 = std::chrono::steady_clock::now();
        one_pass();
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return median_of(times);
}


int main(int argc, char **argv) {
    std::vector<size_t> dims = {1024, 2048, 4096, 8192, 16384};
    int n_repeats = 200;
    int n_warmup  = 20;

    if (argc > 1) n_repeats = std::atoi(argv[1]);
    if (argc > 2) n_warmup  = std::atoi(argv[2]);

    std::cerr << "# n_repeats=" << n_repeats
              << ", n_warmup=" << n_warmup << "\n";

    // CSV header on stdout (consumed by plot_projection_scaling.py)
    std::cout << "dim,fwht_ms,matvec_ms\n";

    for (size_t D : dims) {
        std::cerr << "timing d=" << D << " ..." << std::flush;
        double tf = time_fwht(D, n_repeats, n_warmup);
        double tm = time_matvec(D, n_repeats, n_warmup);
        std::cout << D << "," << tf << "," << tm << "\n";
        std::cout.flush();
        std::cerr << " fwht=" << tf << " ms, matvec=" << tm
                  << " ms (ratio " << (tm / tf) << "x)\n";
    }
    return 0;
}
