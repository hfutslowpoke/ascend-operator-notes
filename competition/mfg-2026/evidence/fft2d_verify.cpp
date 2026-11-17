// Local-only verification for ops-fft fft2_d (32x32 C2C).
// Not part of the third-party repository; only calls the built ACLFFT API.
#include <acl/acl.h>
#include <cann_ops_fft.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(cond, msg) do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s (line %d) msg=%s\n", msg, __LINE__, aclGetRecentErrMsg()); return 1; } } while (0)

int main()
{
    const int nx = 32, ny = 32, batch = 1;
    const int n = nx * ny;
    std::vector<aclfftComplex> in(n), out(n);
    // Impulse at (3,7): the exact DFT is exp(-2*pi*i*(3*k1/nx + 7*k2/ny)).
    for (int i = 0; i < n; ++i) { in[i].x = 0.0f; in[i].y = 0.0f; }
    in[3 * ny + 7].x = 1.0f;

    CHECK(aclInit(nullptr) == ACL_SUCCESS, "aclInit");
    CHECK(aclrtSetDevice(0) == ACL_SUCCESS, "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream) == ACL_SUCCESS, "aclrtCreateStream");

    aclfftHandle plan = nullptr;
    aclfftResult pr = aclfftPlan2d(&plan, batch, nx, ny, ACLFFT_C2C);
    if (pr != ACLFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: aclfftPlan2d ret=%d msg=%s\n", (int)pr, aclGetRecentErrMsg());
        return 1;
    }

    aclfftComplex* din = nullptr;
    aclfftComplex* dout = nullptr;
    CHECK(aclrtMalloc(reinterpret_cast<void**>(&din), n * sizeof(aclfftComplex), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc in");
    CHECK(aclrtMalloc(reinterpret_cast<void**>(&dout), n * sizeof(aclfftComplex), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc out");
    CHECK(aclrtMemcpy(din, n * sizeof(aclfftComplex), in.data(), n * sizeof(aclfftComplex), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, "H2D");

    double bestMs = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        aclfftResult er = aclfftExecC2C(plan, din, dout, ACLFFT_FORWARD);
        if (er != ACLFFT_SUCCESS) {
            std::fprintf(stderr, "FAIL: aclfftExecC2C ret=%d msg=%s\n", (int)er, aclGetRecentErrMsg());
            return 1;
        }
        aclError se = aclrtSynchronizeStream(stream);
        if (se != ACL_SUCCESS) {
            std::fprintf(stderr, "FAIL: sync ret=%d msg=%s\n", (int)se, aclGetRecentErrMsg());
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("run %d: %.4f ms\n", rep + 1, ms);
        if (ms < bestMs) bestMs = ms;
    }
    std::printf("best: %.4f ms\n", bestMs);
    CHECK(aclrtMemcpy(out.data(), n * sizeof(aclfftComplex), dout, n * sizeof(aclfftComplex), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS, "D2H");

    double maxAbs = 0.0, maxRel = 0.0;
    int bad = 0;
    for (int k1 = 0; k1 < nx; ++k1) {
        for (int k2 = 0; k2 < ny; ++k2) {
            const double phase = -2.0 * M_PI * (3.0 * k1 / nx + 7.0 * k2 / ny);
            const double er = std::cos(phase), ei = std::sin(phase);
            const int idx = k1 * ny + k2;
            const double ar = out[idx].x, ai = out[idx].y;
            const double ae = std::hypot(ar - er, ai - ei);
            const double re = ae / std::max(1.0, std::hypot(er, ei));
            if (ae > maxAbs) maxAbs = ae;
            if (re > maxRel) maxRel = re;
            if (ae > 2e-3) ++bad;
        }
    }
    std::printf("shape=%dx%d batch=%d\n", nx, ny, batch);
    std::printf("max_abs=%.6e max_rel=%.6e bad=%d\n", maxAbs, maxRel, bad);
    std::printf("RESULT: %s\n", bad == 0 ? "PASS" : "FAIL");

    aclfftDestroy(plan);
    aclrtFree(din);
    aclrtFree(dout);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return bad == 0 ? 0 : 2;
}
