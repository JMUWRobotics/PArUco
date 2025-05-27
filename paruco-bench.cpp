#include <benchmark/benchmark.h>

#include <opencv2/imgcodecs.hpp>

#include "paruco.hpp"

#define BENCH_IMAGE(filename, ext, dict) \
    static void bench_##filename(benchmark::State& state) { \
        const auto img = cv::imread(#filename ext, cv::IMREAD_GRAYSCALE); \
\
        PArUco::Detections detections; \
        PArUco::Params params(cv::aruco::dict); \
\
        for (auto _ : state) { \
            PArUco::detect(img, detections, params); \
            benchmark::DoNotOptimize(detections); \
        } \
    } \
    BENCHMARK(bench_##filename) \
        ->Unit(benchmark::TimeUnit::kMillisecond) \
        ->ReportAggregatesOnly(false) \
        ->DisplayAggregatesOnly(true)

BENCH_IMAGE(rendered, ".png", DICT_6X6_100)->Repetitions(10);
BENCH_IMAGE(example, ".png", DICT_6X6_100)->Repetitions(10);

static void bench_fitEllipseDualConic(benchmark::State &state) {
    const auto img = cv::imread("circle.png", cv::IMREAD_GRAYSCALE);

    for (auto _ : state) {
        auto ret = PArUco::fitEllipseDualConic(img, cv::Point { 0, 0 });
        benchmark::DoNotOptimize(ret);
    }
}

BENCHMARK(bench_fitEllipseDualConic)
    ->ReportAggregatesOnly(false)->DisplayAggregatesOnly(true)->Repetitions(10);

BENCHMARK_MAIN();
