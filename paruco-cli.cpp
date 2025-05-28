#include <cxxopts.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <variant>

#include "paruco.hpp"

static cv::aruco::PredefinedDictionaryType dict_from_str(std::string str) {
    using namespace cv::aruco;
#define ENTRY(s) {#s, s}
    static const std::map<std::string, PredefinedDictionaryType> map {
        ENTRY(DICT_6X6_100),
        ENTRY(DICT_APRILTAG_36h11)
    };
#undef ENTRY

    std::transform(str.begin(), str.end(), str.begin(), ::toupper);

    return map.at(str);
}

static std::variant<PArUco::RefineParams::OtsuEllipse, PArUco::RefineParams::DualConic> method_from_str(const std::string& str) {
    if (str == "otsu")
        return PArUco::RefineParams::OtsuEllipse { std::nullopt };
    else if (str == "otsu_ams")
        return PArUco::RefineParams::OtsuEllipse { PArUco::RefineParams::OtsuEllipse::EllipseFitVariant::ELLIPSE_AMS };
    else if (str == "otsu_direct")
        return PArUco::RefineParams::OtsuEllipse { PArUco::RefineParams::OtsuEllipse::EllipseFitVariant::ELLIPSE_DIRECT };
    else if (str == "dual_conic")
        return PArUco::RefineParams::DualConic { };
    else
        throw std::invalid_argument("unknown method"); 
}

int main(int argc, const char* const* argv) {
    cxxopts::Options options("paruco-cli", "cli for PArUco detector");

    // clang-format off

    options.add_options()
        ("h,help", "display this message")
        ("input", "input image", cxxopts::value<std::string>())
        ("output", "output file", cxxopts::value<std::string>()->default_value("out.txt"))
        ("draw", "draw detections and show image")
        ("tag-expand-scale", "distance between edge of tag to circle over tag size", cxxopts::value<float>())
        ("max-identification-distance-pixels", "maximum pixel distance for circle identification", cxxopts::value<float>())
        ("refine-bbox-scale", "bounding box scale around circle estimate for refinement", cxxopts::value<float>())
        ("refine-method", "method of refinement one of { otsu, otsu_ams, otsu_direct, dual_conic }", cxxopts::value<std::string>())
        ("dual-conic-gradient-threshold", "gradient threshold for dual conic refinement method", cxxopts::value<float>())
        ("dictionary", "cv::aruco::dictionary for tag detection", cxxopts::value<std::string>()->default_value("DICT_6X6_100"))
        ("compat", "output coordinates w.r.t. the image center and inverted y-axis and pixel origin in top left of pixel");

    options.parse_positional({"input", "output"});
    options.positional_help("input [output]");

    // clang-format on

    const auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cerr << options.help() << '\n';
        return 0;
    }

    const auto input =
        cv::imread(args["input"].as<std::string>(), cv::IMREAD_GRAYSCALE);

    PArUco::Params params(dict_from_str(args["dictionary"].as<std::string>()));
    if (args.count("tag-expand-scale"))
        params.tagExpandScale = args["tag-expand-scale"].as<float>();
    if (args.count("max-identification-distance-pixels"))
        params.maxIdentificationDistancePixels =
            args["max-identification-distance-pixels"].as<float>();
    if (args.count("refine-bbox-scale") || args.count("refine-method")) {
        params.refineParams = PArUco::RefineParams {};
        if (args.count("refine-bbox-scale"))
            params.refineParams.value().blobBboxScale =
                args["refine-bbox-scale"].as<float>();
        if (args.count("refine-method"))
            params.refineParams.value().method = method_from_str(args["refine-method"].as<std::string>());

        if (std::holds_alternative<PArUco::RefineParams::DualConic>(params.refineParams.value().method) && args.count("dual-conic-gradient-threshold"))
            std::get<PArUco::RefineParams::DualConic>(params.refineParams.value().method).gradientThreshold = args["dual-conic-gradient-threshold"].as<float>();
    }

    PArUco::Detections detections;
    PArUco::detect(input, detections, params);

    if (detections.empty()) {
        std::cerr << "no detections\n";
        return 0;
    }

    std::cerr << detections.size() << " detections\n";

    {
        std::ofstream outfile(args["output"].as<std::string>());
        for (const auto& d : detections) {
            for (size_t i = 0; i < d.circleCenters.size(); ++i) {
                if (!d.circleCenters[i].has_value()) {
                    continue;
                }

                cv::Point2f p = *d.circleCenters[i];
                if (args.count("compat")) {
                    p.x -= input.cols / 2.f - 0.5f;
                    p.y -= input.rows / 2.f - 0.5f;
                    p.y = -p.y;
                }

                outfile << 'T' << std::setfill('0') << std::setw(3) << d.tagId
                        << 'C' << std::setw(2) << i << std::setw(0) 
                        << '\t' << p.x << '\t' << p.y << '\n';
            }
        }
    }

    if (args.count("draw")) {
        cv::Mat color;
        cv::cvtColor(input, color, cv::COLOR_GRAY2BGR);
        PArUco::draw(color, detections, true, 0.9f, 2);

        cv::namedWindow("detections", cv::WINDOW_KEEPRATIO);
        cv::imshow("detections", color);
        cv::waitKey();
        cv::destroyAllWindows();
    }

    return 0;
}