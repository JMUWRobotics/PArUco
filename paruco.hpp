#pragma once

#include <tbb/concurrent_vector.h>

#include <opencv2/aruco.hpp>
#include <opencv2/features2d.hpp>
#include <optional>
#include <variant>

namespace PArUco {

struct Detection {
    int arucoId;
    std::vector<cv::Point2f> arucoCorners;
    std::array<std::optional<cv::Point2f>, 12> circleCenters;
};

struct RefineParams {
    struct OtsuEllipse {
        enum class EllipseFitVariant { ELLIPSE_AMS, ELLIPSE_DIRECT };
        std::optional<EllipseFitVariant> variant;
    };
    struct DualConic {
        float gradientThreshold = 0.75f;
    };

    float blobBboxScale = 2.f;
    std::variant<OtsuEllipse, DualConic> method;
};

struct Params {
    struct {
        cv::aruco::Dictionary dictionary = {};
        cv::aruco::DetectorParameters detectorParameters = {};
        cv::aruco::RefineParameters refineParameters = {};
    } aruco;

    cv::SimpleBlobDetector::Params blobDetectorParameters = {};

    float tagExpandScale = 0.5f;

    std::optional<float> maxIdentificationDistancePixels = 15.f;
    std::optional<RefineParams> refineParams = RefineParams {};

    inline Params() {
        blobDetectorParameters.filterByColor = true;
        blobDetectorParameters.blobColor = 0;
    }

    inline Params(cv::aruco::PredefinedDictionaryType predefinedDictionary) :
        Params() {
        aruco.dictionary =
            cv::aruco::getPredefinedDictionary(predefinedDictionary);
    }
};

using Detections = tbb::concurrent_vector<Detection>;

void detect(
    const cv::Mat& image,
    Detections& detections,
    const Params& params = Params {}
);

/* for tags with other types of detectors, e.g. apriltag library */
void detect(
    const cv::Mat& image,
    const std::vector<std::vector<cv::Point2f>>& tagCorners,
    const std::vector<int>& tagIds,
    Detections& detections,
    const Params& params
);

void draw(
    cv::Mat& image,
    const Detections& detections,
    bool numbers = true,
    float drawScale = 1.0f,
    int thickness = 1
);

}  // namespace PArUco