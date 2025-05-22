#pragma once

#include <tbb/concurrent_vector.h>

#include <opencv2/aruco.hpp>
#include <opencv2/features2d.hpp>
#include <optional>

namespace PArUco {

struct Detection {
    int arucoId;
    std::vector<cv::Point2f> arucoCorners;
    std::array<std::optional<cv::Point2f>, 12> circleCenters;
};

struct RefineParams {
    enum Method {
        OTSU_ELLIPSE = (1 << 0),
        ELLIPSE_AMS = (1 << 1),
        ELLIPSE_DIRECT = (1 << 2)
    };

    float blobBboxScale = 2.f;
    int method = Method::OTSU_ELLIPSE;
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
    float drawScale = 1.0f,
    int thickness = 1
);

}  // namespace PArUco