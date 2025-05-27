#include <genicvbridge.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "paruco.hpp"

int main(void) {
    auto capture = XVII::GenICamVideoCapture::OpenAnyCamera();

    cv::Mat image, color;

    cv::namedWindow("detections", cv::WINDOW_KEEPRATIO);

    PArUco::Params params(cv::aruco::DICT_APRILTAG_36h11);
    params.refineParams = PArUco::RefineParams {
        .method = PArUco::RefineParams::DualConic { 0.4f }
    };
    tbb::concurrent_vector<PArUco::Detection> detections;

    while (cv::pollKey() != 'q') {
        if (!capture->read(image))
            continue;

        PArUco::detect(image, detections, params);

        cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
        PArUco::draw(color, detections, false, 0.9f, 2);

        cv::imshow("detections", color);
    }
}
