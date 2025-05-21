#include "paruco.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <genicvbridge.hpp>

int main(void) {
    auto capture = XVII::GenICamVideoCapture::OpenAnyCamera();

    cv::Mat image, color;

    cv::namedWindow("detections", cv::WINDOW_KEEPRATIO);

    PArUco::Params params(cv::aruco::DICT_6X6_100);
    params.refineParams = PArUco::RefineParams {
        .method = PArUco::RefineParams::OTSU_ELLIPSE | PArUco::RefineParams::ELLIPSE_AMS
    };
    tbb::concurrent_vector<PArUco::Detection> detections;

    while (cv::pollKey() != 'q') {
        if (!capture->read(image))
            continue;

        PArUco::detect(image, detections, params);

        cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
        PArUco::draw(color, detections, 0.9f, 2);

        cv::imshow("detections", color);
    }
}
