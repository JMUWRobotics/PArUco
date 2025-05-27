#include "paruco.hpp"

#include <numeric>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>
#include <variant>

namespace PArUco {

static constexpr std::array<int, 4> cornerIdxs = {0, 3, 6, 9};

static void tagCircleEstimates(
    const std::vector<cv::Point2f>& corners,
    float scale,
    std::array<std::optional<cv::Point2f>, 12>& out
) {
    // clang-format off

    out[cornerIdxs[0]] = scale * ( corners[0] - corners[3] + corners[0] - corners[1] ) + corners[0];
    out[cornerIdxs[1]] = scale * ( corners[1] - corners[0] + corners[1] - corners[2] ) + corners[1];
    out[cornerIdxs[2]] = scale * ( corners[2] - corners[1] + corners[2] - corners[3] ) + corners[2];
    out[cornerIdxs[3]] = scale * ( corners[3] - corners[0] + corners[3] - corners[2] ) + corners[3];

    // clang-format on

    for (size_t i = 0; i < cornerIdxs.size(); ++i) {
        const cv::Point2f &a = *out[cornerIdxs[i]],
                          &b = *out[cornerIdxs[(i + 1) % cornerIdxs.size()]];

        for (int j : {1, 2})
            out[cornerIdxs[i] + j] = j / 3.f * (b - a) + a;
    }
}

// Fitting Ellipse Based on the Dual Conic Model
// C. Zhao, M.L. Fu, and J.T. Cheng
// and
// Precise ellipse estimation without contour point extraction
// J.-N. Ouellet and P. Hébert
std::optional<cv::Point2f> fitEllipseDualConic(
    const cv::Mat& src,
    const cv::Point& offset,
    float gradientThreshold
) {
    using Vec5f = cv::Vec<float, 5>;

    cv::Mat1s dx, dy;
    cv::Mat1f gradMagnitude;
    auto sum_KK = cv::Matx<float, 5, 5>::zeros();
    auto sum_KZ2 = Vec5f::zeros();

    cv::spatialGradient(src, dx, dy);

    {
        cv::Mat1f gradMagnitude_(src.size());
        for (int row = 0; row < src.rows; ++row) {
            for (int col = 0; col < src.cols; ++col) {
                const cv::Vec2i grad {dx(row, col), dy(row, col)};
                gradMagnitude_(row, col) = std::sqrt(grad.dot(grad));
            }
        }
        cv::normalize(gradMagnitude_, gradMagnitude, 0.f, 1.f, cv::NORM_MINMAX);
    }

    for (int row = 0; row < src.rows; ++row) {
        for (int col = 0; col < src.cols; ++col) {
            if (gradMagnitude(row, col) < gradientThreshold)
                continue;

            const cv::Vec2f p {float(col), float(row)};
            const cv::Vec2f dp {float(dx(row, col)), float(dy(row, col))};

            const auto& [X, Y] = dp.val;
            const float Z = (dp.t() * p).val[0];

            const Vec5f Kprime {X * X, X * Y, Y * Y, X * Z, Y * Z};
            sum_KK += Kprime * Kprime.t();
            sum_KZ2 += Kprime * Z * Z;
        }
    }

    bool ok;
    const auto [A, B, C, D, E] =
        (sum_KK.inv(cv::DECOMP_SVD, &ok) * sum_KZ2).val;

    if (!ok)
        return std::nullopt;

    return cv::Point2f {0.5f * D + offset.x, 0.5f * E + offset.y};
}

struct AssociatedBlob {
    const cv::KeyPoint* keypoint;
    float distance2;
    size_t ptIdx;
};

void detect(
    const cv::Mat& image,
    const std::vector<std::vector<cv::Point2f>>& tagCorners,
    const std::vector<int>& tagIds,
    Detections& detections,
    const Params& params
) {
    detections.clear();

    if (tagCorners.empty())
        return;

    if (tagCorners.size() != tagIds.size())
        throw std::invalid_argument("need same number of tag corners and ids");

    if (std::any_of(tagCorners.begin(), tagCorners.end(), [](const auto& v) {
            return v.size() != 4;
        }))
        throw std::invalid_argument("can only have four tag corners per tag");

    const auto blobDetector =
        cv::SimpleBlobDetector::create(params.blobDetectorParameters);
    const cv::Rect imgRect {cv::Point {0, 0}, image.size()};

    std::vector<cv::KeyPoint> blobs;

    blobDetector->detect(image, blobs);
    if (blobs.empty())
        return;

    std::sort(
        blobs.begin(),
        blobs.end(),
        [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
            return a.pt.y == b.pt.y ? a.pt.x < a.pt.x : a.pt.y < b.pt.y;
        }
    );

    cv::parallel_for_(
        cv::Range(0, tagCorners.size()),
        [&](const cv::Range& range) {
            cv::Mat blobNeighborhood;
            std::unordered_map<size_t, AssociatedBlob> usedBlobs;

            for (int i = range.start; i < range.end; ++i) {
                Detection dect = {
                    .arucoId = tagIds[i],
                    .arucoCorners = tagCorners[i],
                    .circleCenters = {}
                };

                tagCircleEstimates(
                    dect.arucoCorners,
                    params.tagExpandScale,
                    dect.circleCenters
                );

                usedBlobs.clear();

                for (size_t i = 0; i < dect.circleCenters.size(); ++i) {
                    auto& center = dect.circleCenters[i];
                    const cv::KeyPoint* closestKeyPoint = nullptr;
                    float bestDistance2 =
                        std::numeric_limits<float>::infinity();

                    if (!imgRect.contains(*center)) {
                        center = std::nullopt;
                        continue;
                    }

                    for (const auto& blob : blobs) {
                        if (float distance2 =
                                cv::normL2Sqr<float>(blob.pt - *center);
                            distance2 < bestDistance2) {
                            bestDistance2 = distance2;
                            closestKeyPoint = &blob;
                        }
                    }

                    if (params.maxIdentificationDistancePixels.has_value()
                        && params.maxIdentificationDistancePixels.value()
                            < std::sqrt(bestDistance2)) {
                        center = std::nullopt;
                        continue;
                    }

                    const AssociatedBlob associatedBlob {
                        .keypoint = closestKeyPoint,
                        .distance2 = bestDistance2,
                        .ptIdx = i
                    };

                    if (auto it = usedBlobs.find(closestKeyPoint->hash());
                        it != usedBlobs.end()) {
                        const auto& [_, otherBlob] = *it;

                        if (bestDistance2 < otherBlob.distance2) {
                            dect.circleCenters[otherBlob.ptIdx] = std::nullopt;
                            it->second = std::move(associatedBlob);
                            center = associatedBlob.keypoint->pt;
                        } else {
                            center = std::nullopt;
                            continue;
                        }
                    } else {
                        usedBlobs[closestKeyPoint->hash()] =
                            std::move(associatedBlob);
                        center = closestKeyPoint->pt;
                    }

                    if (params.refineParams.has_value()) {
                        const auto& refineParams = params.refineParams.value();
                        const float radius = 0.5f * closestKeyPoint->size
                            * refineParams.blobBboxScale;
                        const cv::Point ul {
                            int(closestKeyPoint->pt.x - radius),
                            int(closestKeyPoint->pt.y - radius)
                        };
                        const cv::Point br {
                            int(std::ceil(closestKeyPoint->pt.x + radius)),
                            int(std::ceil(closestKeyPoint->pt.y + radius))
                        };
                        const cv::Rect blobNeighborhoodRoi {br, ul};

                        if ((blobNeighborhoodRoi & imgRect)
                            != blobNeighborhoodRoi) {
                            center = std::nullopt;
                            continue;
                        }

                        if (std::holds_alternative<RefineParams::OtsuEllipse>(refineParams.method)) {
                            const auto& params = std::get<RefineParams::OtsuEllipse>(refineParams.method);
                            std::vector<std::vector<cv::Point>> contours;

                            cv::threshold(
                                image(blobNeighborhoodRoi),
                                blobNeighborhood,
                                0,
                                255,
                                cv::THRESH_BINARY_INV | cv::THRESH_OTSU
                            );
                            cv::findContours(
                                blobNeighborhood,
                                contours,
                                cv::RETR_EXTERNAL,
                                cv::CHAIN_APPROX_NONE,
                                ul
                            );

                            if (contours.size() == 1) {
                                if (params.variant.has_value()) {
                                    switch (params.variant.value()) {
                                        case RefineParams::OtsuEllipse::
                                            EllipseFitVariant::ELLIPSE_AMS:
                                            center = cv::fitEllipseAMS(contours[0]).center;
                                            break;
                                        case RefineParams::OtsuEllipse::
                                            EllipseFitVariant::ELLIPSE_DIRECT:
                                            center = cv::fitEllipseDirect(contours[0]).center;
                                            break;
                                    }
                                } else {
                                    center = cv::fitEllipse(contours[0]).center;
                                }
                            } else {
                                center = std::nullopt;
                            }
                        } else if (std::holds_alternative<RefineParams::DualConic>(refineParams.method)) {
                            const auto& params = std::get<RefineParams::DualConic>(refineParams.method);
                            center = fitEllipseDualConic(
                                image(blobNeighborhoodRoi),
                                ul,
                                params.gradientThreshold
                            );
                        }
                    } /* for dect.circleCenters */
                }

                detections.push_back(std::move(dect));
            }
        }
    );
}

void detect(
    const cv::Mat& image,
    Detections& detections,
    const Params& params
) {
    if (params.aruco.dictionary.bytesList.empty())
        throw std::invalid_argument("ArUco dictionary not set");

    const auto arucoDetector = cv::aruco::ArucoDetector(
        params.aruco.dictionary,
        params.aruco.detectorParameters,
        params.aruco.refineParameters
    );

    std::vector<std::vector<cv::Point2f>> arucoCorners;
    std::vector<int> arucoIds;

    arucoDetector.detectMarkers(image, arucoCorners, arucoIds);

    detect(image, arucoCorners, arucoIds, detections, params);
}

#define HEX(rgb) \
    cv::Scalar((rgb & 0xff), ((rgb >> 8) & 0xff), ((rgb >> 16) & 0xff))

void draw(
    cv::Mat& image,
    const tbb::concurrent_vector<Detection>& detections,
    bool numbers,
    float drawScale,
    int thickness
) {
    static const std::array<cv::Scalar, 6> colors = {
        HEX(0x66cdaa),
        HEX(0xff8c00),
        HEX(0x00ff00),
        HEX(0x0000ff),
        HEX(0x1e90ff),
        HEX(0xff1493)
    };

    for (const auto& d : detections) {
        const size_t colorIdx = d.arucoId % colors.size();
        for (size_t i = 0; i < d.arucoCorners.size(); ++i) {
            cv::line(
                image,
                d.arucoCorners[i],
                d.arucoCorners[(i + 1) % d.arucoCorners.size()],
                colors[colorIdx],
                thickness
            );
        }
        cv::putText(
            image,
            std::to_string(d.arucoId),
            0.25f
                * std::accumulate(
                    d.arucoCorners.begin(),
                    d.arucoCorners.end(),
                    cv::Point2f(0, 0)
                ),
            cv::FONT_HERSHEY_SIMPLEX,
            drawScale,
            colors[colorIdx],
            thickness
        );

        for (size_t i = 0; i < d.circleCenters.size(); ++i) {
            auto pos = d.circleCenters[i];
            if (!pos.has_value())
                continue;

            cv::drawMarker(
                image,
                *pos,
                colors[colorIdx],
                cv::MARKER_CROSS,
                20 * drawScale,
                thickness
            );
            if (numbers) {
                cv::putText(
                    image,
                    std::to_string(i),
                    *pos,
                    cv::FONT_HERSHEY_SIMPLEX,
                    drawScale * 0.75,
                    colors[colorIdx],
                    thickness
                );
            }
        }
    }
}

}  // namespace PArUco