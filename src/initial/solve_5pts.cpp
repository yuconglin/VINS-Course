#include "initial/solve_5pts.h"

using namespace cv;

namespace MotionEstimation {

bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres,
                     Matrix3d &Rotation, Vector3d &Translation) {
  if (corres.size() >= 15) {
    vector<cv::Point2f> ll, rr;
    for (int i = 0; i < int(corres.size()); i++) {
      ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
      rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
    }
    const cv::Mat E =
        cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99);
    const cv::Mat cameraMatrix =
        (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat rot, trans;
    int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans);

    Eigen::Matrix3d R;
    Eigen::Vector3d T;

    for (int i = 0; i < 3; i++) {
      T(i) = trans.at<double>(i, 0);
      for (int j = 0; j < 3; j++) {
        R(i, j) = rot.at<double>(i, j);
      }
    }

    Rotation = R.transpose();
    Translation = -R.transpose() * T;

    return inlier_cnt > 12;
  }

  return false;
}
}  // namespace MotionEstimation
