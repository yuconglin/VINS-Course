#pragma once

#include <vector>

#include "../parameters.h"
using namespace std;

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
using namespace Eigen;

/* This class help you to calibrate extrinsic rotation between imu and camera
 * when your totally don't konw the extrinsic parameter */
class InitialEXRotation {
 public:
  InitialEXRotation();

  // Chapter 7: page 10-11.
  bool CalibrationExRotation(const vector<pair<Vector3d, Vector3d>> &corres,
                             const Quaterniond &delta_q_imu,
                             Matrix3d &calib_ric_result);

 private:
  // Computes relative rotation between two image frames using fundmemtal matrix
  // decomposition.
  Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres);

  // Triangluate and return the ratio of inversely projected 3d points in front
  // of both cameras.
  double testTriangulation(const vector<cv::Point2f> &l,
                           const vector<cv::Point2f> &r, cv::Mat_<double> R,
                           cv::Mat_<double> t);

  // Decompose the fundmental matrix into left and right {R,t}s.
  // SLAM 14: p 144-145.
  void decomposeE(cv::Mat E, cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                  cv::Mat_<double> &t1, cv::Mat_<double> &t2);

  int frame_count;

  vector<Matrix3d> Rc;
  vector<Matrix3d> Rimu;
  vector<Matrix3d> Rc_g;
  Matrix3d ric;
};
