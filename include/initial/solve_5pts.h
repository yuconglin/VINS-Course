#pragma once

#include <vector>
using namespace std;

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
using namespace Eigen;

namespace MotionEstimation {
bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres,
                     Matrix3d &R, Vector3d &T);
}
