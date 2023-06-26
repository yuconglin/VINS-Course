#include "backend/edge_reprojection.h"

#include <glog/logging.h>

#include <iostream>

#include "../thirdparty/Sophus/sophus/se3.hpp"
#include "backend/vertex_pose.h"
#include "utility/utility.h"

namespace myslam {
namespace backend {

/*    std::vector<std::shared_ptr<Vertex>> verticies_; // 该边对应的顶点
    VecX residual_;                 // 残差
    std::vector<MatXX> jacobians_;  // 雅可比，每个雅可比维度是 residual x
   vertex[i] MatXX information_;             // 信息矩阵 VecX observation_; //
   观测信息
    */

void EdgeReprojection::ComputeResidual() {
  const double inv_dep_i = verticies_[0]->Parameters()[0];

  const VecX& param_i = verticies_[1]->Parameters();
  const Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
  const Vec3& Pi = param_i.head<3>();

  const VecX& param_j = verticies_[2]->Parameters();
  const Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
  const Vec3 Pj = param_j.head<3>();

  const VecX& param_ext = verticies_[3]->Parameters();
  const Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
  const Vec3& tic = param_ext.head<3>();

  const Vec3 pts_camera_i = pts_i_ / inv_dep_i;
  const Vec3 pts_imu_i = qic * pts_camera_i + tic;
  const Vec3 pts_w = Qi * pts_imu_i + Pi;
  const Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
  const Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

  const double dep_j = pts_camera_j.z();
  residual_ = (pts_camera_j / dep_j).head<2>() -
              pts_j_.head<2>();  /// J^t * J * delta_x = - J^t * r
}

void EdgeReprojection::ComputeJacobians() {
  const double inv_dep_i = verticies_[0]->Parameters()[0];

  const VecX& param_i = verticies_[1]->Parameters();
  const Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
  const Vec3& Pi = param_i.head<3>();

  const VecX& param_j = verticies_[2]->Parameters();
  const Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
  const Vec3& Pj = param_j.head<3>();

  const VecX& param_ext = verticies_[3]->Parameters();
  const Qd qic(param_ext[6], param_ext[3], param_ext[4], param_ext[5]);
  const Vec3& tic = param_ext.head<3>();

  const Vec3 pts_camera_i = pts_i_ / inv_dep_i;
  const Vec3 pts_imu_i = qic * pts_camera_i + tic;
  const Vec3 pts_w = Qi * pts_imu_i + Pi;
  const Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
  const Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

  double dep_j = pts_camera_j.z();

  const Mat33 Ri = Qi.toRotationMatrix();
  const Mat33 Rj = Qj.toRotationMatrix();
  const Mat33 ric = qic.toRotationMatrix();
  Mat23 reduce(2, 3);
  reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j,
      -pts_camera_j(1) / (dep_j * dep_j);

  Eigen::Matrix<double, 2, 6> jacobian_pose_i;
  Eigen::Matrix<double, 3, 6> jaco_i;
  jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
  jaco_i.rightCols<3>() =
      ric.transpose() * Rj.transpose() * Ri * -Sophus::SO3d::hat(pts_imu_i);
  jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

  Eigen::Matrix<double, 2, 6> jacobian_pose_j;
  Eigen::Matrix<double, 3, 6> jaco_j;
  jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
  jaco_j.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_j);
  jacobian_pose_j.leftCols<6>() = reduce * jaco_j;

  Eigen::Vector2d jacobian_feature;
  jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric *
                     pts_i_ * -1.0 / (inv_dep_i * inv_dep_i);

  Eigen::Matrix<double, 2, 6> jacobian_ex_pose;
  Eigen::Matrix<double, 3, 6> jaco_ex;
  jaco_ex.leftCols<3>() =
      ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
  Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
  jaco_ex.rightCols<3>() =
      -tmp_r * Utility::skewSymmetric(pts_camera_i) +
      Utility::skewSymmetric(tmp_r * pts_camera_i) +
      Utility::skewSymmetric(ric.transpose() *
                             (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
  jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;

  jacobians_[0] = jacobian_feature;
  jacobians_[1] = jacobian_pose_i;
  jacobians_[2] = jacobian_pose_j;
  jacobians_[3] = jacobian_ex_pose;
}

void EdgeReprojectionXYZ::ComputeResidual() {
  const Vec3& pts_w = verticies_[0]->Parameters();

  const VecX& param_i = verticies_[1]->Parameters();
  const Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
  const Vec3& Pi = param_i.head<3>();

  const Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
  const Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);

  const double dep_i = pts_camera_i.z();
  residual_ = (pts_camera_i / dep_i).head<2>() - obs_.head<2>();
}

void EdgeReprojectionXYZ::SetTranslationImuFromCamera(Eigen::Quaterniond& qic_,
                                                      Vec3& tic_) {
  qic = qic_;
  tic = tic_;
}

void EdgeReprojectionXYZ::ComputeJacobians() {
  const Vec3& pts_w = verticies_[0]->Parameters();

  const VecX& param_i = verticies_[1]->Parameters();
  const Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
  const Vec3& Pi = param_i.head<3>();

  const Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
  const Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);

  const double dep_i = pts_camera_i.z();

  const Mat33 Ri = Qi.toRotationMatrix();
  const Mat33 ric = qic.toRotationMatrix();
  Mat23 reduce(2, 3);
  reduce << 1. / dep_i, 0, -pts_camera_i(0) / (dep_i * dep_i), 0, 1. / dep_i,
      -pts_camera_i(1) / (dep_i * dep_i);

  Eigen::Matrix<double, 2, 6> jacobian_pose_i;
  Eigen::Matrix<double, 3, 6> jaco_i;
  jaco_i.leftCols<3>() = ric.transpose() * -Ri.transpose();
  jaco_i.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_i);
  jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

  Eigen::Matrix<double, 2, 3> jacobian_feature;
  jacobian_feature = reduce * ric.transpose() * Ri.transpose();

  jacobians_[0] = jacobian_feature;
  jacobians_[1] = jacobian_pose_i;
}

void EdgeReprojectionPoseOnly::ComputeResidual() {
  const VecX& pose_params = verticies_[0]->Parameters();
  const Sophus::SE3d pose(
      Qd(pose_params[6], pose_params[3], pose_params[4], pose_params[5]),
      pose_params.head<3>());

  Vec3 pc = pose * landmark_world_;
  pc = pc / pc[2];
  const Vec2 pixel = (K_ * pc).head<2>() - observation_;
  residual_ = pixel;
}

void EdgeReprojectionPoseOnly::ComputeJacobians() {
  // TODO implement jacobian here
}

}  // namespace backend
}  // namespace myslam