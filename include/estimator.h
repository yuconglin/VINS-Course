#pragma once

#include <queue>
#include <unordered_map>

#include "backend/problem.h"
#include "factor/integration_base.h"
#include "feature_manager.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include "initial/initial_sfm.h"
#include "initial/solve_5pts.h"
#include "opencv2/core/eigen.hpp"
#include "parameters.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"

class Estimator {
 public:
  Estimator();

  void setParameter();

  // interface
  void processIMU(double t, const Vector3d &linear_acceleration,
                  const Vector3d &angular_velocity);

  void processImage(
      const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
      double header);

  // internal
  void clearState();
  bool initialStructure();
  bool visualInitialAlign();
  bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
  void slideWindow();
  void solveOdometry();
  void slideWindowNew();
  void slideWindowOld();
  void backendOptimization();

  void problemSolve();
  void MargOldFrame();
  void MargNewFrame();

  void vector2double();
  void double2vector();
  bool failureDetection();

  enum SolverFlag { INITIAL, NON_LINEAR };

  enum MarginalizationFlag { MARGIN_OLD = 0, MARGIN_SECOND_NEW = 1 };
  //////////////// OUR SOLVER ///////////////////
  MatXX Hprior_;
  VecX bprior_;
  VecX errprior_;
  MatXX Jprior_inv_;

  Eigen::Matrix2d project_sqrt_info_;
  //////////////// OUR SOLVER //////////////////
  SolverFlag solver_flag;
  MarginalizationFlag marginalization_flag;
  Vector3d g;
  MatrixXd Ap[2], backup_A;
  VectorXd bp[2], backup_b;

  Matrix3d ric[NUM_OF_CAM];
  Vector3d tic[NUM_OF_CAM];

  Vector3d Ps[(WINDOW_SIZE + 1)];
  Vector3d Vs[(WINDOW_SIZE + 1)];
  Matrix3d Rs[(WINDOW_SIZE + 1)];
  Vector3d Bas[(WINDOW_SIZE + 1)];
  Vector3d Bgs[(WINDOW_SIZE + 1)];
  double td;

  Matrix3d back_R0, last_R, last_R0;
  Vector3d back_P0, last_P, last_P0;
  double Headers[(WINDOW_SIZE + 1)];

  IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
  Vector3d acc_0, gyr_0;

  vector<double> dt_buf[(WINDOW_SIZE + 1)];
  vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
  vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

  int frame_count;
  int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

  FeatureManager f_manager;
  InitialEXRotation initial_ex_rotation;

  bool first_imu;
  bool is_valid, is_key;
  bool failure_occur;

  vector<Vector3d> point_cloud;
  vector<Vector3d> margin_cloud;
  vector<Vector3d> key_poses;
  double initial_timestamp;

  double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
  double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
  double para_Feature[NUM_OF_F][SIZE_FEATURE];
  double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
  double para_Retrive_Pose[SIZE_POSE];
  double para_Td[1][1];
  double para_Tr[1][1];

  int loop_window_index;

  // MarginalizationInfo *last_marginalization_info;
  vector<double *> last_marginalization_parameter_blocks;
  // timestamp v.s. image frame.
  map<double, ImageFrame> all_image_frame;
  IntegrationBase *tmp_pre_integration;

  // relocalization variable
  bool relocalization_info;
  double relo_frame_stamp;
  double relo_frame_index;
  int relo_frame_local_index;
  vector<Vector3d> match_points;
  double relo_Pose[SIZE_POSE];
  Matrix3d drift_correct_r;
  Vector3d drift_correct_t;
  Vector3d prev_relo_t;
  Matrix3d prev_relo_r;
  Vector3d relo_relative_t;
  Quaterniond relo_relative_q;
  double relo_relative_yaw;
};
