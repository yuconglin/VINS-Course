#pragma once

#include <pangolin/pangolin.h>
#include <stdio.h>

#include <condition_variable>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

#include "estimator.h"
#include "feature_tracker.h"
#include "parameters.h"

// imu for vio
struct IMU_MSG {
  double header;
  Eigen::Vector3d linear_acceleration;
  Eigen::Vector3d angular_velocity;
};

using ImuConstPtr = std::shared_ptr<IMU_MSG const>;

// image for vio
struct IMG_MSG {
  double header;
  vector<Vector3d> points;
  vector<int> id_of_point;
  vector<float> u_of_point;
  vector<float> v_of_point;
  vector<float> velocity_x_of_point;
  vector<float> velocity_y_of_point;
};

using ImgConstPtr = std::shared_ptr<IMG_MSG const>;

class System {
 public:
  System(const std::string &sConfig_files, const std::string &dataset_name);

  ~System();

  void PubImageData(double dStampSec, cv::Mat &img);

  void PubImuData(double dStampSec, const Eigen::Vector3d &vGyr,
                  const Eigen::Vector3d &vAcc);

  // thread: visual-inertial odometry
  void ProcessBackEnd();
  void Draw();

  pangolin::OpenGlRenderState s_cam;
  pangolin::View d_cam;

  FeatureTracker trackerData[NUM_OF_CAM];

#ifdef __APPLE__
  void InitDrawGL();
  void DrawGLFrame();
#endif

 private:
  std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>>
  getMeasurements();

  // feature tracker
  std::vector<uchar> r_status;
  std::vector<float> r_err;

  double first_image_time;
  int pub_count = 1;
  bool first_image_flag = true;
  double last_image_time = 0;
  bool init_pub = 0;

  // estimator
  Estimator estimator;

  std::condition_variable con;
  double current_time = -1;
  std::queue<ImuConstPtr> imu_buf;
  std::queue<ImgConstPtr> feature_buf;

  int sum_of_wait = 0;

  std::mutex m_buf;
  std::mutex m_state;
  std::mutex i_buf;
  std::mutex m_estimator;

  double latest_time;
  Eigen::Vector3d tmp_P;
  Eigen::Quaterniond tmp_Q;
  Eigen::Vector3d tmp_V;
  Eigen::Vector3d tmp_Ba;
  Eigen::Vector3d tmp_Bg;
  Eigen::Vector3d acc_0;
  Eigen::Vector3d gyr_0;
  bool init_feature = 0;
  bool init_imu = 1;
  double last_imu_t = 0;
  std::ofstream ofs_pose;
  std::vector<Eigen::Vector3d> vPath_to_draw;
  bool bStart_backend;

  double total_image_proc_time_ = 0.0;
  int proc_count_ = 0;
};
