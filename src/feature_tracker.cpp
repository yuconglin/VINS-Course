#include "feature_tracker.h"

#include <glog/logging.h>

namespace {
bool inBorder(const cv::Point2f &pt) {
  constexpr int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE &&
         BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

template <typename T>
void reduceVector(vector<T> &v, const vector<uchar> &status) {
  int j = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (status[i]) {
      v[j++] = v[i];
    }
  }
  v.resize(j);
}

}  // namespace

int FeatureTracker::n_id = 0;

void FeatureTracker::setMask() {
  if (FISHEYE) {
    mask = fisheye_mask.clone();
  } else {
    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
  }

  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

  for (unsigned int i = 0; i < forw_pts.size(); i++) {
    cnt_pts_id.emplace_back(track_cnt[i], make_pair(forw_pts[i], ids[i]));
  }

  sort(cnt_pts_id.begin(), cnt_pts_id.end(),
       [](const auto &a, const auto &b) { return a.first > b.first; });

  forw_pts.clear();
  ids.clear();
  track_cnt.clear();

  for (const auto &[track_count, point_ids] : cnt_pts_id) {
    if (mask.at<uchar>(point_ids.first) == 255) {
      forw_pts.push_back(point_ids.first);
      ids.push_back(point_ids.second);
      track_cnt.push_back(track_count);
      cv::circle(mask, point_ids.first, MIN_DIST, 0, -1);
    }
  }
}

void FeatureTracker::addPoints() {
  for (const auto &p : n_pts) {
    forw_pts.push_back(p);
    ids.push_back(-1);
    track_cnt.push_back(1);
  }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time) {
  cv::Mat img;
  TicToc t_r;
  cur_time = _cur_time;

  // if image is too dark or light, trun on equalize to find enough features
  if (EQUALIZE) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    TicToc t_c;
    clahe->apply(_img, img);
  } else {
    img = _img;
  }

  if (forw_img.empty()) {
    prev_img = cur_img = forw_img = img;
  } else {
    forw_img = img;
  }

  forw_pts.clear();

  if (cur_pts.size() > 0) {
    TicToc t_o;
    vector<uchar> status;
    vector<float> err;

    // forw_img: next image.
    // forw_pts: next points.
    cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err,
                             cv::Size(21, 21), 3);

    for (int i = 0; i < int(forw_pts.size()); i++) {
      if (status[i] && !inBorder(forw_pts[i])) {
        // invalid points.
        status[i] = 0;
      }
    }

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
    // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
  }

  for (auto &n : track_cnt) {
    n++;
  }

  if (PUB_THIS_FRAME) {
    // Remove outlier feature points in Fundamental Matrix's RANSCAC based
    // estimation between successive frames.
    rejectWithF();

    // ROS_DEBUG("set mask begins");
    TicToc t_m;

    // It won't affect pinhole cameras.
    setMask();
    // ROS_DEBUG("set mask costs %fms", t_m.toc());

    // ROS_DEBUG("detect feature begins");
    TicToc t_t;
    const int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0) {
      if (mask.empty()) {
        LOG(INFO) << "mask is empty " ;
      }
      if (mask.type() != CV_8UC1) {
        LOG(INFO) << "mask type wrong " ;
      }
      if (mask.size() != forw_img.size()) {
        LOG(INFO) << "wrong size " ;
      }

      // The function finds the most prominent corners in the image.
      cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01,
                              MIN_DIST, mask);
    } else {
      n_pts.clear();
      // ROS_DEBUG("detect feature costs: %fms", t_t.toc());
    }

    // ROS_DEBUG("add feature begins");
    TicToc t_a;
    addPoints();
    // ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
  }

  prev_img = cur_img;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  cur_img = forw_img;
  cur_pts = forw_pts;

  undistortedPoints();

  prev_time = cur_time;
}

// Reject feature points with Fundamental Matrix computation.
void FeatureTracker::rejectWithF() {
  if (forw_pts.size() >= 8) {
    // ROS_DEBUG("FM ransac begins");
    TicToc t_f;

    vector<cv::Point2f> un_cur_pts(cur_pts.size());
    vector<cv::Point2f> un_forw_pts(forw_pts.size());

    // The following loop obtains undistored normalized coorindates for points.
    // reference: https://blog.csdn.net/hltt3838/article/details/119428558
    // reference: https://zhuanlan.zhihu.com/p/51395805
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
      Eigen::Vector3d tmp_p;
      // Lifts the points from the image plane to the projective space: 2D-->3D.
      m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y),
                               tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y),
                               tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD,
                           0.99, status);

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);

    // int size_a = cur_pts.size();
    // ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 *
    // forw_pts.size() / size_a); ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
  }
}

bool FeatureTracker::updateID(unsigned int i) {
  if (i < ids.size()) {
    if (ids[i] == -1) {
      ids[i] = n_id++;
    }
    return true;
  } else {
    return false;
  }
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file) {
  LOG(INFO) << "reading paramerter of camera " << calib_file ;
  m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name) {
  cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;

  for (int i = 0; i < COL; i++) {
    for (int j = 0; j < ROW; j++) {
      const Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
    }
  }

  for (size_t i = 0; i < undistortedp.size(); i++) {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
    pp.at<float>(2, 0) = 1.0;

    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 &&
        pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600) {
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300,
                               pp.at<float>(0, 0) + 300) =
          cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
    } else {
      // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x,
      // pp.at<float>(1, 0), pp.at<float>(0, 0));
    }
  }
  cv::imshow(name, undistortedImg);
  cv::waitKey(0);
}

// Undistorts feature points.
void FeatureTracker::undistortedPoints() {
  cur_un_pts.clear();
  cur_un_pts_map.clear();

  for (unsigned int i = 0; i < cur_pts.size(); i++) {
    const Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
    Eigen::Vector3d b;
    m_camera->liftProjective(a, b);
    cur_un_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
    cur_un_pts_map.emplace(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z()));
  }

  // caculate points velocity
  if (!prev_un_pts_map.empty()) {
    const double dt = cur_time - prev_time;
    pts_velocity.clear();

    for (unsigned int i = 0; i < cur_un_pts.size(); i++) {
      if (ids[i] != -1) {
        const auto it = prev_un_pts_map.find(ids[i]);
        if (it != prev_un_pts_map.end()) {
          const double v_x = (cur_un_pts[i].x - it->second.x) / dt;
          const double v_y = (cur_un_pts[i].y - it->second.y) / dt;
          pts_velocity.emplace_back(v_x, v_y);
        } else {
          pts_velocity.emplace_back(0, 0);
        }
      } else {
        pts_velocity.emplace_back(0, 0);
      }
    }
  } else {
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
      pts_velocity.emplace_back(0, 0);
    }
  }
  prev_un_pts_map = cur_un_pts_map;
}
