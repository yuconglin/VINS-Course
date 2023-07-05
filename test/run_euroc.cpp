
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <eigen3/Eigen/Dense>
#include <functional>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

constexpr int nDelayTimes = 2;
string sData_path;
string sConfig_path;

// ./bin/run_euroc
// --data_path=/home/yuconglin/slam_data/vins_data/MH_05_difficult/mav0/

std::shared_ptr<System> pSystem;

string GetDatasetName(const string& full_path) {
  vector<string> words;
  stringstream str(full_path);
  string tmp;

  while (getline(str, tmp, '/')) {
    words.push_back(tmp);
  }
  assert(words.size() > 2);

  return words[words.size() - 2];
}

void ReadCsv(const string& file_path,
             function<void(const vector<string>&)> process_function) {
  fstream file(file_path);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open file " << file_path << '\n';
    return;
  }

  vector<string> row;
  string line;
  string word;
  bool got_data = false;

  while (getline(file, line)) {
    row.clear();

    stringstream str(line);
    while (getline(str, word, ',')) {
      row.push_back(word);
    }
    if (!got_data) {
      // Deals with title row.
      got_data = true;
    } else {
      // Deals with data rows.
      assert(!row.empty());
      // Deals with '\n' at the end of each line.
      row.back().pop_back();
      process_function(row);
    }
  }
  file.close();
}

void PubImuData() {
  const string sImu_data_file = sData_path + "imu0/data.csv";
  LOG(INFO) << "Read Imu from " << sImu_data_file << '\n';

  const auto process_imu = [&](const vector<string>& row) {
    const double dStampNSec = stod(row[0]);
    const Vector3d vGyr(stod(row[1]), stod(row[2]), stod(row[3]));
    const Vector3d vAcc(stod(row[4]), stod(row[5]), stod(row[6]));
    pSystem->PubImuData(dStampNSec / 1e9, vGyr, vAcc);
    usleep(5000 * nDelayTimes);
  };

  ReadCsv(sImu_data_file, process_imu);
}

void PubImageData() {
  const string sImage_file = sData_path + "cam0/data.csv";
  LOG(INFO) << "Read image from " << sImage_file << '\n';

  const auto process_image = [&](const vector<string>& row) {
    // Deals with data rows.
    const double dStampNSec = stod(row[0]);
    const string& sImgFileName = row[1];
    const string imagePath = sData_path + "cam0/data/" + sImgFileName;

    Mat img = imread(imagePath.c_str(), 0);
    if (img.empty()) {
      LOG(ERROR) << "image is empty! Path: " << imagePath << endl;
      return;
    }

    pSystem->PubImageData(dStampNSec / 1e9, img);
    usleep(50000 * nDelayTimes);
  };

  ReadCsv(sImage_file, process_image);
}

#ifdef __APPLE__
// support for MacOS
void DrawIMGandGLinMainThrd() {
  string sImage_file = sConfig_path + "MH_05_cam0.txt";

  LOG(INFO) << "1 PubImageData start sImage_file: " << sImage_file << endl;

  ifstream fsImage;
  fsImage.open(sImage_file.c_str());
  if (!fsImage.is_open()) {
    LOG(ERROR) << "Failed to open image file! " << sImage_file << endl;
    return;
  }

  std::string sImage_line;
  double dStampNSec;
  string sImgFileName;

  pSystem->InitDrawGL();
  while (std::getline(fsImage, sImage_line) && !sImage_line.empty()) {
    std::istringstream ssImuData(sImage_line);
    ssImuData >> dStampNSec >> sImgFileName;
    // LOG(INFO) << "Image t : " << fixed << dStampNSec << " Name: " <<
    // sImgFileName
    // << endl;
    string imagePath = sData_path + "cam0/data/" + sImgFileName;

    Mat img = imread(imagePath.c_str(), 0);
    if (img.empty()) {
      LOG(ERROR) << "image is empty! path: " << imagePath << endl;
      return;
    }
    // pSystem->PubImageData(dStampNSec / 1e9, img);
    cv::Mat show_img;
    cv::cvtColor(img, show_img, CV_GRAY2RGB);
    if (SHOW_TRACK) {
      for (unsigned int j = 0; j < pSystem->trackerData[0].cur_pts.size();
           j++) {
        double len =
            min(1.0, 1.0 * pSystem->trackerData[0].track_cnt[j] / WINDOW_SIZE);
        cv::circle(show_img, pSystem->trackerData[0].cur_pts[j], 2,
                   cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
      }

      cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
      cv::imshow("IMAGE", show_img);
      // cv::waitKey(1);
    }

    pSystem->DrawGLFrame();
    usleep(50000 * nDelayTimes);
  }
  fsImage.close();
}
#endif

DEFINE_string(data_path, "~/slam_dataset/EuRoc/MH_05_difficult/mav0/",
              "Path to the data folder.");
DEFINE_string(config_path, "config/", "Path to the config folder.");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  sData_path = FLAGS_data_path;
  sConfig_path = FLAGS_config_path;

  pSystem.reset(new System(sConfig_path, GetDatasetName(sData_path)));

  std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);

  // sleep(5);
  std::thread thd_PubImuData(PubImuData);

  std::thread thd_PubImageData(PubImageData);

#ifdef __linux__
  std::thread thd_Draw(&System::Draw, pSystem);
#elif __APPLE__
  DrawIMGandGLinMainThrd();
#endif

  thd_PubImuData.join();
  thd_PubImageData.join();

  // thd_BackEnd.join();
#ifdef __linux__
  thd_Draw.join();
#endif

  LOG(INFO) << "main end... see you ..." << endl;
  return 0;
}
