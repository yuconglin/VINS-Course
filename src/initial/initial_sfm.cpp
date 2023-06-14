#include "initial/initial_sfm.h"

namespace GlobalSFM
{
	namespace
	{
		void triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1,
													const Vector2d &point0, const Vector2d &point1, Vector3d &point_3d)
		{
			Matrix4d design_matrix = Matrix4d::Zero();
			design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
			design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
			design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
			design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
			const Vector4d triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
			point_3d(0) = triangulated_point(0) / triangulated_point(3);
			point_3d(1) = triangulated_point(1) / triangulated_point(3);
			point_3d(2) = triangulated_point(2) / triangulated_point(3);
		}

		// Updates the feature points observed by frame0 and frame1.
		void triangulateTwoFrames(int frame0, const Eigen::Matrix<double, 3, 4> &Pose0,
															int frame1, const Eigen::Matrix<double, 3, 4> &Pose1,
															vector<SFMFeature> &sfm_f)
		{
			assert(frame0 != frame1);
			for (int j = 0; j < sfm_f.size(); j++)
			{
				if (sfm_f[j].state)
				{
					continue;
				}

				bool has_0 = false;
				bool has_1 = false;
				Vector2d point0;
				Vector2d point1;

				for (int k = 0; k < sfm_f[j].observation.size(); k++)
				{
					if (sfm_f[j].observation[k].first == frame0)
					{
						point0 = sfm_f[j].observation[k].second;
						has_0 = true;
					}

					if (sfm_f[j].observation[k].first == frame1)
					{
						point1 = sfm_f[j].observation[k].second;
						has_1 = true;
					}
				}

				if (has_0 && has_1)
				{
					Vector3d point_3d;
					triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
					sfm_f[j].state = true;
					sfm_f[j].position[0] = point_3d(0);
					sfm_f[j].position[1] = point_3d(1);
					sfm_f[j].position[2] = point_3d(2);
				}
			}
		}
	}

	// Updates {R, P}_initial by PnP with sfm_fs that obsovered by frame i.
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
											 const vector<SFMFeature> &sfm_f)
	{
		vector<cv::Point2f> pts_2_vector;
		vector<cv::Point3f> pts_3_vector;
		for (int j = 0; j < sfm_f.size(); j++)
		{
			if (!sfm_f[j].state)
			{
				continue;
			}
			Vector2d point2d;
			for (int k = 0; k < sfm_f[j].observation.size(); k++)
			{
				if (sfm_f[j].observation[k].first == i)
				{
					const Vector2d &img_pts = sfm_f[j].observation[k].second;
					pts_2_vector.emplace_back(img_pts.x(), img_pts.y());
					pts_3_vector.emplace_back(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
					break;
				}
			}
		}

		if (pts_2_vector.size() < 15)
		{
			printf("unstable features tracking, please slowly move you device!\n");
			if (pts_2_vector.size() < 10)
			{
				return false;
			}
		}

		cv::Mat r, rvec, t, D, tmp_r;
		cv::eigen2cv(R_initial, tmp_r);
		cv::Rodrigues(tmp_r, rvec);
		cv::eigen2cv(P_initial, t);
		cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

		if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
		{
			return false;
		}

		cv::Rodrigues(rvec, r);
		MatrixXd R_pnp;
		cv::cv2eigen(r, R_pnp);
		MatrixXd T_pnp;
		cv::cv2eigen(t, T_pnp);
		R_initial = R_pnp;
		P_initial = T_pnp;

		return true;
	}

	// q w_R_cam t w_R_cam
	// c_rotation cam_R_w
	// c_translation cam_R_w
	// relative_q[i][j]  j_q_i
	// relative_t[i][j]  j_t_ji  (j < i)
	bool construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
								 const Matrix3d &relative_R, const Vector3d &relative_T,
								 vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
	{
		// have relative_r relative_t
		// intial two view
		q[l].setIdentity();
		T[l].setZero();

		q[frame_num - 1] = q[l] * Quaterniond(relative_R);
		T[frame_num - 1] = relative_T;

		//rotate to cam frame
		Matrix3d c_Rotation[frame_num];
		Vector3d c_Translation[frame_num];
		Quaterniond c_Quat[frame_num];
		double c_rotation[frame_num][4];
		double c_translation[frame_num][3];
		Eigen::Matrix<double, 3, 4> Pose[frame_num];

		c_Quat[l] = q[l].inverse();
		c_Rotation[l] = c_Quat[l].toRotationMatrix();
		c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
		Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
		Pose[l].block<3, 1>(0, 3) = c_Translation[l];

		c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
		c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
		c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
		Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
		Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

		// Initialize poses i to frame_num - 1 via PnP.
		//1: trangulate between l ----- frame_num - 1
		//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
		for (int i = l; i < frame_num - 1; i++)
		{
			// solve pnp
			if (i > l)
			{
				Matrix3d R_initial = c_Rotation[i - 1];
				Vector3d P_initial = c_Translation[i - 1];

				// Updates {R,P}_initial by PnP with sfm_fs observed by frame i.
				if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				{
					return false;
				}

				c_Rotation[i] = R_initial;
				c_Translation[i] = P_initial;
				c_Quat[i] = c_Rotation[i];
				Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
				Pose[i].block<3, 1>(0, 3) = c_Translation[i];
			}

			// triangulate point based on the solve pnp result
			// Updates sfm_fs that are observed by frame i and frame {frame_num - 1}.
			triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
		}

		//3: triangulate l-----l+1 l+2 ... frame_num -2
		for (int i = l + 1; i < frame_num - 1; i++)
		{
			// Updates sfm_fs that are observed by frame l and frame i.
			triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
		}

		//4: solve pnp l-1; triangulate l-1 ----- l
		//             l-2              l-2 ----- l
		for (int i = l - 1; i >= 0; i--)
		{
			//solve pnp
			Matrix3d R_initial = c_Rotation[i + 1];
			Vector3d P_initial = c_Translation[i + 1];

			// Updates {R,P}_initial by PnP with sfm_fs observed by frame i.
			if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			{
				return false;
			}
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;

			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];

			//triangulate
			// Updates sfm_fs observed by frame i and l.
			triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
		}

		//5: triangulate all other points
		for (int j = 0; j < sfm_f.size(); j++)
		{
			if (sfm_f[j].state)
			{
				continue;
			}

			if (sfm_f[j].observation.size() >= 2)
			{
				const int frame_0 = sfm_f[j].observation[0].first;
				const Vector2d &point0 = sfm_f[j].observation[0].second;
				const int frame_1 = sfm_f[j].observation.back().first;
				const Vector2d &point1 = sfm_f[j].observation.back().second;

				// Updates sfm_f[j]'s position via triangulation.
				Vector3d point_3d;
				triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
				sfm_f[j].state = true;
				sfm_f[j].position[0] = point_3d(0);
				sfm_f[j].position[1] = point_3d(1);
				sfm_f[j].position[2] = point_3d(2);
			}
		}

		//full BA
		ceres::Problem problem;
		ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();

		for (int i = 0; i < frame_num; i++)
		{
			//double array for ceres
			c_translation[i][0] = c_Translation[i].x();
			c_translation[i][1] = c_Translation[i].y();
			c_translation[i][2] = c_Translation[i].z();
			c_rotation[i][0] = c_Quat[i].w();
			c_rotation[i][1] = c_Quat[i].x();
			c_rotation[i][2] = c_Quat[i].y();
			c_rotation[i][3] = c_Quat[i].z();
			problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
			problem.AddParameterBlock(c_translation[i], 3);
			if (i == l)
			{
				problem.SetParameterBlockConstant(c_rotation[i]);
			}
			if (i == l || i == frame_num - 1)
			{
				problem.SetParameterBlockConstant(c_translation[i]);
			}
		}

		for (int i = 0; i < sfm_f.size(); i++)
		{
			if (!sfm_f[i].state)
			{
				continue;
			}

			for (int j = 0; j < sfm_f[i].observation.size(); j++)
			{
				const int l = sfm_f[i].observation[j].first;
				ceres::CostFunction *cost_function = ReprojectionError3D::Create(
						sfm_f[i].observation[j].second.x(),
						sfm_f[i].observation[j].second.y());

				problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l],
																 sfm_f[i].position);
			}
		}

		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		//options.minimizer_progress_to_stdout = true;
		options.max_solver_time_in_seconds = 0.2;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		//std::cout << summary.BriefReport() << "\n";

		if (summary.termination_type != ceres::CONVERGENCE || summary.final_cost > 5e-03)
		{
			return false;
		}

		// Updates the rotation parts.
		for (int i = 0; i < frame_num; i++)
		{
			q[i].w() = c_rotation[i][0];
			q[i].x() = c_rotation[i][1];
			q[i].y() = c_rotation[i][2];
			q[i].z() = c_rotation[i][3];
			q[i] = q[i].inverse();
		}

		// Updates the translation parts.
		for (int i = 0; i < frame_num; i++)
		{
			T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		}

		// Updates the tracked feature points.
		for (int i = 0; i < sfm_f.size(); i++)
		{
			if (sfm_f[i].state)
			{
				sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
			}
		}

		return true;
	}
}
