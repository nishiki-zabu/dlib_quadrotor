#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <opencv2/opencv.hpp>
#include "render_face.hpp"
#define OPENCV_FACE_RENDER

using namespace dlib;
using namespace std;

#include <iostream>
#include <opencv2/core/core.hpp>

#include <geometry_msgs/Twist.h>

#define FACE_DOWNSAMPLE_RATIO 4
#define SKIP_FRAMES 2
#define OPENCV_FACE_RENDER

std::vector<cv::Mat>XYZ_split;
std::vector<double>dataXYZ;

double average_count=0;
int average_frames=6;
//int average_frames=4;

int median_limit=average_frames/2;
int lower_limit=average_frames/3;
int higher_limit=(2*average_frames)/3;

std::deque<double>discrete_deque_x;
std::deque<double>discrete_deque_y;
std::deque<double>discrete_deque_z;

std::deque<double>median_x;
std::deque<double>median_y;
std::deque<double>median_z;

std::deque<double>median_sort_x;
std::deque<double>median_sort_y;
std::deque<double>median_sort_z;

std::deque<double>Med_difference_x;
std::deque<double>Med_difference_y;
std::deque<double>Med_difference_z;

std::deque<double>Med_difference_sort_x;
std::deque<double>Med_difference_sort_y;
std::deque<double>Med_difference_sort_z;

double MAD_x=0;
double MAD_y=0;
double MAD_z=0;

double MADN_x=0;
double MADN_y=0;
double MADN_z=0;

std::deque<double>deviation_value_x;
std::deque<double>deviation_value_y;
std::deque<double>deviation_value_z;

double deviation_count_x=0;
double deviation_count_y=0;
double deviation_count_z=0;

double deviation_limit_x=10;
double deviation_limit_y=10;
double deviation_limit_z=10;

std::deque<double>average_angle_x_deque;
std::deque<double>average_angle_y_deque;
std::deque<double>average_angle_z_deque;

double average_angle_x;
double average_angle_y;
double average_angle_z;

double average_angle_count=0;

double numeric_rotation_x=0;
double numeric_rotation_y=0;
double numeric_rotation_z=0;

std::vector<double>rotation_x;
std::vector<double>rotation_y;
std::vector<double>rotation_z;

std::vector<double>drone_rotation_y;
	
std::vector<cv::Point2d> image_points;
std::vector<cv::Point2d> face_points;
std::vector<cv::Point3d> nose_end_point3D;
std::vector<cv::Point2d> nose_end_point2D;

/*
std::vector<cv::Point3d> get_3d_model_points()
{
	std::vector<cv::Point3d> modelPoints;

	modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));
	modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
	modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));
	modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));
	modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));
	modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));    

	return modelPoints;   
}
*/

std::vector<cv::Point3d> get_3d_model_points()
{
  std::vector<cv::Point3d> modelPoints;

  modelPoints.push_back(cv::Point3d(-228.93,92.14,-325.8)); //0,1,16,15
  modelPoints.push_back(cv::Point3d(-224.79,41.24,-318.69));
  modelPoints.push_back(cv::Point3d(228.93,92.14,-325.8));
  modelPoints.push_back(cv::Point3d(224.79,41.24,-318.69));

  modelPoints.push_back(cv::Point3d(-66.9,99.94,-85.72)); //39,40,42,47
  modelPoints.push_back(cv::Point3d(-89.65,96.74,-75.57));  
  modelPoints.push_back(cv::Point3d(66.9,99.94,-85.72));
  modelPoints.push_back(cv::Point3d(89.65,96.74,-75.57)); 

  modelPoints.push_back(cv::Point3d(0,107.43,-41.93)); //27,28,29,30
  modelPoints.push_back(cv::Point3d(0,71.79,-34.67));  
  modelPoints.push_back(cv::Point3d(0,37.77,-15.83));
  modelPoints.push_back(cv::Point3d(0,0,0));

  modelPoints.push_back(cv::Point3d(-40.61,-29.94,-52.14)); //31,32,33,34,35
  modelPoints.push_back(cv::Point3d(-22.95,-32.59,-46.07));  
  modelPoints.push_back(cv::Point3d(0,-38.54,-42.36));
  modelPoints.push_back(cv::Point3d(22.95,-32.59,-46.07));
  modelPoints.push_back(cv::Point3d(40.61,-29.94,-52.14));

  return modelPoints;   
}

/*
std::vector<cv::Point2d> get_2d_image_points(full_object_detection &d)
{
  std::vector<cv::Point2d> image_points;

  image_points.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );　// Nose tip
  image_points.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );   // Chin
  image_points.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) ); // Left eye left corner
  image_points.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) ); // Right eye right corner
  image_points.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) ); // Left Mouth corner
  image_points.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) ); // Right mouth corner

  return image_points;
}
*/

std::vector<cv::Point2d> get_2d_image_points(full_object_detection &d)
{
  std::vector<cv::Point2d> image_points;

  image_points.push_back( cv::Point2d( d.part(0).x(), d.part(0).y() ) );
  image_points.push_back( cv::Point2d( d.part(1).x(), d.part(1).y() ) );
  image_points.push_back( cv::Point2d( d.part(16).x(), d.part(16).y() ) ); 
  image_points.push_back( cv::Point2d( d.part(15).x(), d.part(15).y() ) );

  image_points.push_back( cv::Point2d( d.part(39).x(), d.part(39).y() ) ); 
  image_points.push_back( cv::Point2d( d.part(40).x(), d.part(40).y() ) ); 
  image_points.push_back( cv::Point2d( d.part(42).x(), d.part(42).y() ) );
  image_points.push_back( cv::Point2d( d.part(47).x(), d.part(47).y() ) );

  image_points.push_back( cv::Point2d( d.part(27).x(), d.part(27).y() ) ); 
  image_points.push_back( cv::Point2d( d.part(28).x(), d.part(28).y() ) );
  image_points.push_back( cv::Point2d( d.part(29).x(), d.part(29).y() ) ); 
  image_points.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );
 
  image_points.push_back( cv::Point2d( d.part(31).x(), d.part(31).y() ) );
  image_points.push_back( cv::Point2d( d.part(32).x(), d.part(32).y() ) );
  image_points.push_back( cv::Point2d( d.part(33).x(), d.part(33).y() ) ); 
  image_points.push_back( cv::Point2d( d.part(34).x(), d.part(34).y() ) );
  image_points.push_back( cv::Point2d( d.part(35).x(), d.part(35).y() ) ); 

  return image_points;
}

std::vector<cv::Point2d> get_2d_face_points(full_object_detection &d)
{
	std::vector<cv::Point2d> face_points;
	for(int i=0;i<68;i++)
	{
		face_points.push_back( cv::Point2d( d.part(i).x(), d.part(i).y() ) );
	} 
	return face_points;
}

cv::Mat get_camera_matrix(float focal_length,cv::Point2d center)
{
	cv::Mat camera_matrix=(cv::Mat_<double>(3,3)<<focal_length,0,center.x,0,focal_length,center.y,0,0,1);
	return camera_matrix;
}

void printData(const std::vector<double> &data)
{
	std::ofstream ofs("print.txt");
	for (int i=0;i<data.size();i++)
	{
		ofs<<data[i]<<" ";
	}
}

const double Kcp=-0.5;
//const double Kci=-0.000004;
const double Kci=-0.000005;
//const double Kcd=-0.0;
const double Kcd=-25;
//const double Kc_gain=38.8051;
//const double Kc_gain=40;
const double Kc_gain=40*1.0;

double PID_camera_controller(double _input_value,double _target)
{

	//誤差
	double error=2*(_input_value/2-_target)/_input_value;

	//時間
	static int last_t=0.0;
	double dt=(cv::getTickCount()-last_t)/cv::getTickFrequency();
	last_t=cv::getTickCount();

	//積分項
	static double integral=0.0;
	if(dt>0.1)
	{
		//リセット
		integral=0.0;
	}

	integral+=(error*dt);

	//微分項
	static double previous_error=0.0;
	if(dt>0.1)
	{
	//リセット
	previous_error=0.0;
	}

	double derivative=(error-previous_error)/dt;

	previous_error=error;

	//操作量
	double _output_value=Kcp*error+Kci*integral+Kcd*derivative;
	//double _output_value=-Kcp*error;

	return _output_value;

}

double camera_pan_error=0;
double pre_camera_pan_error=0;

const double Kdp=1.0;
const double Kdi=0.0;
const double Kdd=0.0;
const double Kd_gain=0.1;

double PID_drone_controller(double _input_value)
{

	//誤差
	double error=_input_value/M_PI;

	//時間
	static int last_t=0.0;
	double dt=(cv::getTickCount()-last_t)/cv::getTickFrequency();
	last_t=cv::getTickCount();

	//積分項
	static double integral=0.0;
	if(dt>0.1)
	{
		//リセット
		integral=0.0;
	}

	integral+=(error*dt);

	//微分項
	static double previous_error=0.0;
	if(dt>0.1)
	{
	//リセット
	previous_error=0.0;
	}

	double derivative=(error-previous_error)/dt;

	previous_error=error;

	//操作量
	double _output_value=Kdp*error+Kdi*integral+Kdd*derivative;

	return _output_value;

}

double drone_vy;

geometry_msgs::Twist camera_control;
geometry_msgs::Twist cmd_vel;

int pre_count=0;

double count_checker()
{
  pre_count++;
  return pre_count;
}

shape_predictor pose_model;
image_window win;

void head_estimator(cv_bridge::CvImagePtr _image_mat)
{
	try
	{

		if(pre_count<1)
		{
    	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		}
		
		// Load face detection and pose estimation models.
		frontal_face_detector detector = get_frontal_face_detector();
		// Detect faces
		cv_image<bgr_pixel> cimg(_image_mat->image);
		std::vector<rectangle> faces = detector(cimg);
		
		// Find the pose of each face.
		std::vector<full_object_detection> shapes;
		for (unsigned long i = 0; i < faces.size(); ++i)
			shapes.push_back(pose_model(cimg, faces[i]));
		


		// Pose estimation
		std::vector<cv::Point3d> model_points = get_3d_model_points();
		

		
		for (unsigned long i = 0; i < faces.size(); i++)
		{
			rectangle r(
				(long)(faces[i].left()),
				(long)(faces[i].top()),
				(long)(faces[i].right()),
				(long)(faces[i].bottom())
			);
			
			full_object_detection shape = pose_model(cimg, r);
			shapes.push_back(shape);
			
			image_points = get_2d_image_points(shape);
			face_points = get_2d_face_points(shape);

			pre_camera_pan_error=camera_pan_error;
			camera_pan_error=Kc_gain*PID_camera_controller(_image_mat->image.cols,face_points[27].x);
			//cout << camera_pan_error << endl;

			double focal_length = _image_mat->image.cols;	
			cv::Mat camera_matrix = get_camera_matrix(focal_length, cv::Point2d(_image_mat->image.cols/2, _image_mat->image.rows/2));
			cv::Mat rotation_vector;
			cv::Mat rotation_matrix;
			cv::Mat translation_vector;
				
			cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);
				
			//cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
			cv::solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

			//cv::Rodrigues(rotation_vector, rotation_matrix);

			nose_end_point3D.push_back(cv::Point3d(0, 0, 1000));

			cv::projectPoints(nose_end_point3D,rotation_vector,translation_vector,camera_matrix,dist_coeffs,nose_end_point2D);

			cv::line(_image_mat->image, image_points[11], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);

			cv::split(rotation_vector, XYZ_split);
			XYZ_split[0].reshape(0, 1).copyTo(dataXYZ);

			if (average_count > average_frames)
			{

				discrete_deque_x.push_back(dataXYZ[0]);
				discrete_deque_y.push_back(dataXYZ[1]);
				discrete_deque_z.push_back(dataXYZ[2]);

				discrete_deque_x.pop_front();
				discrete_deque_y.pop_front();
				discrete_deque_z.pop_front();

				median_x = discrete_deque_x;
				median_y = discrete_deque_y;
				median_z = discrete_deque_z;

				median_sort_x = discrete_deque_x;
				median_sort_y = discrete_deque_y;
				median_sort_z = discrete_deque_z;

				std::sort(median_sort_x.begin(), median_sort_x.end());
				std::sort(median_sort_y.begin(), median_sort_y.end());
				std::sort(median_sort_z.begin(), median_sort_z.end());

				for (int i=0; i<=average_frames; i++)
				{

					Med_difference_x.push_back(std::abs(median_sort_x[i]-median_sort_x[median_limit]));
					Med_difference_y.push_back(std::abs(median_sort_y[i]-median_sort_y[median_limit]));
					Med_difference_z.push_back(std::abs(median_sort_z[i]-median_sort_z[median_limit]));
				}

				Med_difference_sort_x=Med_difference_x;
				Med_difference_sort_y=Med_difference_y;
				Med_difference_sort_z=Med_difference_z;

				std::sort(Med_difference_sort_x.begin(),Med_difference_sort_x.end());
				std::sort(Med_difference_sort_y.begin(),Med_difference_sort_y.end());
				std::sort(Med_difference_sort_z.begin(),Med_difference_sort_z.end());

				MAD_x=Med_difference_sort_x[median_limit];
				MAD_y=Med_difference_sort_y[median_limit];
				MAD_z=Med_difference_sort_z[median_limit];	

				MADN_x=MAD_x/0.675;
				MADN_y=MAD_y/0.675;
				MADN_z=MAD_z/0.675;

				for(int i=0;i<=average_frames;i++)
				{
					deviation_value_x.push_back((10*(median_x[i]-median_sort_x[median_limit]))/MADN_x + 50);  
					deviation_value_y.push_back((10*(median_y[i]-median_sort_y[median_limit]))/MADN_x + 50);
					deviation_value_z.push_back((10*(median_z[i]-median_sort_z[median_limit]))/MADN_x + 50);
				}

				average_angle_x=0;
				average_angle_y=0;
				average_angle_z=0;

				deviation_count_x=0;
				deviation_count_y=0;
				deviation_count_z=0;

				for(int i=0;i<=average_frames;i++)
				{
					if(std::abs(deviation_value_x[i]-50)<deviation_limit_x)
					{
						average_angle_x+=median_x[i];
						deviation_count_x++;
					}
	
					if(std::abs(deviation_value_y[i]-50)<deviation_limit_y)
					{
						average_angle_y+=median_y[i];
						deviation_count_y++;;
					}

					if(std::abs(deviation_value_z[i]-50)<deviation_limit_z)
					{
						average_angle_z+=median_z[i];
						deviation_count_z++;
					}

				}

				average_angle_x/=deviation_count_x;
				average_angle_y/=deviation_count_y;
				average_angle_z/=deviation_count_z;

				median_x.clear();
				median_y.clear();
				median_z.clear();

				median_sort_x.clear();
				median_sort_y.clear();
				median_sort_z.clear();

				Med_difference_x.clear();
				Med_difference_y.clear();
				Med_difference_z.clear();

				Med_difference_sort_x.clear();
				Med_difference_sort_y.clear();
				Med_difference_sort_z.clear();

				deviation_value_x.clear();
				deviation_value_y.clear();
				deviation_value_z.clear();

				if(average_angle_count>2)
				{

					average_angle_x_deque.push_back(average_angle_x);
					average_angle_x_deque.pop_front();

					average_angle_y_deque.push_back(average_angle_y);
					average_angle_y_deque.pop_front();

					average_angle_z_deque.push_back(average_angle_z);
					average_angle_z_deque.pop_front();

					numeric_rotation_x=0;
					numeric_rotation_y=0;
					numeric_rotation_z=0;

					for(int i=0;i<average_angle_count;i++)
					{

						numeric_rotation_x+=average_angle_x_deque[i];
						numeric_rotation_y+=average_angle_y_deque[i];
						numeric_rotation_z+=average_angle_z_deque[i];

					}

					numeric_rotation_x/=average_angle_count;
					numeric_rotation_y/=average_angle_count;
					numeric_rotation_z/=average_angle_count;

					drone_vy=Kd_gain*PID_drone_controller(numeric_rotation_y); 

					rotation_x.push_back(numeric_rotation_x);
					rotation_y.push_back(numeric_rotation_y);
					rotation_z.push_back(numeric_rotation_z);

					drone_rotation_y.push_back(numeric_rotation_y);

				}else{

					average_angle_x_deque.push_back(average_angle_x); 
					average_angle_y_deque.push_back(average_angle_y);
					average_angle_z_deque.push_back(average_angle_z);        

					average_angle_count++;

				}

			}else{

				discrete_deque_x.push_back(dataXYZ[0]);
				discrete_deque_y.push_back(dataXYZ[1]);
				discrete_deque_z.push_back(dataXYZ[2]);

				average_count++;

			}

/*
				double mouth_rx = face_points[54].x - face_points[66].x;
				double mouth_ry = face_points[54].y - face_points[66].y;
				double mouth_lx = face_points[48].x - face_points[66].x;
				double mouth_ly = face_points[48].y - face_points[66].y;

				double cos_smile = (mouth_rx*mouth_lx + mouth_ry*mouth_ly) / (sqrt(mouth_rx*mouth_rx + mouth_ry*mouth_ry)*sqrt(mouth_lx*mouth_lx + mouth_ly*mouth_ly));

				double open_hx = face_points[54].x - face_points[48].x;
				double open_hy = face_points[54].y - face_points[48].y;
				double open_vx = face_points[62].x - face_points[66].x;
				double open_vy = face_points[62].y - face_points[66].y;

				double surprised = sqrt(open_vx*open_vx + open_vy*open_vy) / (sqrt(open_hx*open_hx + open_hy*open_hy));

				if (cos_smile > -0.7&&surprised < 0.4&&-0.5 < average_test_y&&average_test_y < 0.5)
				{
					cv::putText(im, cv::format("smile"), cv::Point(face_points[19].x, (face_points[21].y + face_points[22].y) / 2), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
				}

				if (surprised > 0.4&&-0.5 < average_test_y&&average_test_y < 0.5)
				{
					cv::putText(im, cv::format("surprised"), cv::Point(face_points[19].x, (face_points[21].y + face_points[22].y) / 2), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
				}

				double eyebrows_x = face_points[21].x - face_points[22].x;
				double eyebrows_y = face_points[21].y - face_points[22].y;
				double eyebrows = sqrt(eyebrows_x*eyebrows_x + eyebrows_y*eyebrows_y);

				double nose_length_x = face_points[27].x - face_points[30].x;
				double nose_length_y = face_points[27].y - face_points[30].y;
				double nose_length = sqrt(nose_length_x*nose_length_x + nose_length_y*nose_length_y);

				double angry = eyebrows / nose_length;

				if (angry < 0.4&&-0.5 < average_test_y&&average_test_y < 0.5)
				{
					cv::putText(im, cv::format("angry"), cv::Point(face_points[19].x, (face_points[21].y + face_points[22].y) / 2), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
				}
*/

		}
		
		
		
		if (pre_count % 3 == 0)
		{
			int k = cv::waitKey(3);		
			// Quit if 'q' or ESC is pressed
			if (k == 'q' || k == 27)
			{
				printData(drone_rotation_y);
				ros::shutdown();
				return;
			}
		}
		
		win.clear_overlay();
		win.set_image(cimg);
		win.add_overlay(render_face_detections(shapes));
			
	}

	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}

	catch (exception& e)
	{
		cout << e.what() << endl;
	}

}

ros::Publisher camera_control_pub_;
ros::Publisher cmd_vel_pub_;

cv_bridge::CvImagePtr cv_ptr;

void imageCb(const sensor_msgs::ImageConstPtr& msg)
{

	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		head_estimator(cv_ptr);
  	
		camera_control.angular.z=camera_pan_error+pre_camera_pan_error;
		camera_control_pub_.publish(camera_control);
		
		cout<<camera_control.angular.z<<endl;
  	
		cmd_vel.linear.x=0;		
		cmd_vel.linear.y=-drone_vy;
		cmd_vel.linear.z=0;
		
		cmd_vel_pub_.publish(cmd_vel);
  	
		int count=count_checker();

	}
  
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

/*  
	if(pre_count%2==0){
		camera_control.angular.z=0;
	}else{
		camera_control.angular.z=0;
	}
  
	cout<<camera_control.angular.z<<endl;
	cout<<pre_count<<endl;
  
	camera_control_pub_.publish(camera_control);
*/

	cout<<pre_count<<endl;
  
	cv::imshow("OPENCV_WINDOW", cv_ptr->image);
	cv::waitKey(3);

}

int main(int argc, char** argv)
{

	ros::init(argc, argv, "image_converter");
	ros::NodeHandle nh_;

	image_transport::ImageTransport it_(nh_);
	image_transport::Subscriber image_sub_;

	//ros::Publisher camera_control_pub_;
	camera_control_pub_=nh_.advertise<geometry_msgs::Twist>("/bebop/camera_control",1);
  
	//ros::Publisher cmd_vel_pub_;
	cmd_vel_pub_=nh_.advertise<geometry_msgs::Twist>("/bebop/pre_cmd_vel",1);
  
	image_sub_ = it_.subscribe("/usb_cam/image_raw",1,&imageCb);
	//image_sub_ = it_.subscribe("/bebop/image_raw",1,&imageCb);
	ros::Rate loop_rate(30);

	while(ros::ok())
	{	
		//ros::spin();
	ros::spinOnce();
	loop_rate.sleep();
	cv::waitKey(3);
	}
	
	return 0;

}
