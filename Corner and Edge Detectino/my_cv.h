#ifndef MY_CV_H
#define MY_CV_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

// Basic function
void show_mat(std::string name, cv::Mat im);
void my_sobel(const cv::Mat _src, cv::Mat &dst, int dx, int dy);
void my_cornerMinEigenVal(const cv::Mat src, cv::Mat &dst, int blockSize);
void my_threshold(const cv::Mat _src, cv::Mat &dst, double thresh);
void my_dilate(const cv::Mat _src, cv::Mat &dst);
void draw_corner(cv::Mat &dst, const cv::Mat&corner);

// My Canny
void NMS(cv::Mat G, cv::Mat dir, cv::Mat &dst);
void trace(cv::Mat &nms, cv::Mat &edge, int r, int c, double lowT, double highT);
void double_threshold(cv::Mat nms, cv::Mat &dst, double lowT, double highT);
void my_canny(cv::Mat _src, cv::Mat &dst, double lowT, double highT);

// My Harris
void my_harris(cv::Mat gray, cv::Mat &corner, double qualityLevel, int blockSize, int ksize, double k);
void harris_demo(cv::Mat gray, cv::Mat &corner, double qualityLevel, int blockSize, int ksize, double k);

// My Shi-Tomasi
bool cmpCorners(const float * a, const float * b);
void my_shi_tomasi(cv::Mat _src, cv::Mat &_corners, int maxCorners, double qualityLevel, double minDistance, int blockSize);

#endif // MY_CV_H
