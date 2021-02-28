//
//  my_cv.h
//  OpenCV-1
//
//  Created by wzy on 2019/11/9.
//  Copyright © 2019年 wzy. All rights reserved.
//

#ifndef my_cv_h
#define my_cv_h

#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#define dbg(x) std::cout<<#x<<"="<<x<<std::endl


#define FEATURE_MAX_D 128
typedef float ImgDataType;
struct feature {
    double y;
    double x;
    
    int r;
    int c;
    int o;
    int layer;
    double scl;
    double scl_octv;
    
    int octave;
    double size;
    double angle;
    double response;
    
    int d;                         /**< descriptor length */
    double descr[FEATURE_MAX_D];
    
    void show() {
        std::cout<<"o="<<o<<" layer="<<layer<<" r="<<r<<" c="<<c<<" angle="<<angle<<" scale="<<scl<<std::endl;
    }
};


void show_mat(std::string name, cv::Mat im) {
    std::cout << name << std::endl;
    std::cout << im << std::endl << std::endl;
}

void my_sobel(const cv::Mat _src, cv::Mat &dst, int dx, int dy) {
    cv::Mat src = _src.clone();
    _src.convertTo(src, CV_32FC1);
    int row = src.rows, col = src.cols;
    cv::Mat Ix = cv::Mat::zeros(row, col, CV_32FC1), Iy = cv::Mat::zeros(row, col, CV_32FC1);
    for(int r=1, rlim=row-1; r<rlim; ++r) {
        for(int c=1, clim=col-1;  c<clim; ++c) {
            float y = (src.at<float>(r+1, c-1) + 2*src.at<float>(r+1, c) + src.at<float>(r+1, c+1)) - (src.at<float>(r-1, c-1) + 2*src.at<float>(r-1, c) + src.at<float>(r-1, c+1));
            float x = (src.at<float>(r-1, c+1) + 2*src.at<float>(r, c+1) + src.at<float>(r+1, c+1)) - (src.at<float>(r-1, c-1) + 2*src.at<float>(r, c-1) + src.at<float>(r+1, c-1));
            Ix.at<float>(r, c) = x;
            Iy.at<float>(r, c) = y;
        }
    }
    if(dx == 1 && dy == 0) dst = Ix;
    if(dy == 1 && dx == 0) dst = Iy;
}

void my_cornerMinEigenVal(const cv::Mat src, cv::Mat &dst, int blockSize, int ksize=3) {
    //  x,y方向梯度
    cv::Mat Ix, Iy;
    //    cv::Sobel(src, Ix, CV_32FC1, 1, 0, ksize);
    //    cv::Sobel(src, Iy, CV_32FC1, 0, 1, ksize);
    my_sobel(src, Ix, 1, 0);
    my_sobel(src, Iy, 0, 1);
    
    cv::Mat Ixx, Iyy, Ixy;
    Ixx = Ix.mul(Ix);
    Iyy = Iy.mul(Iy);
    
    //  滤波
    cv::boxFilter(Ixx, Ixx, Ixx.depth(), cv::Size(blockSize, blockSize));
    cv::boxFilter(Iyy, Iyy, Iyy.depth(), cv::Size(blockSize, blockSize));
    
    //  最小特征值
    cv::Mat res;
    res = cv::Mat::zeros(src.size(), CV_32FC1);
    for(int r=0, rlim=src.rows; r<rlim; ++r) {
        for(int c=0, clim=src.cols; c<clim; ++c) {
            float ix2=Ixx.at<float>(r, c), iy2=Iyy.at<float>(r, c);
            res.at<float>(r, c) = std::min(ix2, iy2);
        }
    }
    
    dst = res;
}

void my_threshold(const cv::Mat _src, cv::Mat &dst, double thresh) { // cv::THRESH_TOZERO
    cv::Mat src = _src.clone();
    dst = _src.clone();
    for(int r=0, rlim=src.rows; r<rlim; ++r) {
        const float *src_ptr = (const float*)src.ptr(r);
        float *dst_ptr = (float*)dst.ptr(r);
        
        for(int c=0, clim=src.cols; c<clim; ++c) {
            if(src_ptr[c] <= thresh)
                dst_ptr[c] = 0;
        }
    }
}

void my_dilate(const cv::Mat _src, cv::Mat &dst) { // 3x3
    cv::Mat src = _src.clone();
    dst = _src.clone();
    for(int r=0, rlim=src.rows; r<rlim; ++r) {
        const float *src_ptr = (const float*)src.ptr(r);
        float *dst_ptr = (float*)dst.ptr(r);
        
        for(int c=0, clim=src.cols; c<clim; ++c) {
            
            float maxVal = src_ptr[c];
            for(int rr=r-1, rrlim=r+1; rr<=rrlim; ++rr) {
                const float *cell_ptr = (const float*)src.ptr(rr);
                
                for(int cc=c-1, cclim=c+1; cc<=cclim; ++cc) {
                    if(rr < 0 || rr >= rlim || cc < 0 || cc >= clim) continue;
                    maxVal = std::max(maxVal, cell_ptr[cc]);
                }
            }
            
            dst_ptr[c] = maxVal;
        }
    }
}

#endif /* my_cv_h */
