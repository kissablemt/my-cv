//  
//  my_hough.h  
//  OpenCV-1  
//  
//  Created by wzy on 2019/11/24.  
//  Copyright © 2019年 wzy. All rights reserved.  
//  
#ifndef my_hough_h  
#define my_hough_h  
#include "my_cv.h"

//cv::HoughLines(cvcanny, lines, 1, CV_PI / 180, 120, 0, 0);  
  
void my_hough_line(const cv::Mat &edge, std::vector<cv::Vec2f> &lines, double rho, double theta, int threshold, double minLen, double maxGap) {  
  
    const int row = edge.rows;  
    const int col = edge.cols;  
    const double max_dist = sqrt(row*row + col*col);  
    const int angle_num = cvCeil(2*CV_PI / theta);  
    const int rho_num = cvCeil(max_dist/rho);  
      
    std::vector<float> sintht;  
    std::vector<float> costht;  
      
      
      
    for(int i=0; i<rho_num; ++i) {  
        sintht.push_back(sin(i * theta));  
        costht.push_back(cos(i * theta));  
    }  
//    std::cout << angle_num << std::endl;  
  
    std::vector< std::vector<int> > box(angle_num, std::vector<int>(rho_num + 5));  
    for(int y=0; y<row; ++y) {  
        for(int x=0; x<col; ++x) {  
            if(edge.at<uchar>(y, x) != 0) {  
                for(int i=0; i<angle_num; ++i) {  
                    int r = cvRound((y*costht[i] + x*sintht[i]) / rho);  
                    box[i][r]++;  
                }  
            }  
        }  
    }  
      
    std::vector< std::pair<int, int> > temp;  
    for(int i=0, ilim=(int)box.size(); i<ilim; ++i) {  
        for(int j=0, jlim=(int)(box[i].size()); j<jlim; ++j) {  
            int val = box[i][j];  
            temp.push_back(std::pair<int, int>(i, j));  
            if(val > threshold) {  
                temp.push_back(std::pair<int, int>(i, j));  
                int up = box[i-1>=0? i-1: 0][j];  
                int down = box[i+1<ilim? i+1: ilim-1][j];  
                int left = box[i][j-1>=0? j-1: 0];  
                int right = box[i][j+1<jlim? j+1: jlim-1];  
                if(val >= up && val >=down && val >= left && val >= right) {  
                    temp.push_back(std::pair<int, int>(i, j));  
                }  
            }  
        }  
    }  
      
    lines.clear();  
    for(int i=0, ilim=(int)temp.size(); i<ilim; ++i) {  
        cv::Vec2f v2(temp[i].first * theta, temp[i].second * max_dist / rho);  
        lines.push_back(v2);  
//        std::cout << lines[i] << std::endl;  
    }  
  
}  
  
#endif /* my_hough_h */