#include <iostream>
#include <algorithm>
#include <vector>
#include "my_cv.h"
#include "my_sift.h"
#include "sift_demo.h"
using namespace std;

void DrawSiftFeature(Mat& src, cv::KeyPoint& feat, CvScalar color)
{
    int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
    double scl, ori;
    double scale = 5.0;
    double hscale = 0.75;
    CvPoint start, end, h1, h2;
    
    /* compute points for an arrow scaled and rotated by feat's scl and ori */
    start_x = cvRound( feat.pt.x );
    start_y = cvRound( feat.pt.y );
    scl = feat.size;
    ori = feat.angle;
    len = cvRound( scl * scale );
    hlen = cvRound( scl * hscale );
    blen = len - hlen;
    end_x = cvRound( len *  cos( ori ) ) + start_x;
    end_y = cvRound( len * -sin( ori ) ) + start_y;
    h1_x = cvRound( blen *  cos( ori + CV_PI / 18.0 ) ) + start_x;
    h1_y = cvRound( blen * -sin( ori + CV_PI / 18.0 ) ) + start_y;
    h2_x = cvRound( blen *  cos( ori - CV_PI / 18.0 ) ) + start_x;
    h2_y = cvRound( blen * -sin( ori - CV_PI / 18.0 ) ) + start_y;
    start = cvPoint( start_x, start_y );
    end = cvPoint( end_x, end_y );
    h1 = cvPoint( h1_x, h1_y );
    h2 = cvPoint( h2_x, h2_y );
    
    line( src, start, end, color, 2, 4, 0 );
    line( src, end, h1, color, 2, 4, 0 );
    line( src, end, h2, color, 2, 4, 0 );
}

//ª≠≥ˆSIFTÃÿ’˜µ„
void DrawSiftFeatures(Mat& src, std::vector<cv::KeyPoint>& features, int n=-1)
{
    CvScalar color = CV_RGB( 0, 255, 0 );
    for(int i = 0; i < features.size(); i++)
    {
        if(0<n && i>=n) break;
        DrawSiftFeature(src, features[i], color);
    }
}

int main(int argc, const char * argv[]) {
    cv::Mat img = cv::imread("/Users/wzy/Pictures/opencv_test/2.jpeg");
//    cv::Mat img = cv::imread("/Users/wzy/Pictures/opencv_test/beaver.png");
//    cv::Mat img = cv::imread("/Users/wzy/Pictures/opencv_test/beaver_xform.png");
    cv::Mat gray;
    cv::cvtColor(img, gray, CV_RGB2GRAY);
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0, 0);
//    
//    SIFT sift;
//    sift.read(img);
//    
//    std::vector<cv::Mat> pyr;
//    sift.buildGaussianPyramid(pyr);

//    my_sift(img);
    
//    std::vector<Keypoint> keypoints;
//    Sift(img, keypoints, 1.6, 3);
//    cv::Mat mysift = img.clone();
//    DrawSiftFeatures(mysift, keypoints, 500);
//    cv::imshow("My Sift", mysift);
//
    Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(500);
    std::vector<cv::KeyPoint> kpts;
    sift->detect(img, kpts);
    cv::Mat cvsift = img.clone();
//    cv::drawKeypoints(img, kpts, cvsift);
    DrawSiftFeatures(cvsift, kpts, 500);
    cv::imshow("OpenCv Sift", cvsift);
    
    
    cv::waitKey();
    return 0;
}
