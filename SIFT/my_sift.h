//
//  my_sift.h
//  OpenCV-1
//
//  Created by wzy on 2019/11/20.
//  Copyright © 2019年 wzy. All rights reserved.
//

#ifndef my_sift_h
#define my_sift_h

#include "my_cv.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#define SIFT_INIT_SIGMA 0.5
#define SIFT_IMG_BORDER 5
#define SIFT_MAX_INTERP_STEPS 5
#define SIFT_FIXPT_SCALE 1
#define SIFT_DESCR_WIDTH 4
#define SIFT_DESCR_HIST_BINS 8
#define SIFT_ORI_HIST_BINS 36
#define SIFT_ORI_SIG_FCTR 1.5
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR
#define SIFT_ORI_SMOOTH_PASSES 2
#define SIFT_ORI_PEAK_RATIO 0.8
#define SIFT_DESCR_SCL_FCTR 3.0
#define SIFT_DESCR_MAG_THR 0.2
#define SIFT_INT_DESCR_FCTR 512.0
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

void draw_lowe_features(cv::Mat &img, std::vector<feature> keypoints);
void draw_lowe_feature(cv::Mat &img, feature feat, cv::Scalar color);
void my_sift(const cv::Mat &ori_img);
void init_img(const cv::Mat &img, cv::Mat &dst, double sigma);
void build_gaussian_pyramid(const cv::Mat &img, std::vector<cv::Mat> &pyr, int octvs, int layers, double sigma);
void build_dog_pyramid(const std::vector<cv::Mat> &gauss_pyr,std::vector<cv::Mat> &dog_pyr, int octvs, int layers);
void find_scale_space_extrema(const std::vector<cv::Mat> &dog_pyr, int octvs, int layers, std::vector<feature>& keypoints, double contr_thr, int curv_thr, double sigma);
bool is_extremum(const std::vector<cv::Mat> &dog_pyr, int octvs, int layers, int r, int c);
bool adjust_extremum(const std::vector<cv::Mat> &dog_pyr, feature &keypoint, const int layers, int oct, int layer, int r, int c, float contr_thr, float curv_thr, double sigma);
void interp_step(const std::vector<cv::Mat> &dog_pyr, const int layers, int octv, int layer, int r, int c, double &xi, double &xr, double &xc);
double interp_contr(const std::vector<cv::Mat> &dog_pyr, const int layers, int octv, int layer, int r, int c, double xi, double xr, double xc);
bool is_edge_like(const std::vector<cv::Mat> &dog_pyr, const int layers, int octv, int layer, int r, int c, double curv_thr);
void calc_feature_oris(std::vector<feature> &keypoints, const std::vector<cv::Mat> &gauss_pyr, const int layers);
void ori_hist(const cv::Mat &img, std::vector<double> &hist, int r, int c, int n, int rad, double sigma);
bool calc_grad_mag_ori(const cv::Mat &img, int r, int c, double &mag, double &ori);
void smooth_ori_hist(std::vector<double> &hist);
double dominant_ori(const std::vector<double> &hist);
void new_good_ori_feature(std::vector<feature> &keypoints, feature kpt, const std::vector<double> &hist, const double mag_thr);
void calc_descriptors(std::vector<feature> &keypoints, const std::vector<cv::Mat> &gauss_pyr, const int layers, int d, int n);
double*** descr_hist(const cv::Mat &img, int r, int c, double ori, double scl, int d, int n );
void interp_hist_entry( double*** hist, double rbin, double cbin, double obin, double mag, int d, int n );
void hist_to_descr( double*** hist, int d, int n, feature &feat );
void normalize_descr( feature &feat );
void release_descr_hist( double*** &hist, int d );

void my_sift(const cv::Mat &ori_img) {
    const double sigma = 1.6;
    const int layers = 3;
    
    cv::Mat img;
    init_img(ori_img, img, sigma);
    
    const int octvs = std::log2(std::min(img.rows, img.cols)) - layers;
    
    std::vector<cv::Mat> gauss_pyr;
    build_gaussian_pyramid(img, gauss_pyr, octvs, layers, sigma);
    //    for(int i=0; i<gauss_pyr.size(); ++i) cv::imshow(std::to_string(i), gauss_pyr[i]);
    //    std::cout << gauss_pyr[0].depth() << std::endl;
    
    std::vector<cv::Mat> dog_pyr;
    build_dog_pyramid(gauss_pyr, dog_pyr, octvs, layers);
    //    for(int i=0; i<dog_pyr.size(); ++i) cv::imshow(std::to_string(i), dog_pyr[i]);
    
    
    std::vector<feature> keypoints;
    find_scale_space_extrema(dog_pyr, octvs, layers, keypoints, 0.04, 10, sigma);
    //    std::cout << keypoints.size() << std::endl; return ;
    
    calc_feature_oris(keypoints, gauss_pyr, layers);
    //    std::cout << keypoints.size() << std::endl; return ;
    
    calc_descriptors(keypoints, gauss_pyr, layers, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS);
    //    std::cout << keypoints.size() << std::endl; return ;
    
    //    for(int i=0; i<keypoints.size(); ++i) keypoints[i].show();
    cv::Mat dst = ori_img.clone();
    draw_lowe_features(dst, keypoints);
    cv::imshow("My Sift", dst);
    
}

void draw_lowe_features(cv::Mat &img, std::vector<feature> keypoints) {
    const cv::Scalar color = cv::Scalar(0, 255, 0);
    for( int i = 0, n=(int)keypoints.size(); i < n; i++ )
        draw_lowe_feature( img, keypoints[i], color );
}

void draw_lowe_feature(cv::Mat &img, feature feat, const cv::Scalar color) {
    int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
    double scl, ori;
    double scale = 5.0;
    double hscale = 0.75;
    //    CvPoint start, end, h1, h2;
    cv::Point start, end, h1, h2;
    
    /* compute points for an arrow scaled and rotated by feat's scl and ori */
    start_x = cvRound( feat.x );
    start_y = cvRound( feat.y );
    scl = feat.scl;
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
    
    start = cv::Point( start_x, start_y );
    end = cv::Point( end_x, end_y );
    h1 = cv::Point( h1_x, h1_y );
    h2 = cv::Point( h2_x, h2_y );
    
    //    cvLine( dst, start, end, color, 1, 8, 0 );
    //    cvLine( dst, end, h1, color, 1, 8, 0 );
    //    cvLine( dst, end, h2, color, 1, 8, 0 );
    cv::line(img, start, end, color, 3);
    cv::line(img, end, h1, color, 3);
    cv::line(img, end, h2, color, 3);
}

void init_img(const cv::Mat &img, cv::Mat &dst, double sigma) {
    cv::cvtColor(img, dst, cv::COLOR_RGB2GRAY);
    double sig_diff = std::sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
    cv::GaussianBlur(dst, dst, cv::Size(), sig_diff, sig_diff);
    dst.convertTo(dst, CV_32FC1);
}

void build_gaussian_pyramid(const cv::Mat &img, std::vector<cv::Mat> &gauss_pyr, int octvs, int layers, double sigma) {
    const int nOL = layers + 3;
    
    /* 预处理高斯尺度 */
    std::vector<double> sig(layers + 3);
    const double k = std::pow(2., 1. / layers);
    sig[0] = sigma;
    sig[1] = sigma * std::sqrt(k*k - 1);
    for(int i=2; i<nOL; ++i) {
        sig[i] = sig[i-1] * k;
    }
    
    /* 获得金字塔 */
    gauss_pyr.resize(octvs * nOL);
    for(int o=0; o<octvs; ++o) {
        for(int i=0; i<nOL; ++i) {
            cv::Mat &cur = gauss_pyr[o * nOL + i];
            
            if(o == 0 && i==0) {
                cur = img.clone();
            }
            /* 下采样 */
            else if(i == 0) {
                const cv::Mat &prevO = gauss_pyr[(o-1) * nOL + nOL - 3];
                cv::resize(prevO, cur, cv::Size(prevO.cols/2, prevO.rows/2), 0, 0, cv::INTER_NEAREST);
            }
            else {
                const cv::Mat &prev = gauss_pyr[o*nOL + i-1];
                cv::GaussianBlur(prev, cur, cv::Size(), sig[i], sig[i]);
            }
        }
    }
}

void build_dog_pyramid(const std::vector<cv::Mat> &gauss_pyr,std::vector<cv::Mat> &dog_pyr, int octvs, int layers) {
    const int nOL = layers + 2;
    //        获取高斯差分金字塔
    dog_pyr.resize(octvs * nOL);
    for(int o=0; o<octvs; ++o) {
        for(int i=0; i<nOL; ++i) {
            cv::Mat &cur = dog_pyr[o * nOL + i];
            const cv::Mat &m = gauss_pyr[o * (nOL + 1) + i + 1];
            const cv::Mat &n = gauss_pyr[o * (nOL + 1) + i];
            cur = m - n;
            //            cv::normalize(cur, cur, 0, 255, cv::NORM_MINMAX);
        }
    }
}

void find_scale_space_extrema(const std::vector<cv::Mat> &dog_pyr, int octvs, int layers, std::vector<feature>& keypoints, double contr_thr, int curv_thr, double sigma) {
    const int threshold = 0.5 * contr_thr / layers;
    
    keypoints.clear();
    feature kpt;
    for(int o=1, olim=octvs-1; o<olim; ++o) {
        for(int i=1; i<=layers; ++i) {
            const int idx = o * (layers + 2) + i;
            const cv::Mat &cur = dog_pyr[idx];
            
            for(int r=SIFT_IMG_BORDER, rlim=cur.rows-SIFT_IMG_BORDER; r<rlim; ++r) {
                for(int c=SIFT_IMG_BORDER, clim=cur.cols-SIFT_IMG_BORDER; c<clim; ++c) {
                    if(std::abs(cur.at<ImgDataType>(r, c)) > threshold) {
                        if(is_extremum(dog_pyr, o, i, r, c)) {
                            //                            keypoints.push_back(kpt);
                            if(adjust_extremum(dog_pyr, kpt, layers, o, i, r, c, contr_thr, curv_thr, sigma))
                                keypoints.push_back(kpt);
                        }
                    }
                }
            }
            
        }
    }
}

bool is_extremum(const std::vector<cv::Mat> &dog_pyr, int octvs, int layers, int r, int c) {
    const ImgDataType val = dog_pyr[octvs].at<ImgDataType>(r, c);
    
    for(int o=-1; o<=1; ++o) {
        const cv::Mat &cur = dog_pyr[octvs + o];
        for(int i=-1; i<=1; ++i) {
            for(int j=-1; j<=1; ++j) {
                if((val > 0 && cur.at<ImgDataType>(r+i, c+j) > val) ||
                   (val < 0 && cur.at<ImgDataType>(r+i, c+j) < val)) return false;
            }
        }
    }
    
    return true;
}

bool adjust_extremum(const std::vector<cv::Mat> &dog_pyr, feature &keypoint, const int layers, int oct, int layer, int r, int c, float contr_thr, float curv_thr, double sigma) {
    
    const int idx = oct * (layers + 2) + layer;
    const cv::Mat &img = dog_pyr[idx];
    
    double xi=0, xr=0, xc=0;
    
    int iter = 0;
    while(iter < SIFT_MAX_INTERP_STEPS) {
        
        if(layer < 1 || layer >layers ||
           r < SIFT_IMG_BORDER || r >= img.rows-SIFT_IMG_BORDER ||
           c < SIFT_IMG_BORDER || c >= img.cols-SIFT_IMG_BORDER)
            return false;
        
        
        interp_step(dog_pyr, layers, oct, layer, r, c, xi, xr, xc);
        if(std::abs(xi) < 0.5 && std::abs(xr) < 0.5 && std::abs(xc) < 0.5) break;
        
        layer += cvRound(xi);
        r += cvRound(xr);
        c += cvRound(xc);
        
        iter++;
    }
    
    if(iter >= SIFT_MAX_INTERP_STEPS) return false;
    
    double contr = interp_contr(dog_pyr, layers, oct, layer, r, c, xi, xr, xc);
    if(std::abs(contr) < contr_thr) return false;
    
    if(is_edge_like(dog_pyr, layers, oct, layer, r, c, curv_thr)) return false;
    
    
    keypoint.r = r;
    keypoint.c = c;
    keypoint.o = oct;
    //    std::cout<<"dog="<<dog_pyr[oct*(layers+2)+layer].depth()<<std::endl;
    keypoint.layer = layer;
    keypoint.y = (r + xr) * (1 << oct);
    keypoint.x = (c + xc) * (1 << oct);
    keypoint.scl_octv = sigma * pow( 2.0, layer / layers );
    keypoint.scl = sigma * pow( 2.0, oct + layer / layers );
    keypoint.octave = oct + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
    keypoint.size = sigma*powf(2.f, (layer + xi) / layers)*(1 << oct)*2;
    keypoint.response = std::abs(contr);
    //    keypoint.show();
    return true;
}

void interp_step(const std::vector<cv::Mat> &dog_pyr, const int layers, int octv, int layer, int r, int c, double &xi, double &xr, double &xc) {
    const int idx = octv * (layers + 2) + layer;
    
    const cv::Mat &cur = dog_pyr[idx];
    const cv::Mat &pre = dog_pyr[idx - 1];
    const cv::Mat &nxt = dog_pyr[idx + 1];
    
    float dx = (cur.at<ImgDataType>(r, c+1) - cur.at<ImgDataType>(r, c-1)) / 2.0;
    float dy = (cur.at<ImgDataType>(r+1, c) - cur.at<ImgDataType>(r-1, c)) / 2.0;
    float ds = (nxt.at<ImgDataType>(r, c) - pre.at<ImgDataType>(r, c)) / 2.0;
    
    float val2 = cur.at<ImgDataType>(r, c) * 2;
    float dxx = cur.at<ImgDataType>(r, c+1) + cur.at<ImgDataType>(r, c-1) - val2;
    float dyy = cur.at<ImgDataType>(r+1, c) + cur.at<ImgDataType>(r-1, c) - val2;
    float dss = nxt.at<ImgDataType>(r, c) + pre.at<ImgDataType>(r, c) - val2;
    float dxy = (cur.at<ImgDataType>(r+1, c+1) + cur.at<ImgDataType>(r-1, c-1) - cur.at<ImgDataType>(r+1, c-1) -cur.at<ImgDataType>(r-1, c+1)) / 4.0;
    float dxs = (nxt.at<ImgDataType>(r, c+1) - nxt.at<ImgDataType>(r, c-1) - pre.at<ImgDataType>(r, c+1) + pre.at<ImgDataType>(r, c-1)) / 4.0;
    float dys = (nxt.at<ImgDataType>(r+1, c) - nxt.at<ImgDataType>(r-1, c) - pre.at<ImgDataType>(r+1, c) + pre.at<ImgDataType>(r-1, c)) / 4.0;
    
    cv::Vec3f dD(dx, dy, ds);
    cv::Matx33f dDD(
                    dxx, dxy, dxs,
                    dxy, dyy, dys,
                    dxs, dys, dss
                    );
    /*
     Dr = D^(-1)  Dt是D的逆矩阵
     c = - dDrDr * dD => - dDD * c = dD
     
     */
    
    cv::Vec3f X = dDD.solve(dD, cv::DECOMP_LU);
    xi = -X[2];
    xr = -X[1];
    xc = -X[0];
}

double interp_contr(const std::vector<cv::Mat> &dog_pyr, const int layers, int octv, int layer, int r, int c, double xi, double xr, double xc) {
    const double img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const int idx = octv * (layers + 2) + layer;
    
    const cv::Mat &cur = dog_pyr[idx];
    const cv::Mat &pre = dog_pyr[idx - 1];
    const cv::Mat &nxt = dog_pyr[idx + 1];
    
    float dx = (cur.at<ImgDataType>(r, c+1) - cur.at<ImgDataType>(r, c-1)) / 2.0;
    float dy = (cur.at<ImgDataType>(r+1, c) - cur.at<ImgDataType>(r-1, c)) / 2.0;
    float ds = (nxt.at<ImgDataType>(r, c) - pre.at<ImgDataType>(r, c)) / 2.0;
    
    cv::Matx31f dD(dx, dy, ds);
    float t = dD.dot(cv::Matx31f(xc, xr, xi));
    //    float contr = cur.at<ImgDataType>(r, c)*img_scale + t * 0.5f;
    float contr = cur.at<ImgDataType>(r, c) + t * 0.5f;
    return contr;
}

bool is_edge_like(const std::vector<cv::Mat> &dog_pyr, const int layers, int octv, int layer, int r, int c, double curv_thr) {
    const int idx = octv * (layers + 2) + layer;
    const cv::Mat &cur = dog_pyr[idx];
    
    float val2 = cur.at<ImgDataType>(r, c) * 2;
    float dxx = cur.at<ImgDataType>(r, c+1) + cur.at<ImgDataType>(r, c-1) - val2;
    float dyy = cur.at<ImgDataType>(r+1, c) + cur.at<ImgDataType>(r-1, c) - val2;
    float dxy = (cur.at<ImgDataType>(r+1, c+1) + cur.at<ImgDataType>(r-1, c-1) - cur.at<ImgDataType>(r+1, c-1) -cur.at<ImgDataType>(r-1, c+1)) / 4.0;
    
    float tr = dxx + dyy;
    float det = dxx * dyy - dxy * dxy;
    
    if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
        return false;
    return true;
}

void calc_feature_oris(std::vector<feature> &keypoints, const std::vector<cv::Mat> &gauss_pyr, const int layers) {
    const int n = (int)keypoints.size();
    
    std::vector<feature> nkpts;
    std::vector<double> hist;
    
    //    std::cout<<gauss_pyr.size()<<std::endl;
    //    std::cout<<gauss_pyr[3*(layers+3) + 2].depth()<<std::endl;
    //    return ;
    
    for(int i=0; i<n; ++i) {
        feature kpt = keypoints[i];
        
        ori_hist(gauss_pyr[kpt.o * (layers + 3) + kpt.layer], hist,
                 kpt.r, kpt.c,
                 SIFT_ORI_HIST_BINS,
                 cvRound( SIFT_ORI_RADIUS * kpt.scl_octv ),
                 SIFT_ORI_SIG_FCTR * kpt.scl_octv);
        
        for(int j=0; j<SIFT_ORI_SMOOTH_PASSES; ++j)
            smooth_ori_hist(hist);
        
        double omax = dominant_ori(hist);
        
        new_good_ori_feature(nkpts, kpt, hist, omax * SIFT_ORI_PEAK_RATIO);
    }
    
    swap(keypoints, nkpts);
}

void ori_hist(const cv::Mat &img, std::vector<double> &hist, int r, int c, int n, int rad, double sigma) {
    
    const double PI2 = CV_PI * 2.0;
    
    hist.resize(n);
    
    double exp_denom = 2.0 * sigma * sigma;
    for(int i=-rad; i<=rad; ++i) {
        for(int j=-rad; j<=rad; ++j) {
            double mag, ori;
            if(calc_grad_mag_ori(img, r+i, c+j, mag, ori)) {
                double w = std::exp(-(i*i + j*j) / exp_denom);
                int bin = cvRound(n * (ori + CV_PI) / PI2);
                bin = (bin < n)? bin: 0;
                hist[bin] += w * mag;
            }
        }
    }
}


bool calc_grad_mag_ori(const cv::Mat &img, int r, int c, double &mag, double &ori) {
    if(r > 0 && r < img.rows-1 && c > 0 && c < img.cols-1) {
        
        double dx = img.at<ImgDataType>(r, c+1) - img.at<ImgDataType>(r, c-1);
        double dy = img.at<ImgDataType>(r-1, c) - img.at<ImgDataType>(r+1, c);
        mag = sqrt(dx * dx + dy * dy);
        ori = atan2(dy, dx);
        return true;
    }
    return false;
}

void smooth_ori_hist(std::vector<double> &hist) {
    const double h0=hist[0];
    const int n=(int)hist.size();
    
    double pre=hist[n-1], cur;
    for(int i=0; i<n; ++i) {
        cur = hist[i];
        hist[i] = 0.25*pre + 0.5*cur + 0.25*(i+1==n?h0:hist[i+1]);
        pre = cur;
    }
}

double dominant_ori(const std::vector<double> &hist) {
    double omax = hist[0];
    for(int i=1, n=(int)hist.size(); i<n; ++i) {
        if(hist[i] > omax) omax=hist[i];
    }
    return omax;
}


inline double interp_hist_peak(int l, int c, int r) {
    return 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r));
}

void new_good_ori_feature(std::vector<feature> &keypoints, feature kpt, const std::vector<double> &hist, const double mag_thr) {
    
    const double PI2 = CV_PI * 2.0;
    const int n = (int)hist.size();
    
    for(int i=0; i<n; ++i) {
        const int l = (i-1+n)%n;
        const int r = (i+1)%n;
        //        std::cout<<hist[i]<<" ";
        if(hist[i]>hist[l] && hist[i]>hist[r] && hist[i]>=mag_thr) {
            double bin = i + interp_hist_peak(hist[l], hist[i], hist[r]);
            bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;
            
            kpt.angle = (PI2 * bin / n) - CV_PI;
            keypoints.push_back(kpt);
        }
    }
    //    std::cout<<std::endl;
}

void calc_descriptors(std::vector<feature> &keypoints, const std::vector<cv::Mat> &gauss_pyr, const int layers, int d, int n) {
    double*** hist;
    const int sz = (int)keypoints.size();
    
    for(int i=0; i<sz; ++i) {
        feature kpt = keypoints[i];
        hist = descr_hist(gauss_pyr[kpt.o * (layers + 3) + kpt.layer], kpt.r, kpt.c, kpt.angle, kpt.scl_octv, d, n);
        
        hist_to_descr( hist, d, n, kpt );
        
        release_descr_hist(hist, d);
    }
}

double*** descr_hist(const cv::Mat &img, int r, int c, double ori, double scl, int d, int n ) {
    double*** hist;
    double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
    grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
    int radius, i, j;
    
    hist = new double**[d];
    for( i = 0; i < d; i++ ) {
        hist[i] = new double*[d];
        for( j = 0; j < d; j++ ) hist[i][j] = new double[n];
    }
    
    cos_t = cos( ori );
    sin_t = sin( ori );
    bins_per_rad = n / PI2;
    exp_denom = d * d * 0.5;
    hist_width = SIFT_DESCR_SCL_FCTR * scl;
    radius = hist_width * sqrt(2) * ( d + 1.0 ) * 0.5 + 0.5;
    for( i = -radius; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ ) {
            c_rot = ( j * cos_t - i * sin_t ) / hist_width;
            r_rot = ( j * sin_t + i * cos_t ) / hist_width;
            rbin = r_rot + d / 2 - 0.5;
            cbin = c_rot + d / 2 - 0.5;
            
            if( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )
                if( calc_grad_mag_ori( img, r + i, c + j, grad_mag, grad_ori ))
                {
                    grad_ori -= ori;
                    while( grad_ori < 0.0 )
                        grad_ori += PI2;
                    while( grad_ori >= PI2 )
                        grad_ori -= PI2;
                    
                    obin = grad_ori * bins_per_rad;
                    w = exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );
                    interp_hist_entry( hist, rbin, cbin, obin, grad_mag * w, d, n );
                }
        }
    
    return hist;
}

void interp_hist_entry( double*** hist, double rbin, double cbin, double obin, double mag, int d, int n ) {
    double d_r, d_c, d_o, v_r, v_c, v_o;
    double** row, * h;
    int r0, c0, o0, rb, cb, ob, r, c, o;
    
    r0 = cvFloor( rbin );
    c0 = cvFloor( cbin );
    o0 = cvFloor( obin );
    d_r = rbin - r0;
    d_c = cbin - c0;
    d_o = obin - o0;
    
    for( r = 0; r <= 1; r++ ) {
        rb = r0 + r;
        if( rb >= 0  &&  rb < d ) {
            v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
            row = hist[rb];
            for( c = 0; c <= 1; c++ ) {
                cb = c0 + c;
                if( cb >= 0  &&  cb < d ) {
                    v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
                    h = row[cb];
                    for( o = 0; o <= 1; o++ ) {
                        ob = ( o0 + o ) % n;
                        v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
                        h[ob] += v_o;
                    }
                }
            }
        }
    }
}

void hist_to_descr( double*** hist, int d, int n, feature &feat ) {
    int int_val, i, r, c, o, k = 0;
    
    for( r = 0; r < d; r++ )
        for( c = 0; c < d; c++ )
            for( o = 0; o < n; o++ )
                feat.descr[k++] = hist[r][c][o];
    
    feat.d = k;
    normalize_descr( feat );
    for( i = 0; i < k; i++ )
        if( feat.descr[i] > SIFT_DESCR_MAG_THR )
            feat.descr[i] = SIFT_DESCR_MAG_THR;
    normalize_descr( feat );
    
    for( i = 0; i < k; i++ ) {
        int_val = SIFT_INT_DESCR_FCTR * feat.descr[i];
        feat.descr[i] = MIN( 255, int_val );
    }
}

void normalize_descr( feature &feat ) {
    double cur, len_inv, len_sq = 0.0;
    int i, d = feat.d;
    
    for( i = 0; i < d; i++ ) {
        cur = feat.descr[i];
        len_sq += cur*cur;
    }
    len_inv = 1.0 / sqrt( len_sq );
    for( i = 0; i < d; i++ )
        feat.descr[i] *= len_inv;
}

void release_descr_hist( double*** &hist, int d ) {
    int i, j;
    
    for( i = 0; i < d; i++) {
        for( j = 0; j < d; j++ )
            delete[] hist[i][j];
        delete[] hist[i];
    }
    delete[] hist;
    *hist = NULL;
}

#endif /* my_sift_h */
