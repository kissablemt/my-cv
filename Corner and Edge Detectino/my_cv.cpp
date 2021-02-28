#include "my_cv.h"

// Basic function
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

void my_cornerMinEigenVal(const cv::Mat src, cv::Mat &dst, int blockSize) {
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

void draw_corner(cv::Mat &dst, const cv::Mat&corner) {
    for(int r=0, rlim=corner.rows; r<rlim; ++r) {
        for(int c=0, clim=corner.cols; c<clim; ++c) {
            float v = corner.at<float>(r, c);
            if(v) {
//                cv::circle(dst, cv::Point(c,r), 2, cv::Scalar(0, 255, 0), 3);
                cv::rectangle(dst, cv::Point(c-5, r-5), cv::Point(c+5, r+5), cv::Scalar(0, 255, 0), 2);
            }
        }
    }
}

// My Canny
void NMS(cv::Mat G, cv::Mat dir, cv::Mat &dst) {
    int row = G.rows, col = G.cols;
    dst = cv::Mat::zeros(row, col, CV_32FC1);
    for(int r=1, rlim=row-1; r<rlim; ++r) {
        for(int c=1, clim=col-1;  c<clim; ++c) {
            float mag = G.at<float>(r, c);
            float angle = dir.at<float>(r, c);
            float p, q;

            if(fabs(angle) <= 22.5 || fabs(angle) > 157.5){
                p = G.at<float>(r, c - 1);
                q = G.at<float>(r, c + 1);
            }
            else if ((angle>112.5 && angle<=157.5) || (angle>-67.5 && angle<=-22.5)){
                p = G.at<float>(r - 1, c + 1);
                q = G.at<float>(r + 1, c - 1);
            }
            else if ((angle>=67.5 && angle<=112.5 ) || (angle>=-112.5 && angle<=-67.5)){
                p = G.at<float>(r - 1, c);
                q = G.at<float>(r + 1, c);
            }
            else /* if ((angle >=22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5)) */ {
                p = G.at<float>(r - 1, c - 1);
                q = G.at<float>(r + 1, c + 1);
            }

            if(mag >= p && mag >= q) dst.at<float>(r, c) = mag;
        }
    }
}

void trace(cv::Mat &nms, cv::Mat &edge, int r, int c, double lowT, double highT) { //3x3
    if(edge.at<float>(r, c) == 0) {
        edge.at<float>(r, c) = 255;
        for(int i=-1; i<=1; ++i) {
            for(int j=-1; j<=1; ++j) {
                int nr = r + i, nc = c + j;
                if(nr < 0 || nr >= nms.rows || nc < 0 || nc >= nms.cols) continue;
                float mag = nms.at<float>(nr, nc);
                if(mag >= lowT) trace(nms, edge, nr, nc, lowT, highT);
            }
        }
    }
}

void double_threshold(cv::Mat nms, cv::Mat &dst, double lowT, double highT) {
    int row = nms.rows, col = nms.cols;
    cv::Mat edge = cv::Mat::zeros(row, col, CV_32FC1);
    for(int r=0, rlim=row-1; r<rlim; ++r) {
        for(int c=0, clim=col-1; c<clim; ++c) {
            float mag = nms.at<float>(r, c);
            if(mag >= highT)
//                edge.at<float>(r, c) = 255;
                trace(nms, edge, r, c, lowT, highT);
            if(mag < lowT) edge.at<float>(r, c) = 0;
        }
    }
    dst = edge;
}

void my_canny(cv::Mat _src, cv::Mat &dst, double lowT, double highT) {
    cv::Mat src = _src.clone();
    src.convertTo(src, CV_32FC1);
    int row = src.rows, col = src.cols;
//    Step1.获得梯度幅值与方向
    cv::Mat G = cv::Mat::zeros(row, col, CV_32FC1), dir = cv::Mat::zeros(row, col, CV_32FC1);
    cv::Mat Ix, Iy;
    my_sobel(src, Ix, 1, 0);
    my_sobel(src, Iy, 0, 1);
    for(int r=1, rlim=row-1; r<rlim; ++r) {
        for(int c=1, clim=col-1;  c<clim; ++c) {
            float y = Iy.at<float>(r, c);
            float x = Ix.at<float>(r, c);
            G.at<float>(r, c) = std::abs(y) + std::abs(x);
            dir.at<float>(r, c) = std::atan2f((float)y, (float)x) / CV_PI * 180;
        }
    }
//    Step2.非极大值抑制
    NMS(G, dir, dst);
//    Step3.双阈值处理
    double_threshold(dst, dst, lowT, highT);
}

// My Harris
void my_harris(cv::Mat gray, cv::Mat &corner, double qualityLevel, int blockSize=3, int ksize=3, double k=0.04) {
    gray.convertTo(gray, CV_32FC1);
//    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0, 0);

// x,y方向梯度
    cv::Mat Ix, Iy;
//    cv::Sobel(gray, Ix, CV_32FC1, 1, 0, ksize);
//    cv::Sobel(gray, Iy, CV_32FC1, 0, 1, ksize);
    my_sobel(gray, Ix, 1, 0);
    my_sobel(gray, Iy, 0, 1);

    cv::Mat Ixx, Iyy, Ixy;
    Ixx = Ix.mul(Ix);
    Iyy = Iy.mul(Iy);
    Ixy = Ix.mul(Iy);

//  滤波
    cv::boxFilter(Ixx, Ixx, Ixx.depth(), cv::Size(blockSize, blockSize));
    cv::boxFilter(Iyy, Iyy, Iyy.depth(), cv::Size(blockSize, blockSize));
    cv::boxFilter(Ixy, Ixy, Ixy.depth(), cv::Size(blockSize, blockSize));

// 计算harris响应
    cv::Mat res;
    res = cv::Mat::zeros(gray.size(), CV_32FC1);
    for(int r=0, rlim=gray.rows; r<rlim; ++r) {
        for(int c=0, clim=gray.cols; c<clim; ++c) {
            float ix2=Ixx.at<float>(r, c), iy2=Iyy.at<float>(r, c), ixy=Ixy.at<float>(r, c);
            float det = ix2 * iy2 - ixy * ixy;
            float trace = ix2 + iy2;
            res.at<float>(r, c) = det - k * trace * trace;
        }
    }

//    cv::imshow("R", res);

// 非极大值抑制
    cv::Mat nms;
    nms = cv::Mat::zeros(res.size(), CV_32FC1);

    double maxs;
    cv::minMaxLoc(res, NULL, &maxs);
    double thresh = qualityLevel * maxs;

    int h = blockSize / 2;
    for(int r=h, rlim=res.rows-h; r<rlim; ++r) {
        for(int c=h, clim=res.cols-h; c<clim; ++c) {
            float v = res.at<float>(r, c);

            bool is_max = true;
            for(int i=r-h, ilim=r+h; i<ilim && is_max; ++i) {
                for(int j=c-h, jlim=c+h; j<jlim; ++j) {
                    if(v < res.at<float>(i, j)) {
                        is_max = false;
                        v = 0;
                        break;
                    }
                }
            }

            if(v > thresh) nms.at<float>(r, c) = 255;
            else nms.at<float>(r, c) = 0;
        }
    }

    corner = nms;
}

void harris_demo(cv::Mat gray, cv::Mat &corner, double qualityLevel, int blockSize=3, int ksize=3, double k=0.04) {
    gray.convertTo(gray, CV_32FC1);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0, 0);

    cv::Mat res;
    res = cv::Mat::zeros(gray.size(), CV_32FC1);
    cv::cornerHarris(gray, res, blockSize, ksize, k);
//    cv::imshow("R_demo", res);

    // 非极大值抑制
    cv::Mat nms;
    nms = cv::Mat::zeros(res.size(), CV_32FC1);

    double maxs;
    cv::minMaxLoc(res, NULL, &maxs);
    double thresh = qualityLevel * maxs;
    int h = blockSize / 2;
    for(int r=h, rlim=res.rows-h; r<rlim; ++r) {
        for(int c=h, clim=res.cols-h; c<clim; ++c) {
            float v = res.at<float>(r, c);

            bool is_max = true;
            for(int i=r-h, ilim=r+h; i<ilim && is_max; ++i) {
                for(int j=c-h, jlim=c+h; j<jlim; ++j) {
                    if(v < res.at<float>(i, j)) {
                        is_max = false;
                        v = 0;
                        break;
                    }
                }
            }

            if(v > thresh) nms.at<float>(r, c) = 255;
            else nms.at<float>(r, c) = 0;
        }
    }

    corner = nms;
}

// My Shi-Tomasi
bool cmpCorners(const float * a, const float * b) {
    return (*a > *b) ? true : (*a < *b) ? false : (a > b);
}

void my_shi_tomasi(cv::Mat _src, cv::Mat &_corners, int maxCorners, double qualityLevel, double minDistance, int blockSize) {
    cv::Mat image = _src.clone();
    int row = image.rows, col = image.cols;


    cv::Mat eig, dil;
//    cv::cornerMinEigenVal(image, eig, blockSize, 3);
    my_cornerMinEigenVal(image, eig, blockSize);
    double maxVal = 0;
    cv::minMaxLoc(eig, NULL, &maxVal);
//    cv::threshold(eig, eig, qualityLevel * maxVal, 0, cv::THRESH_TOZERO);
    my_threshold(eig, eig, qualityLevel * maxVal);
//    cv::dilate(eig, dil, cv::Mat());
    my_dilate(eig, dil);


    std::vector<const float*> tmpCorners;
    for(int y=1, ylim=row-1; y<ylim; ++y) {
        const float *eig_ptr = (const float*)eig.ptr(y), *dil_ptr = (const float*)dil.ptr(y);

        for(int x=1, xlim=col-1; x<xlim; ++x) {
            float val = eig_ptr[x];
            if(val == dil_ptr[x]) tmpCorners.push_back(eig_ptr + x);
        }
    }
    std::sort(tmpCorners.begin(), tmpCorners.end(), cmpCorners);


    std::vector<cv::Point2f> corners;
    int ncorners=0;
    int cell_size = cvRound(std::max(0.0, minDistance));
    if(cell_size > 0) {
        int grid_w = (col - 1 + cell_size) / cell_size;
        int grid_h = (row - 1 + cell_size) / cell_size;

        int dis2 = cell_size * cell_size;
        std::vector< std::vector<cv::Point2f> > grid(grid_w * grid_h);
        for(auto it = tmpCorners.begin(); it != tmpCorners.end(); ++it) {
            int loc = (int)((const uchar*)(*it) - eig.data);
            int y = (int)(loc / eig.step);
            int x = (int)((loc - y * eig.step) / sizeof(float));

            int cell_y = y / cell_size;
            int cell_x = x / cell_size;

            int y1 = std::max(cell_y - 1, 0);
            int y2 = std::min(cell_y + 1, grid_h - 1);
            int x1 = std::max(cell_x - 1, 0);
            int x2 = std::min(cell_x + 1, grid_w - 1);

            bool keep = true;
            for(int yy=y1; yy<=y2 && keep; ++yy) {
                for(int xx=x1; xx<=x2; ++xx) {
                    std::vector<cv::Point2f> &pt = grid[yy * grid_w + xx];
                    for(auto i = pt.begin(); i != pt.end(); ++i) {
                        int dx = (*i).x - x;
                        int dy = (*i).y - y;
                        if(dx * dx + dy * dy < dis2) {
                            keep = false;
                            break;
                        }
                    }
                }
            }

            if(keep) {
                grid[cell_y * grid_w + cell_x].push_back(cv::Point2f((float)x, (float)y));
                corners.push_back(cv::Point2f((float)x, (float)y));
                ncorners += 1;
                if(maxCorners > 0 && ncorners >= maxCorners) break;
            }
        }
    }
    else {
        for(auto it = tmpCorners.begin(); it != tmpCorners.end(); ++it) {
            int loc = (int)((const uchar*)(*it) - eig.data);
            int y = (int)(loc / eig.step);
            int x = (int)((loc - y * eig.step) / sizeof(float));
            corners.push_back(cv::Point2f((float)x, (float)y));
            ncorners += 1;
            if(maxCorners > 0 && ncorners >= maxCorners) break;
        }
    }

    cv::Mat(corners).convertTo(_corners, CV_32FC1);
}
