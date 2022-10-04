#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "main.h"
using namespace std;

using namespace cv;

void corner_fast() {

	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	vector<KeyPoint> keypoints; 
	FAST(src, keypoints, 60, true);
	Mat dst; cvtColor(src, dst, COLOR_GRAY2BGR);
	for (KeyPoint kp : keypoints) {
		Point pt(cvRound(kp.pt.x), cvRound(kp.pt.y));
		circle(dst, pt, 5, Scalar(0, 0, 255), 2);
	}
	imshow("src", src);
	imshow("dst", dst);
	waitKey()	; 
	destroyAllWindows();

}
void detect_keypoint() {
	//orb 생성시 파라메터 변경
	Mat src = imread("box_in_scene.png", IMREAD_GRAYSCALE);
	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Ptr<Feature2D> feature = ORB::create(500,2.0f,8,31,0,2,ORB::HARRIS_SCORE,31,20);


	vector<KeyPoint> keypoints; 
	feature->detect(src, keypoints);
			
	Mat desc; 
	feature->compute(src, keypoints, desc);
	cout << "keypoints.size(): " << keypoints.size() << endl;
	cout << "desc.size(): " << desc.size() << endl;
	Mat dst; 
	drawKeypoints(src, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("src", src);
	imshow("dst", dst);
	waitKey(); 
	destroyAllWindows();

}

void keypoint_matching() {
	Mat src1 = imread("box.png", IMREAD_GRAYSCALE);
	Mat src2 = imread("box_in_scene.png", IMREAD_GRAYSCALE);
	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Ptr<Feature2D> feature = ORB::create();
	vector<KeyPoint> keypoints1, keypoints2; 
	Mat desc1, desc2;
	feature->detectAndCompute(src1, Mat(), keypoints1, desc1); 
	feature->detectAndCompute(src2, Mat(), keypoints2, desc2); 
	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);
	std::sort(matches.begin(), matches.end());
	vector<DMatch> good_matches(matches.begin(), matches.begin()+50);
	Mat dst; drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst);
	imshow("dst", dst);
	waitKey(); 
	destroyAllWindows();

}


void template_matching() {
	Mat img = imread("circuit.bmp", IMREAD_COLOR);
	Mat templ = imread("crystal.bmp", IMREAD_COLOR);
	if (img.empty() || templ.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	img = img + Scalar(50, 50, 50);
	Mat noise(img.size(), CV_32SC3);
	randn(noise, 0, 10);
	add(img, noise, img, Mat(), CV_8UC3);
	Mat res, res_norm;
	matchTemplate(img, templ, res, TM_CCOEFF_NORMED);
	normalize(res, res_norm, 0, 255, NORM_MINMAX, CV_8U);
	double maxv;
	Point maxloc;
	minMaxLoc(res, 0, &maxv, 0, &maxloc);
	cout << "maxc : " << maxv << endl;
	rectangle(img, Rect(maxloc.x, maxloc.y, templ.cols, templ.rows), Scalar(0, 0, 255), 2);

	imshow("templ", templ);
	imshow("res_norm", res_norm);
	imshow("img", img);
	waitKey(0);
	destroyAllWindows();


}
void corner_harris() {
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat harris;
	cornerHarris(src, harris, 3, 3, 0.04);

	Mat harris_norm;
	normalize(harris, harris_norm, 0, 255, NORM_MINMAX, CV_8U);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (int j = 1;  j < harris.rows - 1; j++) {
		for (int i = 1; i < harris.cols - 1; i++) {
			if (harris_norm.at<uchar>(j, i) > 120) {
				if (harris.at<float>(j, i) > harris.at<float>(j - 1, i) &&
					harris.at<float>(j, i) > harris.at<float>(j + 1, i) &&
					harris.at<float>(j, i) > harris.at<float>(j, i - 1) &&
					harris.at<float>(j, i) > harris.at<float>(j, i + 1)
					) {

					circle(dst, Point(i, j), 5, Scalar(0, 0, 255), 2);
				}
			}
		}
	}
	imshow("src", src);
	imshow("harris_norm", harris_norm);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}
int main() {
	keypoint_matching();
}
