#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;
using namespace cv::face;
using namespace cv;

int input_Mode = 0;
Mat image;
string windowName = "Face Detection";
CascadeClassifier face_cascade;
std::vector<Rect> faces;
VideoCapture cap;
void onInputChange(int selected, void*) {
	if (selected == 1) 
		cap.open(0);
	else {
		cap.release();
		image = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	}
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main()
{
	//variable
	Mat face_resized, face_gray;
	vector<Mat> images;
	vector<int> labels;
	int prediction, index;
	double confidence;
	auto it=NULL;
	//code
	namedWindow(windowName);
	image = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	createTrackbar("Input Mode", windowName, &input_Mode, 1, &onInputChange);
	face_cascade.load("lbpcascade_frontalface.xml");
	try {
		read_csv("image.ext", images, labels);
	}
	catch (cv::Exception& e) {
		CV_Error(CV_StsError,"Error opening file, Reason: "+ e.msg);
		exit(EXIT_FAILURE);
	}
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);
	while (!(waitKey(30) >= 0)) {
		if (input_Mode == 1) 
			cap >> image;
		face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		for (int i = 0; i < faces.size(); i++) {
			rectangle(image, faces[i], Scalar(255, 0, 255));
			cvtColor(image, face_gray, CV_BGR2GRAY);
			resize(face_gray(faces[i]), face_resized, Size(images[0].cols, images[0].rows), 1.0, 1.0, INTER_CUBIC);
			model->predict(face_resized, prediction, confidence);
			//Mat temp = image(faces[i]);
			//resize(temp, temp, cvSize(200, 200));
			//imwrite("face" + std::to_string(i) + ".jpg", temp);
			auto it = std::find(labels.begin(), labels.end(), prediction);
			if (it != labels.end())
				index=std::distance(labels.begin(), it);
			if(prediction==1)
				putText(image, "Robert Downey Jr.", Point(std::max(faces[i].tl().x - 10, 0), std::max(faces[i].tl().y - 10, 0)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(183, 100, 173), 2.0);
			else
				putText(image, "Arnold Schwarzenegger", Point(std::max(faces[i].tl().x - 10, 0), std::max(faces[i].tl().y - 10, 0)), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(183, 100, 173), 2.0);
		}
		cout<<"Confidence : "<<confidence<<endl;
		imshow(windowName, image);
		imshow("Face in Database", images[index]);
	}

	return EXIT_SUCCESS;
}
