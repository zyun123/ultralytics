#ifndef HANDFOOTDETECTOR_H
#define HANDFOOTDETECTOR_H
#include<iostream>
#include<opencv2/opencv.hpp>
#include<torch/script.h>
using namespace std;

struct Detection{
    int classId;
    string className;
    vector<cv::Rect> box;
};

class HandFootDetector{
public:
    HandFootDetector(const string& model_path);
    // ~HandFootDetector();
    // ~HandFootDetector(void);
    vector<cv::Scalar> getColors(void);
    bool loadModel(const string& modelPath);
    void processImage(cv::Mat& src, cv::Mat& img);
    void runOnImage(cv::Mat& img,cv::Mat& output_mat);
    void predict(cv::Mat& src);

    // void HandFootDetector::drawBoundingBox(cv::Mat& src,string& label,cv::Rect& new_box,cv::Scalar color,double& scale);
private:
    vector<string> classes;
    vector<cv::Scalar> colors;
    string modelPath;
    double scale;
    // cv::Mat output;
    //保存网络推理粗结果
    
    torch::jit::script::Module module;
};


#endif