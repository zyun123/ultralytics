#include<iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

// #include "../../../../../../usr/include/c++/9/bits/stream_iterator.h"


typedef struct _Result{
    int classId;
    float confidence;
    cv::Rect box;
}Result;

// void warmUpSession(Ort::RunOptions& options,Ort::Session* session,std::vector<int>& imageSize,std::vector<const char*>& inputNodeNames, std::vector<const char*>& outputNodeNames);




std::pair<float,std::vector<int>> PreProcess(cv::Mat& inputImage,cv::Mat& processedImage,std::vector<int>& imgSize){
    // std::vector<int> imgSize{640,640}; // hw
    int imgHeight = inputImage.rows;
    int imgWidth = inputImage.cols;
    std::cout << "imgheight: " << imgHeight << "  imgwidth: " << imgWidth << std::endl;
    std::cout << "imgsize: " << imgSize[0] << "x " << imgSize[1] << std::endl;
    float wScale = static_cast<float>(imgSize[1]) / static_cast<float>(imgWidth);
    float hScale = static_cast<float>(imgSize[0]) / static_cast<float>(imgHeight);

    std::cout << "wscale : " << wScale << " hscale: " << hScale << std::endl;
    float scale = std::min(wScale,hScale);
    std::cout << "final scale: " << scale << std::endl;

    std::vector<int> newShape{static_cast<int>(imgHeight * scale), static_cast<int>(imgWidth * scale)};

    std::cout << "new shape : " << newShape[0] << "x " << newShape[1] << std::endl;


    cv::Mat tmpImg;
    cv::resize(inputImage,tmpImg,cv::Size(newShape[1],newShape[0]),0,0,cv::INTER_AREA);



    float padh = (imgSize[0] - tmpImg.rows) / 2;
    float padw = (imgSize[1] - tmpImg.cols) / 2;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    std::cout << "top:" << top << "bottom: " << bottom << " left: " << left << " right: " << right << std::endl;

    cv::copyMakeBorder(tmpImg,processedImage,top,bottom,left,right,cv::BORDER_CONSTANT,cv::Scalar(114.));

    cv::namedWindow("resizeimg",cv::WINDOW_NORMAL);
    cv::imshow("resizeimg",processedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return {scale,{top,bottom,left,right}};

}

void BlobFromImage(cv::Mat& img,float* iblob){
    int channels = img.channels();
    int imgHeight = img.rows;
    int imgWidth = img.cols;

    for (int c = 0; c< channels; c++){
        for (int h = 0; h < imgHeight; h++){
            for (int w = 0; w < imgWidth; w++){
                iblob[c*imgHeight*imgWidth + h*imgWidth + w] = static_cast<float>(img.at<cv::Vec3b>(h,w)[c] / 255.0f);
            }
        }
    }


}

void warmUpSession(Ort::RunOptions& options,Ort::Session* session,std::vector<int>& imageSize,std::vector<const char*>& inputNodeNames, std::vector<const char*>& outputNodeNames){
    cv::Mat inputImage = cv::Mat(cv::Size(imageSize.at(1),imageSize.at(0)),CV_8UC3);
    cv::Mat processedImage;
    PreProcess(inputImage,processedImage,imageSize);
    float* blob = new float[inputImage.total() * 3];
    BlobFromImage(processedImage,blob);

    std::vector<int64_t> yolo_input_node_dims = {1,3,imageSize[0],imageSize[1]};
    // std::cout << "yolo input node dims: " << yolo_input_node_dims[0] << std::endl;
    std::cout << "yolo input node dims:" << std::endl;
    for(auto& i : yolo_input_node_dims){
        std::cout << i << "x";
    }
    std::cout << std::endl;
    // std::copy(yolo_input_node_dims.begin(),yolo_input_node_dims.end(),std::ostream_iterator<int>(std::cout," "));
    // std::cout << std::endl;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,OrtMemTypeCPU),blob,3*imageSize[0]*imageSize[1],yolo_input_node_dims.data(),yolo_input_node_dims.size()
    );
    auto output_tensors = session->Run(options,inputNodeNames.data(),&input_tensor,1,outputNodeNames.data(),1);

    delete[] blob;

}

void sessionInference(std::string& onnxPath,cv::Mat& inputImage, float rectThreshold, float iouThreshold,std::vector<int>& imgSize){
    
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"yolo");
    Ort::SessionOptions sessionOption;
    // OrtCUDAProviderOptions cudaOption;
    // cudaOption.device_id = 0;
    // cudaOption.arena_extend_strategy = 0;
    // cudaOption.gpu_mem_limit = (size_t)1*1024*1024*1024;
    // cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
    // cudaOption.do_copy_in_default_stream = 1;
    // sessionOption.AppendExecutionProvider_CUDA(cudaOption);


    std::cout << "add cuda option" << std::endl;
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOption.SetIntraOpNumThreads(1);
    sessionOption.SetLogSeverityLevel(3);


    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOption, 0));

    std::cout << "set done" << std::endl;

    const char* modelPath = onnxPath.c_str();
    auto session = new Ort::Session(env,modelPath,sessionOption);

    std::cout << "create session done" << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
    size_t inputNodesNum = session->GetInputCount();
    std::vector<const char*> inputNodeNames{"images"};
    std::vector<const char*> outputNodeNames{"output0"};

    std::cout << "input name:" << inputNodeNames[0] << std::endl;
    std::cout << "output name:" << outputNodeNames[0] << std::endl;


    Ort::RunOptions options = Ort::RunOptions{nullptr};

    warmUpSession(options,session,imgSize,inputNodeNames,outputNodeNames);

    std::cout << "warmup done  " << std::endl;
    // std::vector<int> imgSize{640,640}; // wh
    cv::Mat processedimg;
    std::pair<float, std::vector<int>> scale_pad;
    scale_pad = PreProcess(inputImage,processedimg,imgSize);
    float* blob = new float[processedimg.total() * 3];
    BlobFromImage(processedimg, blob);
    std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };

    //session run
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());

    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
    outputNodeNames.size());

    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info  = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<float>();
    delete[] blob;

    int strideNum = outputNodeDims[1]; // 8400
    int signalResultNum = outputNodeDims[2];  // 5  

    std::cout << "stridenum :" << strideNum << "  signalresultnum: " << signalResultNum << std::endl;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat rowData = cv::Mat(strideNum,signalResultNum,CV_32FC1,output);   // 8400 x 5    5:(x,y,w,h,cls_score)

    float* data = (float*)rowData.data;

    for (int i =0; i< strideNum;i++){
        float classScore = *(data + 4);
        int clasId = 0;  //只有一个类
        if (classScore > rectThreshold){
            confidences.push_back(classScore);
            class_ids.push_back(clasId);
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            std::cout << "x: " << x << " y: " << y << " w: " << w << " h: " << h <<std::endl;
            float scale = scale_pad.first;
            int top = scale_pad.second.at(0);
            int left = scale_pad.second.at(2);

            std::cout << "****top*****: " << top << "******left*****" << left<<std::endl;
            std::cout << "scale:  " << scale << std::endl;
            //return {scale,{top,bottom,left,right}};
            int tx = int((x - 0.5 * w - left) / scale);
            int ty = int((y - 0.5 * h - top) / scale);
            int width = int(w / scale);
            int height = int(h / scale);

            std::cout << "tx: " << tx << " ty: " << ty << " width: " << width << " height: " << height << std::endl;
            boxes.push_back(cv::Rect(tx,ty,width,height));
        }
        data+= signalResultNum;
        
    }
    std::vector<Result> pred_results;
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes,confidences,rectThreshold,iouThreshold,nmsResult);
    for (int i =0; i< nmsResult.size(); i++){
        int idx = nmsResult[i];
        Result result;
        result.classId = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        pred_results.push_back(result);
    }

}


int main(void){
    std::string onnxPath = "/home/zy/vision/ultralytics/runs/detect/train4/weights/best.onnx";
    std::string imgPath = "/home/zy/vision/ultralytics/hand_data/val/m_down_wai_20231016101752843_-499.jpg";
    std::vector<int> imgSize{640,640};
    cv::Mat inputImage = cv::imread(imgPath,cv::IMREAD_COLOR);
    sessionInference(onnxPath,inputImage,0.5,0.3,imgSize);
}