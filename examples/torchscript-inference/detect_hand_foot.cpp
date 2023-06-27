#include"detect_hand_foot.h"


HandFootDetector::HandFootDetector(const string& model_path){
    modelPath = model_path;
    classes = {"left_hand","right_hand","left_foot","right_foot"};
    colors = getColors();
    bool res = loadModel(modelPath);
    if(!res){
        cout << "load model err" << endl;
        // return -1;
    }
}


vector<cv::Scalar> HandFootDetector::getColors(void){
    vector<cv::Scalar> colors;
    cv::RNG rng;
    for(int i = 0; i<classes.size(); i++){
        int red = rng.uniform(0,255);
        int green = rng.uniform(0,255);
        int blue = rng.uniform(0,255);
        colors.push_back(cv::Scalar(red,green,blue));
    }
    return colors;
}
bool HandFootDetector::loadModel(const string& modelPath){
    try{
        module = torch::jit::load(modelPath);
    }
    catch(const c10::Error& e){
        std::cerr << "error loading model\n";
        return -1;
    }
    return 1;
}
void HandFootDetector::processImage(cv::Mat& src,cv::Mat& img){

    const int channels = src.channels(),height = src.rows,width = src.cols;
    int length = max(height,width);
    // cout << "height" << height << "width" <<width << "channels" << channels << "length" << length << endl;

    img = cv::Mat(length,length,CV_8UC3,cv::Scalar(0,0,0));
    src.copyTo(img(cv::Rect(0,0,width,height)));  //将原图像拷贝到新的图像上,左上角为原点

    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    img.convertTo(img,CV_32FC3,1.0/255,0);
    cv::resize(img,img,cv::Size(640,640)); //将img resize到640

    //记录一下length的正方形，缩放到640的比例
    scale = static_cast<double>(length) / 640;

}
void HandFootDetector::runOnImage(cv::Mat& img,cv::Mat& output_mat){
    auto input = torch::from_blob(img.data,{1,640,640,3}).toType(torch::kFloat32);
    input = input.permute({0,3,1,2});
    torch::Tensor outputs = module.forward({input}).toTensor();   //shape(1,84,8400)
    // int rows = outputs[0].size[1];
    // int dimensions = outputs[0].size[2];
    // std::cout << "outputs: " <<outputs.sizes()<< std::endl;  //output.shape: [1,(x,y,x,y,class_score),anchor_num]  ex: cooc数据集里80个类别，那么shape[0] = 80+4

    int rows = outputs.sizes()[1];
    int cols = outputs.sizes()[2];
    // cout << "rows:" << rows << ", cols: " <<cols << endl;

    std::vector<float> data(outputs.data_ptr<float>(), outputs.data_ptr<float>() + outputs.numel());
    output_mat = cv::Mat(rows,cols,CV_32FC1,data.data());
    // cout << output_mat.size() << endl;

    cv::transpose(output_mat,output_mat);
}
// void HandFootDetector::drawBoundingBox(cv::Mat& src,string& label,cv::Rect& new_box,cv::Scalar& color,double& scale){
//     cv::putText(src,label,cv::Point(round(box.x * scale)-10,round(box.y*scale)-10),cv::FONT_HERSHEY_SIMPLEX,0.5,color,2);
//     cv::rectangle(src,new_box,color,2,8);
// }


void HandFootDetector::predict(cv::Mat& src){
    cv::Mat img,output_mat;
    processImage(src,img);
    runOnImage(img,output_mat);
    int rows = output_mat.rows;
    vector<cv::Rect> boxes;
    vector<float> scores;
    vector<int> classIds;
    cv::Point maxLoc;
    double maxScore;
    for(int i = 0; i<rows;i++) {
        cv::Mat scoresMat = output_mat.row(i).colRange(4,output_mat.cols);
        cv::minMaxLoc(scoresMat,nullptr,&maxScore,nullptr,&maxLoc);
        if(maxScore >= 0.25){
            cv::Rect box(output_mat.at<float>(i,0) - (0.5* output_mat.at<float>(i,2)),
                        output_mat.at<float>(i,1) - (0.5 * output_mat.at<float>(i,3)),
                        output_mat.at<float>(i,2),
                        output_mat.at<float>(i,3));
            boxes.push_back(box);
            scores.push_back(maxScore);
            classIds.push_back(maxLoc.x);
            // cout << "max score: " << maxScore << endl; 
            // cout << "max loc: " << maxLoc << endl;
        }
    }
    vector<int> resultIndices;
    // vector<Detection> detections;
    cv::dnn::NMSBoxes(boxes,scores,0.25,0.45,resultIndices);
    for(int i : resultIndices){
        cv::Rect box = boxes[i];
        double confidence = scores[i];
        int classId = classIds[i];
        string className = classes[classId];
        cv::Scalar color = colors[classId];
        // detections.push_back(Detection(classIds[i],classes[classId],scores[i],boxes[i],scale))
        cv::Rect new_box(round(box.x * scale),round(box.y*scale),round(box.width * scale),round(box.height*scale));
        ostringstream oss;
        oss << fixed << setprecision(2) << confidence;
        string new_confidence = oss.str();
        string label = className + "("+new_confidence+")";
        cv::putText(src,label,cv::Point(round(box.x * scale)-10,round(box.y*scale)-10),cv::FONT_HERSHEY_SIMPLEX,0.5,color,2);
        cv::rectangle(src,new_box,cv::Scalar(0,255,0),2,8);
        
    }
    cv::imshow("res",src);
    cv::waitKey(1);
    // cv::destroyAllWindows();
}
