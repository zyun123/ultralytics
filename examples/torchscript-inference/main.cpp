#include"detect_hand_foot.h"
#include<experimental/filesystem>
namespace fs = std::experimental::filesystem;
int main(void) {
    string imgPath = "/911G/data/cure_images/一楼拷贝数据/up_nei/middle_up_nei/test/middle_up_nei_20230112103306626.jpg";
    string modelPath = "/home/zy/vision/ultralytics/runs/detect/train/weights/best.torchscript";
    cv::Mat src = cv::imread(imgPath);
    
    // cv::imshow("img",src);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    HandFootDetector detector(modelPath);
    detector.predict(src);
    detector.predict(src);
    detector.predict(src);
    detector.predict(src);
    detector.predict(src);

    string rootDir = "/911G/data/cure_images/一楼拷贝数据/up_nei/middle_up_nei/yolo_dataset/test/images";
    int count = 1;
   
    for(const auto& entry: fs::directory_iterator(rootDir)){
        double startTime = cv::getTickCount();
        // auto start = chrono::high_resolution_clock::now();
        string filePath = entry.path().string();
        // cout << filePath << endl;
        cv::Mat src = cv::imread(filePath);
        detector.predict(src);
        count++;
        double endTime = cv::getTickCount();
        double duration = ((endTime-startTime)/cv::getTickFrequency())*1000;
        // auto end = chrono::high_resolution_clock::now();
        // chrono::duration<double,milli> duration = end -start;
        // cout << "pred use time:" << duration.count() << " ms" << endl;
        cout << "predict use time: " << duration << endl;
    }
    
    cv::destroyAllWindows();
    cout << count << endl;

}




