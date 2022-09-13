#ifndef __HUMANSEG_H__
#define __HUMANSEG_H__

#include <android/log.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> // 图像处理库，如resisz
#include <opencv2/highgui.hpp> // 可视化
#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"

#ifndef LOG_TAG
#define LOG_TAG "tfj"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif


class HumanSegment
{

public:
    HumanSegment(const std::string &mnn_path, bool useGPU);
    ~HumanSegment();
    cv::Mat Inference(cv::Mat& img) const;
    static bool hasGPU;
    static bool toUseGPU;
    static HumanSegment *detector;

private:
    cv::Mat Decode(cv::Mat& srcimg,int* res) const;
    int input_w_ = 192;
    int input_h_ = 192;
    std::shared_ptr<MNN::Interpreter> net_ = nullptr;
    MNN::ScheduleConfig config_;
    MNN::Session *session_ = nullptr;
    MNN::Tensor *inTensor_ = nullptr;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat_ = nullptr;
    MNN::BackendConfig backend_config_;

    const float mean_vals_[3] = { 127.5f, 127.5f, 127.5f };
    const float norm_vals_[3] = { 1/127.5f, 1/127.5f, 1/127.5f }; // (-1,1)


};



#endif