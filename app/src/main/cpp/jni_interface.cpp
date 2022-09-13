#include <jni.h>
#include <string>

#include "human_seg.hpp"

#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/imgproc/types_c.h>

#ifndef LOG_TAG
#define LOG_TAG "WZT_MNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif

/* ======================================[ NanoDet ]======================================*/
extern "C"
JNIEXPORT void JNICALL
Java_com_example_tfjtest_model_HumanSegment_init(JNIEnv *env, jclass clazz, jstring name, jstring path, jboolean use_gpu) { // tfj, 前两项可不管，加载模型
	if (HumanSegment::detector != nullptr) {
        delete HumanSegment::detector;
        HumanSegment::detector = nullptr;
    }
    if (HumanSegment::detector == nullptr) {
        const char *pathTemp = env->GetStringUTFChars(path, 0);
        std::string modelPath = pathTemp;
        if (modelPath.empty()) {
            LOGE("model path is null");
            return;
        }
        modelPath = modelPath + env->GetStringUTFChars(name, 0);
        LOGI("model path:%s", modelPath.c_str());
        HumanSegment::detector = new HumanSegment(modelPath, use_gpu);
    }
}

extern "C"
JNIEXPORT jobject JNICALL
    Java_com_example_tfjtest_model_HumanSegment_detect(JNIEnv *env, jclass clazz, jobject bitmap, jbyteArray image_bytes, jint width, // tfj, 前两项可不管，加载模型
                                      jint height, jdouble threshold, jdouble nms_threshold) {
    jbyte *imageDate = env->GetByteArrayElements(image_bytes, nullptr);
    if (nullptr == imageDate) {
        LOGE("input image is null");
        return nullptr;
    }
    // 输入参数bitmap为Bitmap, image_bytes为bitmap的byte[]
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, bitmap, &img_size);
    if (img_size.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("input image format not support");
        return nullptr;
    }
    if (width == 333)
        LOGD("input image format not support");

    auto *dataTemp = (unsigned char *) imageDate;
    cv::Mat srcMatImg;
    cv::Mat tempMat(height, width, CV_8UC4, dataTemp); // 应该是 480*640
    cv::cvtColor(tempMat, srcMatImg, CV_RGBA2BGR);
    if (srcMatImg.channels() != 3) {
        LOGE("input image format channels != 3");
    }

    cv::Mat result = HumanSegment::detector->Inference(srcMatImg); // 获得结果，转换为java对应的类型，传给mainactivity使用

    AndroidBitmapInfo info; //保存图像参数
    void *pixels = 0;       //保存图像数据
    cv::Mat &src = result;

    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(src.dims == 2 && info.height == (uint32_t) src.rows &&
              info.width == (uint32_t) src.cols);
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);

    cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
    cv::cvtColor(result, tmp, cv::COLOR_RGB2BGRA);
    AndroidBitmap_unlockPixels(env, bitmap);
    srcMatImg.release();
    tempMat.release();
    env->ReleaseByteArrayElements(image_bytes, imageDate, 0);

//    auto line_cls = env->FindClass("com/example/tfjtest/model/LaneInfo");
//    auto cid = env->GetMethodID(line_cls, "<init>", "(FFFFFF)V");
//    jobjectArray ret = env->NewObjectArray(result.size(), line_cls, nullptr);
//    int i = 0;
//    for (auto &line:result) {
//        env->PushLocalFrame(1);
//        jobject obj = env->NewObject(line_cls, cid, line.x1, line.y1, line.x2, line.y2, line.lens, line.conf);
//        obj = env->PopLocalFrame(obj);
//        env->SetObjectArrayElement(ret, i++, obj);
//    }
    return bitmap;
}




