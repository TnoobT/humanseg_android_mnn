#include "human_seg.hpp"


bool HumanSegment::hasGPU = false;
bool HumanSegment::toUseGPU = false;
HumanSegment *HumanSegment::detector = nullptr;

HumanSegment::HumanSegment(const std::string &mnn_path, bool useGPU)
{
    toUseGPU = hasGPU && useGPU;
    net_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    backend_config_.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;  // 精度
    backend_config_.power = MNN::BackendConfig::Power_Normal; // 功耗
    backend_config_.memory = MNN::BackendConfig::Memory_Normal; // 内存占用
    config_.backendConfig = &backend_config_;
    config_.numThread = 2;
    if (useGPU) {
        config_.type = MNN_FORWARD_OPENCL;
    }
    config_.backupType = MNN_FORWARD_CPU;

    MNN::CV::ImageProcess::Config img_config; // 图像处理
    ::memcpy(img_config.mean, mean_vals_, sizeof(mean_vals_)); // (img - mean)*norm
    ::memcpy(img_config.normal, norm_vals_, sizeof(norm_vals_));
    img_config.sourceFormat = MNN::CV::BGR;
    img_config.destFormat = MNN::CV::BGR;
    pretreat_ = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_config));
    MNN::CV::Matrix trans;
    trans.setScale(1.0f, 1.0f); // scale
    pretreat_->setMatrix(trans);

    session_ = net_->createSession(config_); //创建session
    inTensor_ = net_->getSessionInput(session_, NULL);
    net_->resizeTensor(inTensor_, {1, 3, input_h_, input_w_});
    net_->resizeSession(session_);

}


HumanSegment::~HumanSegment()
{
    net_->releaseModel();
    net_->releaseSession(session_);
}



cv::Mat HumanSegment::Decode(cv::Mat& srcimg,int* res) const
{
    // 用float取mask里面的值会是一个很小的值e-45，需要用int取
    cv::Mat mask(input_h_, input_w_, CV_32FC1, res);
    cv::Mat segmentation_map;
    cv::resize(mask, segmentation_map, cv::Size(srcimg.cols, srcimg.rows),0,0,cv::INTER_CUBIC);
    //开操作（先腐蚀再膨胀）
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    // morphologyEx(segmentation_map, segmentation_map,cv::MORPH_OPEN, kernel);
    cv::Mat dstimg = srcimg.clone();

    //**矩阵相乘**// 5ms
    cv::threshold(segmentation_map, segmentation_map, 0, 1, cv::THRESH_BINARY); // 浮点型，大于阈值0，则置为1，否则为0
    cv::Mat bg(segmentation_map.rows,segmentation_map.cols,CV_32FC3, cv::Scalar(0.2, 0.2, 0.2));
    cv::cvtColor(segmentation_map,segmentation_map,cv::COLOR_GRAY2RGB);
    dstimg.convertTo(dstimg, CV_32FC3, 1 / 255.0);
    segmentation_map = segmentation_map + bg;
    dstimg = dstimg.mul(segmentation_map);
    dstimg.convertTo(dstimg,CV_8UC3,255);
    //**矩阵相乘**//

    return dstimg;
}


cv::Mat HumanSegment::Inference(cv::Mat& img) const
{
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    // 一：数据预处理
    cv::Mat preImage = img.clone();
    cv::resize(preImage,preImage,cv::Size(input_w_,input_h_));
    pretreat_->convert(preImage.data, input_w_, input_h_, 0, inTensor_);
    // 二：推理
    net_->runSession(session_);
    MNN::Tensor *output= net_->getSessionOutput(session_, NULL);
    // 三：得到结果指针
    MNN::Tensor tensor_scores_host(output, output->getDimensionType());
    output->copyToHostTensor(&tensor_scores_host);
    int* res = output->host<int>();  // !!! 注意，返回的不是0就是1，所以是整形
    // 四：解码
    auto masked_img = Decode(img,res);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();

    return masked_img;

}