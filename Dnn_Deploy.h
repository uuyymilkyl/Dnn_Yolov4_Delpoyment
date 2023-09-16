#ifndef _DNN_DEPLOY_H
#define _DNN_DEPLOY_H

#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include<string>

using namespace cv;
using namespace dnn;
using namespace std;

#define DEBUGMODE 0 

constexpr auto configs = "";
constexpr auto weights = "";


struct result 
{

	Rect box;
	float score ;
	int classId ;

};

class DnnDeploy
{
public:
	DnnDeploy(Net& _net, Mat& _image);
	~DnnDeploy();

public:
	/**
	 * @brief  读取Dnn配置文件
	 * @param  _cfgpath config路径
	 * @param  _wgtpath weight路径
	 * @return Net      网络
	 * @author 黄敏瑜
	 * @date   2023.9.10
	 */
	static dnn::Net DnnReader(std::string _cfgpath, std::string _wgtpath);

	/**
	 * @brief  读取Dnn配置文件
	 * @param  _cfgpath config路径
	 * @param  _wgtpath weight路径
	 * @return Net      网络
	 * @author 黄敏瑜
	 * @date   2023.9.10
	 */
	int DnnPreprocess(dnn::Net& _net, cv::Mat& _image, int _W, int _H, int _C);

	/**
	 * @brief  获取Dnn推理结果
	 * @return vector<Mat> vmatOutputs推理结果拓扑容器
	 * @author 黄敏瑜
	 * @date   2023.9.10
	 */
	std::vector<Mat> DnnOutputs();

		/**
	 * @brief  Dnn推理结果后处理
	 * @param  _image 原图
	 * @param  confThred 置信度
	 * @return nmsThered 非极大值抑制
	 * @author 黄敏瑜
	 * @date   2023.9.10
	 */
	void DnnPostProcess(cv::Mat& _image, float& confThred, float& nmsThred);



public:
	dnn::Net net;
	cv::Mat blob;
	cv::Mat m_show_image;
	vector<int> out_Idx;
	vector<float> out_Scores;     // 分数结果
	vector<Rect> out_Boxes;       // 描框结果

	vector<int> classIds;         // 类别名

	vector<Mat> vmDetectAreaList;

};



#endif