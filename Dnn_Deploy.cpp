#include "Dnn_Deploy.h"


DnnDeploy::DnnDeploy(Net& _net, cv::Mat& _image)
{
	float confThreshold = 0.6;
	float nmsThreshold = 0.2;
	if (DnnPreprocess(_net, _image, 320, 240, 3) != 0) {
		DnnPostProcess(_image, confThreshold, nmsThreshold);
	}
}

DnnDeploy::~DnnDeploy()
{
}

Net DnnDeploy::DnnReader(std::string _cfgpath, std::string _wgtpath)
{
	dnn::Net Darknet;
	Darknet = readNetFromDarknet(_cfgpath, _wgtpath);
	Darknet.setPreferableBackend(DNN_BACKEND_OPENCV);
	Darknet.setPreferableTarget(DNN_TARGET_CPU);

	return Darknet;
}

int DnnDeploy::DnnPreprocess(dnn::Net& _net, cv::Mat& _image, int _W, int _H, int _C)
{
	int imgW = _W;
	int imgH = _H;
	int imgChannels = _C;
	net = _net;

	Mat image = _image;
	if (image.empty() == true)
	{
		std::cout << " Darknet Input is Empty !" << std::endl;
		return 0;
	}
	dnn::blobFromImage(image, blob, 1 / 255.0, Size(imgW, imgH), Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	return 1;
}

std::vector<Mat> DnnDeploy::DnnOutputs()
{
	std::vector<String> vstNames;
	std::vector<Mat>    vmatOutputs;
	if (vstNames.empty())
	{
		std::vector<int> vnOutLayers = net.getUnconnectedOutLayers();
		std::vector<String> vstLayersNames = net.getLayerNames();

		vstNames.resize(vnOutLayers.size());
		for (unsigned i = 0; i < vnOutLayers.size(); i++)
		{
			vstNames[i] = vstLayersNames[vnOutLayers[i] - 1];
		}

		net.forward(vmatOutputs, vstNames);

	}
	return vmatOutputs;
}

void DnnDeploy::DnnPostProcess(cv::Mat& _image, float& confThred, float& nmsThred)
{
	vector<int> vnClassIdx;
	vector<float> vnConfidence;
	vector<Rect>  vrBoxes;

#ifdef DEBUGMODE
	Mat show_image = _image;


#endif

	vector<Mat> vmOutputs = DnnOutputs();

	for (unsigned i = 0; i < vmOutputs.size(); ++i)
	{
		float* data = (float*)vmOutputs[i].data;
		for (int j = 0; j < vmOutputs[i].rows; ++j, data += vmOutputs[i].cols)
		{
			Mat scores = vmOutputs[i].row(j).colRange(5, vmOutputs[i].cols);
			Point classIdPoint;
			double confidence;

			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThred)
			{
				int centerX = (int)(data[0] * _image.cols);
				int centerY = (int)(data[1] * _image.rows);
				int width = (int)(data[2] * _image.cols);
				int height = (int)(data[3] * _image.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				vnClassIdx.push_back(classIdPoint.x);
				vnConfidence.push_back((float)confidence);
				vrBoxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	vector<int> vnIndeces;
	NMSBoxes(vrBoxes, vnConfidence, confThred, nmsThred, vnIndeces);

	for (unsigned q = 0; q < vnIndeces.size(); ++q)
	{
		int idx = vnIndeces[q];
		Rect box = vrBoxes[idx];
		float score = vnConfidence[idx];
		int classId = vnClassIdx[idx];

		out_Idx.push_back(classId);
		out_Boxes.push_back(box);
		out_Scores.push_back(score);
	}

	if (out_Boxes.size() >= 1)
	{
		Mat image = _image.clone();
		Mat show_image = _image.clone();
		for (int p = 0; p < out_Boxes.size(); p++)
		{
			Mat cut_Area = image(out_Boxes[p]).clone();
			vmDetectAreaList.push_back(cut_Area);

			rectangle(show_image, out_Boxes[p], Scalar(0, 0, 255), 2, LINE_8, 0);
			Point draw;
			draw.x = out_Boxes[p].x;
			draw.y = out_Boxes[p].y;
			putText(show_image, "TYPE:[" + to_string(out_Idx[p]) + "]", draw, FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2);

		}
		m_show_image = show_image;
	}

}

