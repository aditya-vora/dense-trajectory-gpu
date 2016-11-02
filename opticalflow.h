/**
This file contains the GPU version of the optical flow that is used in the dense trajectory algorithm.
* @auther: Aditya Vora
*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>

/// Include namespaces
using namespace std;
using namespace cv;
using namespace cv::gpu;

/// Some global parameters for optical flow computation.
const int numLevels = 1;
const bool fastPyramids = true;
const int winSize = 10;
const int numIters = 2;
const int polyN = 5; // 5 or 7
const float polySigma = 2.4;

namespace myopticalflow{
	void calcFarnebackOpticalFlow(FarnebackOpticalFlow d_calc, std::vector<Mat>& frame0_pyr, std::vector<Mat>& frame1_pyr,
                              std::vector<Mat>& flow_pyr)
	{

			GpuMat d_frame0, d_frame1, d_flowx, d_flowy; /// Initialize the variables
			Mat h_flowx, h_flowy;                       /// Variables for storing optical flow
		    /// Set the parameters
		    d_calc.numLevels = numLevels;
    		d_calc.fastPyramids = fastPyramids;
    		d_calc.winSize = winSize;
    		d_calc.numIters = numIters;
    		d_calc.polyN = polyN;
    		d_calc.polySigma = polySigma;

    		/// For every scale compute optical flow
			for(int iScale = flow_pyr.size()-1; iScale >= 0; iScale--){
				Mat flow;
				Mat channels[2];
				flow.create(frame0_pyr[iScale].size(), CV_32FC2);
				split(flow, channels);
				d_frame0.upload(frame0_pyr[iScale]);
				d_frame1.upload(frame1_pyr[iScale]);
				d_calc(d_frame0, d_frame1, d_flowx, d_flowy);
				d_flowx.download(h_flowx);
				d_flowy.download(h_flowy);
				h_flowx.copyTo(channels[0]);
				h_flowy.copyTo(channels[1]);
				merge(channels,2, flow);
				flow.copyTo(flow_pyr[iScale]);
		}

	}

}




