/**
* This is the main file containing the code to compute the dense trajectory given a video as input.
* The optical flow is computed using the opencv GPU version of farneback optical flow.
* Note: Here we have computed only the histogram of optical flow field features. However the original
*       CPU code computes all the features i.e. Histogram of oriented gradients, Histogram of optical
*       flow, and Motion boundary descriptor.

* @auther: Aditya Vora

*/

#include "DenseTrack.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "opticalflow.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <vector>

/// Include the namespaces
using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace myopticalflow;

/// Make 1 in order to visualize the dense trajectories.
int show_track = 1;


int main(int argc, char** argv){
    /// Set the time counter.
	double t = (double)getTickCount();
	VideoCapture capture; /// Video file capture.

	/// Make a object to find the farneback optical flow.
	FarnebackOpticalFlow d_calc;

    // Size S = Size(320,240);

	/// Read the video into Video capture object.
	int frame_num = 0;
	double total_time = 0;
	char* video = argv[1];
	capture.open(video);
	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}
	if(show_track == 1)
		namedWindow("DenseTrack", 0);

	/// Track Information Initialization.
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;
	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	/// Declare the matrices to store the variables.
	Mat frame, frame0_rgb, frame1_rgb, frame0, frame1;

	/// Declare the size of the pyramid scales.
	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	/// For optical flow
	std::vector<Mat> frame0_pyr(0), frame1_pyr(0), flow_pyr(0);


	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; /// indicate when to detect new feature points
	while(true) {
 		int i, j;
		/// Get a frame from the video.
		capture >> frame;

		if(frame.empty())
			break;
        // cv::resize(frame, frame, S);
		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

		/// For first frame
		if(frame_num == start_frame) {
			frame0_rgb.create(frame.size(), CV_8UC3);
			frame1_rgb.create(frame.size(), CV_8UC3);
			frame1.create(frame.size(), CV_8UC1);
			frame0.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, frame0_pyr);
			BuildPry(sizes, CV_8UC1, frame1_pyr);

			BuildPry(sizes, CV_32FC2, flow_pyr);

			xyScaleTracks.resize(scale_num);

			frame.copyTo(frame0_rgb);
			cvtColor(frame0_rgb, frame0, CV_BGR2GRAY);

			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
					frame0.copyTo(frame0_pyr[0]);
				else
					resize(frame0_pyr[iScale-1], frame0_pyr[iScale], frame0_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				/// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(frame0_pyr[iScale], points, quality, min_distance);

				/// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));

			}

			frame_num++;

			continue;
		}

		/// For remaining frames
		init_counter++;
		frame.copyTo(frame1_rgb);
		cvtColor(frame1_rgb, frame1, CV_BGR2GRAY);
		for(int iScale = 0; iScale < scale_num; iScale++){
			if(iScale == 0)
				frame1.copyTo(frame1_pyr[0]);
			else
				resize(frame1_pyr[iScale-1], frame1_pyr[iScale], frame1_pyr[iScale].size(), 0, 0, INTER_LINEAR);
		}

        /// Compute the farneback optical flow and store into flow pyramid.
		myopticalflow::calcFarnebackOpticalFlow(d_calc, frame0_pyr, frame1_pyr, flow_pyr);

		for(int iScale = 0; iScale < scale_num; iScale++) {
			int width = frame1_pyr[iScale].cols;
			int height = frame1_pyr[iScale].rows;

			/// compute the integral histograms of optical flow pyramid
			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_pyr[iScale], hofMat->desc, hofInfo);
			/// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];

			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];

				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				/// get the descriptors for the feature point
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hofInfo);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				iTrack->addPoint(point);

				/// draw the trajectories at the first scale
				if(show_track == 1 && iScale == 0)
					DrawTrack(iTrack->point, iTrack->index, fscales[iScale], frame0_rgb);

				/// if the trajectory achieves the maximal length
				if(iTrack->index >= trackInfo.length) {
					std::vector<Point2f> trajectory(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];

					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					iTrack = tracks.erase(iTrack);
					continue;
				}
				++iTrack;
			}

			ReleDescMat(hofMat);

			if(init_counter != trackInfo.gap)
				continue;

			/// detect new feature points every initGap frames
			std::vector<Point2f> points(0);
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(frame1_pyr[iScale], points, quality, min_distance);
			/// save the new feature points
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		frame1.copyTo(frame0);
		for(i = 0; i < scale_num; i++) {
			frame1_pyr[i].copyTo(frame0_pyr[i]);
		}

		frame_num++;
		if( show_track == 1 ) {
			imshow( "DenseTrack", frame0_rgb);
			if (waitKey(1) == 27)
				break;
		}
	}

	if( show_track == 1 )
		destroyWindow("DenseTrack");

	t = ((double)getTickCount() - t)/getTickFrequency();
	cout<< "Time taken: "<< t << endl;
	return 0;
}


















