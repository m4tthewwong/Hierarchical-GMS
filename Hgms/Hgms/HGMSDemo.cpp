//---------------------------------------------------------------------------
// HGMSDemo.cpp
// Driver code for testing the HGMS algorithm and associated class files
// Performs the following operations:
//	1. Find matches by local affine transformation
// Authors:  Brennan O’Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- Source image for running demo
//  -- If running the VGG benchmarks, then update the VGG_ROOT_FOLDER_PATH
//     to the VGGIMages https://www.robots.ox.ac.uk/~vgg/research/affine/
//  -- The structure of the VGG files must look like the following
//  -- VGGImages
//  --  |
//  --  |-> bark
//  --       |
//  --       |-> H1to2p, H1to3p, H1to4p, H1to5p, H1to6p
//  --       |-> img1.ppm, img2.ppm, img3.ppm, img4.ppm, img5.ppm, img6.ppm
// 
// Outputs:
// -- Depending on the RUN constants values, the Unit tests will execute,
//    the benchmark tests will execute, and the HGMS pipeline demo will run.
// 
// Description:
//    The purpose of this driver file is to assist with testing of individual
//    HGMS classes and functions.
//
// Assumptions:
//   -- At least one of the RUN flags is enabled below
//   -- SOURCE_IMAGE and DEST_IMAGE variables are updated with the names
//   -- of other images to test HGMS.

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "HGMSUnitTests.h"
#include "VGGBenchmarkRunner.h"

// Constants for determining if unit tests and benchmark tests are run
// benchmark tests require valid path to VGG affine image/homography dataset
// setting a flag to true will run that specific set of tests or demo. To 
// only see the demo just leave RUN_PIPELINE_DEMO set to true and verify
// the images are in the same folder as the directory
const bool RUN_UNIT_TESTS = false;
const bool RUN_BENCHMARK_TESTS = false;
const bool RUN_PIPELINE_DEMO = true;

// Name of input source images to run with demo. Update these two variables
// with specific source/dest images to run through HGMS
const string SOURCE_IMAGE = "lambo1.jpg";
const string DEST_IMAGE = "lambo1-zoom.jpg";

// Folder path for VGG images if running benchmarks
// must be set to valid path if running benchmarks
const string VGG_ROOT_FOLDER_PATH = "C:\\Personal\\School\\UWB\\587\\FinalSource\\VGGImages";

// Number of feature matches to calculate for running benchmarks and demo
const int MAX_FEATURES = 10000;

// Max distance between predicted target and actual target to consider valid match
const float BENCHMARK_EVAL_THRESHOLD = 3.0;

// Constants for window naming
const string NO_HGMS_WINDOW_NAME = "All Matches No Filter";
const string GMS_WINDOW_NAME = "GMS Filtered Matches";
const string INPUT_IMAGE_WINDOW_NAME = "Input Images";

// const for image size scale
const float DISPLAY_SCALE = 2;

/*----------------------------- isValidImage --------------------------
 * Helper method to validate that the image provided is a valid image.
 * Preconditions: Image reference is provided as input
 * Postconditions: A true or false is returned with true = valid image and false
 *                 indicating invalid image.
 */
const bool isValidImage(const Mat& image);

/*----------------------------- executeStagePipeline --------------------------
 * Helper method to validate that the image provided is a valid image.
 * Preconditions: Image reference is provided as input
 * Postconditions: A true or false is returned with true = valid image and false
 *                 indicating invalid image.
 */
void executeStagePipeline(const string windowName, HGMSPipeline pipeline, const string image1, const string image2);

/*----------------------------- drawFilteredMatches --------------------------
 * Helper method to draw and display image pairs and their feature matches
 * Preconditions: Valid references to source/target image matches, keypoints, filtered matches
 *                window name for display, and coordinates to move the window to so multiple
 *                windows can be displayed together
 * Postconditions: Images are displayed to screen with feature matches drawn
 */
void drawFilteredMatches(const Mat imageMat1, const vector<KeyPoint> k1kp, const Mat imageMat2,
	const vector<KeyPoint> k2kp, const vector<DMatch> filteredMatches, const string windowName,
	const int moveX, const int moveY);

/*----------------------------- videoDemo --------------------------
 * Helper method to display video and HGMS match real-time.
 * The method below is borrowed the opencv_contrib\modules\xfeatures2d\samples\gms_matcher.cpp
 * class with modifications to use HGMS instead of GMS. Also there has been some
 * code removal to make this a method versus run from commandline
 * Preconditions: HGMSPipeline with all stages registered and camera ready computer
 * Postconditions: Live video with HGMS matching enabled and new images to reset
 *                 the source image for matching against video can be captured using the R key.
 *                 escape to stop the output.
 */
void videoDemo(HGMSPipeline pipeline);


/*----------------------------- main --------------------------
 * Main method to check which functions to run based on the RUN flags
 * The main method is also where the file names are set to load in the demo
 * Preconditions: RUN flags have been set to true for those functions to run. Valid
 *                images are in the same folder as the exec and the image file names
 *                
 * Postconditions: Live video with HGMS matching enabled and new images to reset
 *                 the source image for matching against video can be captured using the R key.
 *                 escape to stop the output.
 */
int main()
{
	// run unit tests and benchmark tests based on setting
	if (RUN_UNIT_TESTS)
	{
		HGMSUnitTests unitTests;
		unitTests.runUnitTests();
	}

	if (RUN_BENCHMARK_TESTS)
	{
		VGGBenchmarkRunner vggRunner;
		// run for all affine covariant folders
		string barkFolderPath = VGG_ROOT_FOLDER_PATH + "\\bark";
		vggRunner.run("Bark", barkFolderPath, MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		//vggRunner.printResults();
		cout << "------------------------" << std::endl;
		cout << "Bark Image Dataset Averages (Precision, Recall, FScore):" << std::endl;
		vggRunner.printCsvSummaryResults();

		string bikesFolderPath = VGG_ROOT_FOLDER_PATH + "\\bikes";
		vggRunner.run("Bikes", bikesFolderPath, MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		//vggRunner.printResults();
		cout << "------------------------" << std::endl;
		cout << "Bikes Image Dataset Averages (Precision, Recall, FScore):" << std::endl;
		vggRunner.printCsvSummaryResults();

		string grafFolderPath = VGG_ROOT_FOLDER_PATH + "\\graf";
		vggRunner.run("Graf", grafFolderPath, MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		//vggRunner.printResults();
		cout << "------------------------" << std::endl;
		cout << "Graf Image Dataset Averages (Precision, Recall, FScore):" << std::endl;
		vggRunner.printCsvSummaryResults();

		string leuvenFolderPath = VGG_ROOT_FOLDER_PATH + "\\leuven";
		vggRunner.run("Leuven", leuvenFolderPath, MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		//vggRunner.printResults();
		cout << "------------------------" << std::endl;
		cout << "Leuven Image Dataset Averages (Precision, Recall, FScore):" << std::endl;
		vggRunner.printCsvSummaryResults();

		string treesFolderPath = VGG_ROOT_FOLDER_PATH + "\\trees";
		vggRunner.run("Trees", treesFolderPath, MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		//vggRunner.printResults();
		cout << "------------------------" << std::endl;
		cout << "Trees Image Dataset Averages (Precision, Recall, FScore):" << std::endl;
		vggRunner.printCsvSummaryResults();

		string ubcFolderPath = VGG_ROOT_FOLDER_PATH + "\\ubc";
		vggRunner.run("UBC", ubcFolderPath, MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		//vggRunner.printResults();
		cout << "------------------------" << std::endl;
		cout << "UBC Image Dataset Averages (Precision, Recall, FScore):" << std::endl;
		vggRunner.printCsvSummaryResults();

		string wallFolderPath = VGG_ROOT_FOLDER_PATH + "\\wall";
		vggRunner.run("Wall", wallFolderPath, MAX_FEATURES, BENCHMARK_EVAL_THRESHOLD);
		//vggRunner.printResults();
		cout << "------------------------" << std::endl;
		cout << "Wall Image Dataset Averages (Precision, Recall, FScore):" << std::endl;
		vggRunner.printCsvSummaryResults();
	}

	if (RUN_PIPELINE_DEMO)
	{
		// run demo of HGMS with all stages registered - Dog images
		HGMSPipeline allStagePipeline(HGMSPipeline::FILTER);
		allStagePipeline.addStage(std::make_shared<HGMSStage>());
		allStagePipeline.addStage(std::make_shared<LATStage>());
		allStagePipeline.addStage(std::make_shared<MPStage>());

		// run demo of HGMS with all stages registered - Lambo
		executeStagePipeline("HGMS Filtered Matches", allStagePipeline, SOURCE_IMAGE, DEST_IMAGE);

		// run video demo
		videoDemo(allStagePipeline);
	}

	return EXIT_SUCCESS;
}

/*----------------------------- isValidImage --------------------------
 * Helper method to validate that the image provided is a valid image.
 * Preconditions: Image reference is provided as input
 * Postconditions: A true or false is returned with true = valid image and false
 *                 indicating invalid image.
 */
const bool isValidImage(const Mat& image)
{
	// check that the image is not empty
	return (!image.empty());
}

/*----------------------------- executeStagePipeline --------------------------
 * Helper method to validate that the image provided is a valid image.
 * Preconditions: Image reference is provided as input
 * Postconditions: A true or false is returned with true = valid image and false
 *                 indicating invalid image.
 */
void executeStagePipeline(const string windowName, HGMSPipeline pipeline, const string image1, const string image2)
{
	const Mat imageMat1 = imread(image1);
	const Mat imageMat2 = imread(image2);

	// check for valid images
	if (!isValidImage(imageMat1) || !isValidImage(imageMat2))
	{
		std::cerr << "Unable to load images. Please check that VGG images exist in the"
			<< "specified filepath\r\n";
		return;
	}

	// create keypoints and descriptors
	std::vector<KeyPoint> k1kp, k2kp;
	Mat k1d, k2d;

	// create features
	Ptr<Feature2D> k1detector = ORB::create(MAX_FEATURES);
	k1detector->detectAndCompute(imageMat1, noArray(), k1kp, k1d);
	Ptr<Feature2D> k2detector = ORB::create(MAX_FEATURES);
	k2detector->detectAndCompute(imageMat2, noArray(), k2kp, k2d);

	// compute matches
	// Create brute-force matcher to compute matches between descriptors
	Ptr<BFMatcher> matcher = BFMatcher::create(cv::NORM_HAMMING, false);
	std::vector<DMatch> matchesAll;
	matcher->match(k1d, k2d, matchesAll);

	// execute stage with image parameters
	vector<DMatch> hgmsFilteredMatches;
	pipeline.match(k1kp, imageMat1.size(), k2kp, imageMat2.size(), matchesAll, hgmsFilteredMatches, 6.0f);

	// get GMS filtered matches
	vector<DMatch> gmsFilteredMatches;
	xfeatures2d::matchGMS(imageMat1.size(), imageMat2.size(), k1kp, k2kp, matchesAll, gmsFilteredMatches);

	// get metrics
	ExecutionMetrics execMetrics = pipeline.getExecMetrics();

	// draw input images
	Mat concatImages;
	hconcat(imageMat1, imageMat2, concatImages);
	namedWindow(INPUT_IMAGE_WINDOW_NAME, WINDOW_NORMAL);
	resizeWindow(INPUT_IMAGE_WINDOW_NAME, concatImages.cols / DISPLAY_SCALE, concatImages.rows / DISPLAY_SCALE);
	imshow(INPUT_IMAGE_WINDOW_NAME, concatImages);
	moveWindow(INPUT_IMAGE_WINDOW_NAME, 10, 10);

	// draw no filtered matches
	drawFilteredMatches(imageMat1, k1kp, imageMat2, k2kp, matchesAll, NO_HGMS_WINDOW_NAME, concatImages.cols, 0);

	// draw GMS matches
	drawFilteredMatches(imageMat1, k1kp, imageMat2, k2kp, gmsFilteredMatches, GMS_WINDOW_NAME, 10, concatImages.rows);

	// draw HGMS matches
	drawFilteredMatches(imageMat1, k1kp, imageMat2, k2kp, hgmsFilteredMatches, windowName, concatImages.cols, concatImages.rows);

	// wait for input
	waitKey();

	// clean-up
	destroyAllWindows();

	// print metrics to screen
	cout << execMetrics;
}

/*----------------------------- drawFilteredMatches --------------------------
 * Helper method to draw and display image pairs and their feature matches
 * Preconditions: Valid references to source/target image matches, keypoints, filtered matches
 *                window name for display, and coordinates to move the window to so multiple
 *                windows can be displayed together
 * Postconditions: Images are displayed to screen with feature matches drawn
 */
void drawFilteredMatches(const Mat imageMat1, const vector<KeyPoint> k1kp, const Mat imageMat2,
	const vector<KeyPoint> k2kp, const vector<DMatch> filteredMatches, const string windowName,
	const int moveX, const int moveY)
{
	Mat filteredMatchesImage;
	drawMatches(imageMat1, k1kp, imageMat2, k2kp, filteredMatches, filteredMatchesImage);
	// resize and display matches
	namedWindow(windowName, WINDOW_NORMAL);
	resizeWindow(windowName, filteredMatchesImage.cols / DISPLAY_SCALE, filteredMatchesImage.rows / DISPLAY_SCALE);
	imshow(windowName, filteredMatchesImage);
	moveWindow(windowName, moveX / DISPLAY_SCALE, moveY / DISPLAY_SCALE);
}

/*----------------------------- videoDemo --------------------------
 * Helper method to display video and HGMS match real-time.
 * The method below is borrowed the opencv_contrib\modules\xfeatures2d\samples\gms_matcher.cpp
 * class with modifications to use HGMS instead of GMS. Also there has been some
 * code removal to make this a method versus run from commandline
 * Preconditions: HGMSPipeline with all stages registered and camera ready computer
 * Postconditions: Live video with HGMS matching enabled and new images to reset 
 *                 the source image for matching against video can be captured using the R key.
 *                 escape to stop the output.
 */
void videoDemo(HGMSPipeline pipeline)
{
	// CODE BELOW IS FROM opencv_contrib\modules\xfeatures2d\samples\gms_matcher.cpp
	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb.dynamicCast<cv::ORB>()->setFastThreshold(20);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	{
		std::vector<KeyPoint> kpRef;
		Mat descRef;

		VideoCapture capture(0);
		//Camera warm-up
		for (int i = 0; i < 10; i++)
		{
			Mat frame;
			capture >> frame;
		}

		Mat frameRef;
		bool simpleDraw = false;

		for (;;)
		{
			Mat frame;
			capture >> frame;

			if (frameRef.empty())
			{
				frame.copyTo(frameRef);
				orb->detectAndCompute(frameRef, noArray(), kpRef, descRef);
			}

			TickMeter tm;
			tm.start();
			std::vector<KeyPoint> kp;
			Mat desc;
			orb->detectAndCompute(frame, noArray(), kp, desc);
			tm.stop();
			double t_orb = tm.getTimeMilli();

			tm.reset();
			tm.start();
			std::vector<DMatch> matchesAll, matchesHGMS;
			matcher->match(desc, descRef, matchesAll);
			tm.stop();
			double t_match = tm.getTimeMilli();
			
			// NOTE: pipeline.match is calling HGMS and the call to GMS is removed
			pipeline.match(kp, frame.size(), kpRef, frame.size(), matchesAll, matchesHGMS, 6.0f);

			tm.stop();
			Mat frameMatches;
			drawMatches(frame, kp, frameRef, kpRef, matchesHGMS, frameMatches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			String label = format("ORB: %.2f ms", t_orb);
			putText(frameMatches, label, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
			label = format("Matching: %.2f ms", t_match);
			putText(frameMatches, label, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
			label = format("HGMS matching: %.2f ms", tm.getTimeMilli());
			putText(frameMatches, label, Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
			putText(frameMatches, "Press r to reinitialize the reference image.", Point(frameMatches.cols - 380, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
			putText(frameMatches, "Press esc to quit.", Point(frameMatches.cols - 180, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

			imshow("Matches HGMS", frameMatches);
			int c = waitKey(30);
			if (c == 27)
				break;
			else if (c == 'r')
			{
				frame.copyTo(frameRef);
				orb->detectAndCompute(frameRef, noArray(), kpRef, descRef);
			}
		}

	}
}