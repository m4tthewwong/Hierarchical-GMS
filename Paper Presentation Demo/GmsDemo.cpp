//---------------------------------------------------------------------------
// GmsDemo.cpp
// Driver code for testing the build of opencv and opencv_contrib with GMS demo
// following operations:
//	Authors:  Brennan O'Reilly, Pranshu Bhardwaj, Matthew Wong
//---------------------------------------------------------------------------
// Inputs:
//  -- dog01.jpg and dog02.jpg files located in same folder as executable
// 
// Outputs:
// -- The source images are displayed to the user and then 3 different windwows
//    are displayed to the user showing different configurations of GMS:
//		1. GMS with no rotation or scale support enabled
//      2. GMS with rotation and scale support enabled
//	    3. GMS with scale support enabled
//   
// 
// Description:
//    This code includes calls to the OpenCV library to load two input files and then create
//    descriptors/keypoints for the input images using ORB for 10,000 features. Based on the descriptors, matches
//    are found using the brute-force matches. The matches are then filtered using different GMS configurations
//    and drawn to the screen.
//
// Assumptions:
//   -- The dog01.jpg and dog02.jpg files are located in the same folder as the executable and are named
//      Dogs01.jpg and Dogs01.jpg. The opencv_world4120d.dll is built from opencv and opencv_contrib version 4.12.0
//      and is either in the executable folder or on the system environment path
// 
//---------------------------------------------------------------------------

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;

// enum for testing different types of detectors/descriptors
enum DETECTOR_TYPE {ORB};

// Max features for feature detection
const int MAX_FEATURES = 10000;

// Window names
const String GMS_NO_RS_MATCHES_WINDOW_NAME = "GMS No Rotation or Scale Support";
const String GMS_RS_MATCHES_WINDOW_NAME = "GMS with Rotation and Scale Support";
const String GMS_NRS_MATCHES_WINDOW_NAME = "GMS with Scale Support and No Rotation";
const String GMS_VIDEO_MATCHES_WINDOW_NAME = "GMS with Video Support";

/*----------------------------- isValidImage --------------------------
 * Helper method to validate that the image provided is a valid image
 * Preconditions: Image reference is provided as input
 * Postconditions: A true or false is returned with true = valid images and false
 *                 indicating invalid images.
 */
const bool isValidImage(const Mat& image);

/*----------------------------- detectAndCompute --------------------------
 * Helper method to perform detect and compute of inbound image and generate
 * output keypoints and descriptors.
 * Preconditions: valid image reference is provided as input
 * Postconditions: vector of keypoints and mat of descriptors and bool indicating
 *                 success or failure. True = success an false = fail.
 */
const bool detectAndCompute(const DETECTOR_TYPE detectorType, const Mat& image,
	std::vector<KeyPoint>& keyPoints, Mat& descriptors);

/*----------------------------- createDetector --------------------------
 * Helper method to create detector object based on inbound type.
 * Preconditions: valid detector type from enum
 * Postconditions: Feature2d class representing the detector object generated
 */
const Ptr<Feature2D> createDetector(const DETECTOR_TYPE detectorType);

/*----------------------------- computeMatches --------------------------
 * Helper method to compute matches for images based on descriptors using Brute-force.
 * Preconditions: valid descriptors for images that are being matched for detecting matches
 * Postconditions: vector of descriptor matches
 */
const std::vector<DMatch> computeMatches(const Mat& descriptor1, const Mat& descriptor2);

/*----------------------------- gmsCreateDisplayMatches --------------------------
 * Helper method to filter matches using GMS and then display to user.
 * Preconditions: valid images, keypoints, and matches. Window name for display and
                  settings for rotation and scale support are required
 * Postconditions: Filter matches with GMS and display image to user scaled in size to fit window
 */
void gmsCreateDisplayMatches(const Mat& image1, const Mat& image2, const std::vector<KeyPoint>& kp1, const std::vector<KeyPoint>& kp2,
	const std::vector<DMatch>& matchesAll, const String windowName, const bool rotation, const bool scale);

/*----------------------------- displayMatches --------------------------
 * Helper method to display matches in image in sized window.
 * Preconditions: valid image with matches drawn on image
 * Postconditions: display image to user scaled in size to fit window
 */
void displayMatches(const Mat& imageMatches, const String windowName);

/*----------------------------- main --------------------------
 * Main orchestration method that performs match related operations on input images
 * using ORB detector/descriptor and then filtering with GMS.
 * Preconditions: dog01.jpg and dog02.jpg files exist in the code directory
 *                and are valid JPG files. If the progam is run with arguments,
 *                then the arguments will be ignored.
 * Postconditions: Images are displayed which with matches drawn between images.
 *                 If files are missing then an error message is returned to the user 
 *                 and the program exits with error code.
 */
int main()
{

	// load input images
	const Mat dog01 = imread("dog01.jpg");
	const Mat dog02 = imread("dog02.jpg");

	// display dog source images
	displayMatches(dog01, "Dog01 Image");
	displayMatches(dog02, "Dog02 Image");

	// check for valid images
	if (!isValidImage(dog01) || !isValidImage(dog02))
	{
		std::cerr << "Unable to load images. Please check that dog01.jpg and "
			<< "dog02.jpg images exist in the same folder as the executable.\r\n";
		return EXIT_FAILURE;
	}

	// Create keypoints and descriptors objects for matching data
	std::vector<KeyPoint> dog01KP, dog02KP;
	Mat dog01Desc, dog02Desc;

	// detect and compute the keypoints and descriptors
	bool img1detect = false, img2detect = false;
	img1detect = detectAndCompute(DETECTOR_TYPE::ORB, dog01, dog01KP, dog01Desc);
	img2detect = detectAndCompute(DETECTOR_TYPE::ORB, dog02, dog02KP, dog02Desc);
	
	// check for failure in detection
	if (!img1detect || !img2detect)
	{
		std::cerr << "Invalid detector type provided. "
			<< "Unable to detect and compute keypoints and descriptors with "
			<< "selected detector type.\r\n";
		return EXIT_FAILURE;
	}

	// Create brute-force matcher to compute matches between descriptors
	std::vector<DMatch> matchesAll = computeMatches(dog01Desc, dog02Desc);

	// Run GMS test with no scale or rotation support
	gmsCreateDisplayMatches(dog01, dog02, dog01KP, dog02KP, matchesAll, GMS_NO_RS_MATCHES_WINDOW_NAME, false, false);

	// Run GMS test with scale and rotation support
	gmsCreateDisplayMatches(dog01, dog02, dog01KP, dog02KP, matchesAll, GMS_RS_MATCHES_WINDOW_NAME, true, true);

	// Run GMS test with scale and no rotation support
	gmsCreateDisplayMatches(dog01, dog02, dog01KP, dog02KP, matchesAll, GMS_NRS_MATCHES_WINDOW_NAME, false, true);

	// clean-up
	destroyAllWindows();

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


/*----------------------------- detectAndCompute --------------------------
 * Helper method to perform detect and compute of inbound image and generate
 * output keypoints and descriptors.
 * Preconditions: valid image reference is provided as input
 * Postconditions: vector of keypoints and mat of descriptors and bool indicating
 *                 success or failure. True = success an false = fail.
 */ 
const bool detectAndCompute(const DETECTOR_TYPE detectorType, const Mat& image,
	std::vector<KeyPoint>& keyPoints, Mat& descriptors)
{
	// Determine type of detector to create
	Ptr<Feature2D> detector = createDetector(detectorType);
	
	// check for null
	if (!detector)
	{
		return false;
	}
	
	// Generate keypoints and descriptors using detectAndCompute
	std::cout << "Detecting keypoints for input image\r\n";
	detector->detectAndCompute(image, noArray(), keyPoints, descriptors);

	return true;
}

/*----------------------------- createDetector --------------------------
 * Helper factory method to create detector object based on inbound type.
 * Preconditions: valid detector type from enum
 * Postconditions: Feature2d class representing the detector object generated
 */
const Ptr<Feature2D> createDetector(const DETECTOR_TYPE detectorType)
{
	// create object based on enum as these all inherit from 
	// Feature 2D: ORB
	switch (detectorType)
	{
	case DETECTOR_TYPE::ORB:
		return ORB::create(MAX_FEATURES);
	default:
		return nullptr;
	}
}

/*----------------------------- computeMatches --------------------------
 * Helper method to compute matches for images based on descriptors using Brute-force.
 * Preconditions: valid descriptors for images that are being matched for detecting matches
 * Postconditions: vector of descriptor matches
 */
const std::vector<DMatch> computeMatches(const Mat& descriptor1, const Mat& descriptor2)
{
	// Create brute-force matcher to compute matches between descriptors
	Ptr<BFMatcher> matcher = BFMatcher::create(cv::NORM_HAMMING, false);
	std::vector<DMatch> matches;

	matcher->match(descriptor1, descriptor2, matches);

	return matches;
}

/*----------------------------- gmsCreateDisplayMatches --------------------------
 * Helper method to filter matches using GMS and then display to user.
 * Preconditions: valid images, keypoints, and matches. Window name for display and
				  settings for rotation and scale support are required
 * Postconditions: Filter matches with GMS and display image to user scaled in size to fit window
 */
void gmsCreateDisplayMatches(const Mat& image1, const Mat& image2, const std::vector<KeyPoint>& kp1, const std::vector<KeyPoint>& kp2,
	const std::vector<DMatch>& matchesAll, const String windowName, const bool rotation, const bool scale)
{
	// create GMS matcher to computer matches between descriptors
	std::vector<DMatch> matchesGMS;
	xfeatures2d::matchGMS(image1.size(), image1.size(), kp1, kp2, matchesAll, matchesGMS, rotation, scale);
	std::cout << "MatchGMS Size: " << matchesGMS.size() << std::endl;

	// Draw matches
	Mat imageMatches;
	drawMatches(image1, kp1, image2, kp2, matchesGMS, imageMatches);

	// Output matches to display
	displayMatches(imageMatches, windowName);

	// Wait for user input
	waitKey(0);
}

/*----------------------------- displayMatches --------------------------
 * Helper method to display matches in image in sized window.
 * Preconditions: valid image with matches drawn on image
 * Postconditions: display image to user scaled in size to fit window
 */
void displayMatches(const Mat& imageMatches, const String windowName)
{
	// create window for displaying image with ability to resize
	namedWindow(windowName, WINDOW_NORMAL);

	// resize based on scale factor
	resizeWindow(windowName, imageMatches.cols, imageMatches.rows);
	
	// show image
	imshow(windowName, imageMatches);
}
