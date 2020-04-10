/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.hpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */
#include <opencv2/opencv.hpp>

#ifndef FGSEG_H_INCLUDE
#define FGSEG_H_INCLUDE
using namespace cv;
using namespace std;

namespace fgseg {

	//Declaration of FGSeg class based on BackGround Subtraction (bgs)
	class bgs{
        public:
            /*  ***************************************************
             *
             *  CONSTRUCTORS
             *
             *  *************************************************
             */

            //constructor with parameter "threshold"
            bgs(double threshold, bool rgb);

            //destructor
            ~bgs(void);

            /*  ***************************************************
             *
             *  HELPER FUNCTIONS
             *
             *  *************************************************
             */

            // method to initialize bkg (first frame - hot start)
            void init_bkg(cv::Mat Frame);

            // method to perform BackGroundSubtraction
            void bkgSubtraction(cv::Mat Frame);

            //method to detect and extract blobs from the binary BGS mask
            bool extractBlob(cv::Mat binaryImage, cv::Mat &outPutImage);
            void Morphology_Operations(int size, cv::Mat &binaryImage);
            void initializeKalman(cv::KalmanFilter &kf, int measSize, int stateSize, int type);
            void uplateKalman(cv::KalmanFilter &kf);
            void predictKalman(cv::KalmanFilter &kf);

            /*  ***************************************************
             *
	         *  GETTERS
	         *
	         *  ***************************************************
	         */

            //returns the BG image
            cv::Mat getBG(){return _bkg;};
            //returns the DIFF image
            cv::Mat getDiff(){return _diff;};
            //returns the BGS mask
            cv::Mat getBGSmask(){return _bgsmask;};
			vector<cv::Point > getBalls(){return balls;};
			vector<cv::Point > getBallCord(){return ballCord;};
            //ADD ADITIONAL METHODS HERE
            //...
        private:
            cv::Mat _bkg; //Background model
            cv::Mat	_frame; //current frame
            cv::Mat _diff; //abs diff frame
            cv::Mat _bgsmask; //binary image for bgssub (FG)

            bool _rgb;
            double _threshold;

			// vector to store blobs
			vector<cv::Point > balls;
			vector<cv::Point > ballCord;



            // Variable for  color images
            cv::Mat _diff_channels[3],_frame_channels[3],_bkg_channels[3],_bgsmask_channels[3]; // for the splits
            cv::Mat frame_hsv , bkg_hsv;

        };//end of class bgs
}//end of namespace

#endif
