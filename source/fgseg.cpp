/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.cpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */

#include <opencv2/opencv.hpp>
#include "fgseg.hpp"

using namespace fgseg;


// Lab 3.1 constructor - selective background update
bgs::bgs(double threshold, bool rgb ){
    _rgb = rgb ;
    _threshold = threshold;
}


//default destructor
bgs::~bgs(void)
{

}

//method to initialize bkg (first frame - hot start)
void bgs::init_bkg(cv::Mat Frame)
{
        if (!_rgb){

            cvtColor(Frame, Frame, COLOR_BGR2GRAY); // to work with gray even if input is color
           _bkg = Mat::zeros(Size(Frame.cols,Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix
        }

        Frame.copyTo(_bkg);
}

//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame)
{

        /*
         *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         *  Input : Frame ,
         *  Output : _bgsmask -> 1/255 - forgroud
         *                    -> 0     - background
         *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         *
        */
	if (!_rgb){

            _bgsmask = Mat::zeros(Size(Frame.cols,Frame.rows), CV_8UC1); // void function for Lab1.0 - returns zero matrix

            // Intialize and convert to gray scale and copy
             _diff = Mat::zeros(Size(Frame.cols,Frame.rows), CV_8UC1);
            cvtColor(Frame, Frame, COLOR_BGR2GRAY);
            Frame.copyTo(_frame);

            // calculate the difference between current frame and bgmodel
            absdiff(Frame,this->getBG(),this->_diff);

            // Threshold difference
             threshold(this->_diff,this-> _bgsmask, this->_threshold, 255, cv::THRESH_BINARY);

	}
	else{


            Frame.copyTo(this->_frame);

            // GENERAL difference between the 3 channel frame and 3 channel background model
            absdiff(this->_frame ,this-> _bkg ,this-> _diff);

            // split the difference and threshold each channel
            split(this->_diff , this->_diff_channels);
            cvtColor(this->_diff, this->_diff, COLOR_BGR2GRAY);

            // threshold every split
            threshold(this->_diff_channels[0],this-> _bgsmask_channels[0], this->_threshold, 255, cv::THRESH_BINARY); // 	BLUE
            threshold(this->_diff_channels[1],this-> _bgsmask_channels[1], this->_threshold, 255, cv::THRESH_BINARY); //  GREEN
            threshold(this->_diff_channels[2],this-> _bgsmask_channels[2], this->_threshold, 255, cv::THRESH_BINARY); //  RED

            // combine every split using bitwise_or and add them to _bgmask
            bitwise_or(this->_bgsmask_channels[0],this->_bgsmask_channels[1],this->_bgsmask);
            bitwise_or(this->_bgsmask_channels[2],this->_bgsmask, this->_bgsmask);

        }
}

bool bgs::extractBlob(cv::Mat binaryImage, cv::Mat &outPutImage){

        bool found = false;
        Mat labels;
        Mat stats;
        Mat centroids;
        cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids);


        //std::cout << labels << std::endl;
        // std::cout << "stats.size()=" << stats.size() << std::endl;
        //std::cout << centroids << std::endl;

        for(int i=0; i<stats.rows; i++){
            int x = stats.at<int>(Point(0, i));
            int y = stats.at<int>(Point(1, i));
            int w = stats.at<int>(Point(2, i));
            int h = stats.at<int>(Point(3, i));
            int radius = w/2;
            // std::cout << "x=" << x << " y=" << y << " w=" << w << " h=" << h << std::endl;
            if (w>10&&w<400 && h > 10 && h < 300){
                found = true;
                Scalar color(0,0,255);
                cv::Point center = cv::Point(x+radius,y+radius);
                cv::circle(outPutImage, center,radius, color);
                balls.push_back(center);
                ballCord.push_back(cv::Point(w,h));
            }
        }
        return found;
}


void bgs::Morphology_Operations( int size, cv::Mat &binaryImage)
{
  Mat element = getStructuringElement( cv::MORPH_RECT, Size( size, size ), Point( 1, 1 ) );
  morphologyEx( binaryImage, binaryImage, cv::MORPH_OPEN, element );
}
void bgs::initializeKalman( cv::KalmanFilter &kf, int measSize, int stateSize, int type)
{
    //
    /* state transition matrix A
    *  1 1 0 0
    *  0 1 0 0
    *  0 0 1 1
    *  0 0 0 1
    */

    cv::setIdentity(kf.transitionMatrix);
    kf.transitionMatrix.at<float>(1)  = 1.0f;
    kf.transitionMatrix.at<float>(11) = 1.0f;

    /* measurement matrix H
    * 1 0 0 0
    * 0 0 1 0
    */
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(6) = 1.0f;

    /* Process Noise Covariance Matrix Q
    * 25 0 0 0
    * 0 10 0 0
    * 0 0 25 0
    * 0 0 0 10
    */
    // cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf.processNoiseCov.at<float>(0)  = 25.0f;
    kf.processNoiseCov.at<float>(5)  = 10.0f;
    kf.processNoiseCov.at<float>(10)  = 25.0f;
    kf.processNoiseCov.at<float>(15)  = 10.0f;

    /* Measures Noise Covariance Matrix R
    * 25 0
    * 0 25
    */
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(25));
}

void bgs::uplateKalman( cv::KalmanFilter &kf)
{


}
void bgs::predictKalman( cv::KalmanFilter &kf)
{


}
