/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1: Background Subtraction - Unix version
 *	Lab1.1.1AVSA2020.cpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */


#include <iostream> //system libraries
#include <sstream> //for stringstream
#include <opencv2/opencv.hpp> //opencv libraries
//Header ShowManyImages
#include "ShowManyImages.hpp"
//Header fseg
#include "fgseg.hpp"
//namespaces
using namespace std; //to avoid using'std' to declare std functions and variables (e.g., std::out -> out)
using namespace cv;  //to avoid using 'cv' to declare OpenCV functions and variables (e.g., cv::Mat -> Mat)

// using namespace fgseg;

#define _CONSOLE_DEBUG 1 // flag for showing console debug messages (old style ;) )

int main(int argc, char ** argv)
{

	double t, acum_t; //variables for execution time
	int t_freq = getTickFrequency();

    bool rgb = false;
    string dataset_path = "./dataset_lab3"; //SET THIS DIRECTORY according to your download
    string baseline_seq[5] = {"lab3.1","lab3.2","lab3.3"};
    string image_path = "/singleball.mp4"; //path to images - this format allows to read consecutive images with filename inXXXXXX.jpq (six digits) starting with 000001

    //SET THIS DIRECTORY according to your project
    string project_root_path = "./AVSA2020results";
    string project_name = "Lab3.1AVSA2020";
	string results_path = project_root_path+"/"+project_name+"/results";

	// create directory to store results
	string makedir_cmd = "mkdir -p "+project_root_path+"/"+project_name;
	system(makedir_cmd.c_str()); //may raise error in console if path exists, but will work ...
	makedir_cmd = "mkdir -p "+results_path;
	system(makedir_cmd.c_str()); //may raise error in console if path exists, but will work ...

	std::vector<cv::Point> predicPos;
	std::vector<cv::Point> predicPosCorrected;

    int NumCat = sizeof(baseline_seq)/sizeof(baseline_seq[0]);

	//Loop for all categories
	for (int c=0; c<NumCat; c++ )
	{
		// create directory to store results for category
        string makedir_cmd = "mkdir -p "+results_path ;
		system(makedir_cmd.c_str()); //may raise error in console if path exists, but will work ...
		int NumSeq = sizeof(baseline_seq)/sizeof(baseline_seq[0]);  //number of sequences per category ((have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

		//Loop for all sequence of each category
		for (int s=0; s<NumSeq; s++ )
		{
			VideoCapture cap; //reader to grab videoframes
			// Compose full path of images
            string inputvideo = dataset_path + "/" + baseline_seq[s] +image_path;
			cout << "Accessing sequence at " << inputvideo << endl;
			//open the video file to check if it exists
			cap.open(inputvideo);
			if (!cap.isOpened()) {
				cout << "Could not open video file " << inputvideo << endl;
			return -1;
			}

			// create directory to store results for sequence
            string makedir_cmd = "mkdir -p "+results_path + "/" + baseline_seq[s];
			system(makedir_cmd.c_str()); //may raise error in console if path exists, but will work ...

			//background subtraction parameters
            double tau = 128; // to set ...
            double alpha=0.1; // to set ...
            int size = 3; // structuring elment size for morphological operation

            fgseg::bgs avsa_bgs(tau, alpha ); //construct object of the bgs class

			// Main loop
			Mat img; // current Frame
            // background subtraction using MOG2
            Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();

            // kalman filter
            int stateSize = 4;
            int measSize = 4;
            int contrSize = 4;
            unsigned int type = CV_32F;
            cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

            cv::Mat state(stateSize, 1, type);  // state
            cv::Mat meas(measSize, 1, type);   // measurement
			avsa_bgs.initializeKalman(kf,stateSize, measSize, type);
			// meas = kf.measurementMatrix;
            bool found = false;
			int foundCount = 0;
            double ticks = 0;

			int it=1; // iteration
			acum_t=0; // time;
			Mat bg0;  // initial background;
			for (;;) {

				//get frame
				cap >> img;

                                 //check if we achieved the end of the file (e.g. img.data is empty)
				if (!img.data)
					break;

				// it=1 => initialize bkg
				 if (it==1)
				 	 {
					 avsa_bgs.init_bkg(img);
					 Mat aux=avsa_bgs.getBG();
					 aux.copyTo(bg0); // to visualize the first background (for analyzing results visually)
				 }

		   		//Time measurement
           		t = (double)getTickCount();

           		//Apply your bgs algorithm
		        //...
                // GET all the values of the images.
                Mat bgsmask=avsa_bgs.getBGSmask();

                double learningRate = 0.0001;
                pBackSub->apply(img,bgsmask,learningRate);

                //threshold the grayscale output of mog2

                threshold(bgsmask, bgsmask, tau, 255, cv::THRESH_BINARY);
                avsa_bgs.Morphology_Operations(size, bgsmask);
		        //Time measurement
		        t = (double)getTickCount() - t;
//		        if (_CONSOLE_DEBUG) cout << "bgs_seg = " << 1000*t/t_freq << " milliseconds."<< endl;
		        acum_t=+t;

                //extract blobs
                found = avsa_bgs.extractBlob( bgsmask, img);
        		//in a mosaic (images are resized!)

                double precTick = ticks;
                ticks = (double) cv::getTickCount();
                double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
				vector <cv::Point> centers = avsa_bgs.getBalls();

				if(centers.size() != 0){
					vector <cv::Point> cord = avsa_bgs.getBallCord();

					meas.at<float>(0) = centers[centers.size()-1].x ;
		            meas.at<float>(1) = centers[centers.size()-1].y ;
					meas.at<float>(2) = cord[cord.size()-1].x;
		            meas.at<float>(3) = cord[cord.size()-1].y;
				}

                if (found)
                {
					if (foundCount==0){
						// >>>> Initialization for the first detection
						kf.errorCovPre.at<float>(0) = 25; // px
						kf.errorCovPre.at<float>(5) = 10; // px
						kf.errorCovPre.at<float>(10) = 25;
						kf.errorCovPre.at<float>(15) = 10; // px

						state.at<float>(0) = meas.at<float>(0);
						state.at<float>(2) = meas.at<float>(1);
						state.at<float>(1) = meas.at<float>(2);
						state.at<float>(3) = meas.at<float>(3);
						// <<<< Initialization

						kf.statePost = state;
					}else{
						//-------------Matrix A--------------
	                    kf.transitionMatrix.at<float>(2) = dT;
	                    kf.transitionMatrix.at<float>(11) = dT;
	                    //------------ Matrix A--------------
	                	state = kf.predict();
	                    cv::Point center;
	                    center.x = (int)state.at<float>(0);
	                    center.y = (int)state.at<float>(2);
						predicPosCorrected.push_back(center);
						cv::putText(img, "Corrected", center,
						            cv::FONT_HERSHEY_DUPLEX, 1.0,
									CV_RGB(0, 0, 255),  1);
					}

					// std::cout << "till this point works fine" << "\n";
					foundCount++;
                }else{

						if(foundCount>0){
							state = kf.predict();
		                    cv::Point center=cv::Point(state.at<float>(0),state.at<float>(2));

							predicPos.push_back(center);

							meas.at<float>(0) = state.at<float>(0);
				            meas.at<float>(1) = state.at<float>(2);
							meas.at<float>(2) = state.at<float>(1);
				            meas.at<float>(3) = state.at<float>(3);
							cv::putText(img, "Predicted", center,
							            cv::FONT_HERSHEY_DUPLEX, 1.0,
										CV_RGB(0, 255, 0),  1);
						}
				}
				if(foundCount>1){
					kf.correct(meas);
				}
				for (int i =0; i < (int)centers.size();i++){
					cv::circle(img, centers[i], 8, CV_RGB(0,0,255));
				}
				for (int i =0; i < (int)predicPos.size();i++){
					cv::circle(img, predicPos[i], 6, CV_RGB(0,255,0), -1);
				}
				for (int i =0; i < (int)predicPosCorrected.size();i++){
					cv::circle(img, predicPosCorrected[i], 6, CV_RGB(255,0,0), -1);
				}

		        stringstream it_ss;
		      	it_ss << setw(6) << setfill('0') << it;

		      	string color_flag;
		      	if (rgb) color_flag= "color";
                else color_flag = "black and white";
                string title= project_name + " | Frame - Bg - Bg0  || Diff - FgM - | (" + baseline_seq[s] + " " + color_flag + ")";
                ShowManyImages(title, 4, img, bg0, bgsmask, img);
                //save results of your bgs algorithm
		        //...
                string outFile=results_path + "/" + baseline_seq[s] + "/"  +  "out"+ it_ss.str() +".png";

//		       	if (_CONSOLE_DEBUG){cout << outFile << endl;}
		        bool write_result=false;

                        write_result=imwrite(outFile, img);
		        if (!write_result) printf("ERROR: Can't save fg mask.\n");
				//exit if ESC key is pressed
				if(waitKey(30) == 27) break;
				it++;
			} //main loop
			cout << it-1 << "frames processed in " << 1000*acum_t/t_freq << " milliseconds."<< endl;

			//release all resources

			cap.release();
			destroyAllWindows();
			waitKey(0); // (should stop till any key is pressed .. doesn't!!!!!)
		}


	}

	return 0;
}
