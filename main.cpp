// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "utils.h"

// include my project functions
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "iostream"

typedef std::vector<cv::Mat> channels;
typedef std::vector<std::vector<cv::Point>> contours;
void glcm(cv::Mat img, float &uniformity, float &contrast, float &homogeneity, float &IDM, float &entropy);
float sigmoid(double val);

namespace
{
    FILE *fp;
    int underS = 0;
    unsigned long int nGTRect = 0;
    int TP = 0;
    int BG = 0;
    contours gtContours;
    contours FOV;

#ifdef ARFF
    std::string des = ".arff";
#endif
#ifdef CSV
    std::string des = ".csv";
#endif

    //Path where to save .csv/.arff file
    std::string filePath = "./GLOB_FILE_111_NORM_GLCM_TRAIN";

}

int main()
{
    try
    {
        fp = fopen((filePath += des).c_str(), "w");

#ifdef ARFF
        //Preparing ARFF TRAINING file
        fprintf(fp, "@relation SE_OR_NOT\n");
        fprintf(fp, "@attribute LIGHTNESS numeric\n@attribute VALUE_A numeric\n@attribute VALUE_B numeric\n@attribute STDDEV_L numeric\n@attribute STDDEV_A numeric\n@attribute STDDEV_B numeric\n@attribute AREA numeric\n@attribute CIRCULARITY numeric\n@attribute EXTENT numeric\n@attribute UNIFORMITY numeric\n@attribute CONTRAST numeric\n@attribute HOMOGENEITY numeric\n@attribute IDM numeric\n@attribute ENTROPY numeric\n@attribute CLASS {SE,BACKGROUND}\n\n");
        fprintf(fp, "@data\n");
#endif
#ifdef CSV
        fprintf(fp, "LIGHTNESS,VALUEA,VALUEB,STDDEVL,STDDEVA,STDDEVB,AREA,CIRCULARITY,EXTENT,UNIFORMITY,CONTRAST,HOMOGENEITY,IDM,ENTROPY,CLASS\n");
#endif
        //Load FOV Mask and retrieve its contour
        cv::Mat FOVmask = cv::imread(std::string(RETINA_MASK) + "/mask_FOV.tif", cv::IMREAD_GRAYSCALE);
        if (!FOVmask.data)
            throw aia::error("FOV MASK NOT FOUND IN PATH");
        else
            cv::resize(FOVmask, FOVmask, cv::Size(0, 0), 0.15, 0.15, cv::INTER_AREA);
        
        cv::findContours(FOVmask, FOV, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::Rect FOV_Rect = cv::boundingRect(FOV[0]);

        for (int k = 1; k <= 81; k++)
        {
            if (k == 55) //Change some variables when entering test section
            {
                fclose(fp);
                filePath = "./GLOB_FILE_111_NORM_GLCM_TEST";
                fp = fopen((filePath += des).c_str(), "w");

#ifdef ARFF
                //Preparing ARFF TRAINING file
                fprintf(fp, "@relation SE_OR_NOT\n");
                fprintf(fp, "@attribute LIGHTNESS numeric\n@attribute VALUE_A numeric\n@attribute VALUE_B numeric\n@attribute STDDEV_L numeric\n@attribute STDDEV_A numeric\n@attribute STDDEV_B numeric\n@attribute AREA numeric\n@attribute CIRCULARITY numeric\n@attribute EXTENT numeric\n@attribute UNIFORMITY numeric\n@attribute CONTRAST numeric\n@attribute HOMOGENEITY numeric\n@attribute IDM numeric\n@attribute ENTROPY numeric\n@attribute CLASS {SE,BACKGROUND}\n\n");
                fprintf(fp, "@data\n");
#endif
#ifdef CSV
                fprintf(fp, "LIGHTNESS,VALUEA,VALUEB,STDDEVL,STDDEVA,STDDEVB,AREA,CIRCULARITY,EXTENT,UNIFORMITY,CONTRAST,HOMOGENEITY,IDM,ENTROPY,CLASS\n");
#endif
            }

            std::string imageNumber = (k < 10)? ucas::strprintf("0%d", k) : 
                                                ucas::strprintf("%d", k);

            std::string gt_path = (k <= 54)? std::string(RETINA_GROUNDTRUTH_TRAIN_SOFT_EXUDATES) : 
            			   		     	     std::string(RETINA_GROUNDTRUTH_TEST_SOFT_EXUDATES);

            std::string img_path = (k <= 54)? std::string(RETINA_IMG_PATH_TRAIN) : 
            							      std::string(RETINA_IMG_PATH_TEST);
            
            //Load Groundtruth Mask and find its contours
            cv::Mat GTmask = cv::imread(gt_path += "/IDRiD_" + imageNumber + "_SE.tif", cv::IMREAD_GRAYSCALE);
            if (GTmask.data)
            {
                cv::resize(GTmask, GTmask, cv::Size(0, 0), 0.15, 0.15, cv::INTER_AREA);
                GTmask = GTmask(FOV_Rect);
                cv::findContours(GTmask, gtContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            }
#ifdef SHOW_INTERMEDIATE
            if (GTmask.data)
            {
                cv::Mat foundedContours(GTmask.rows, GTmask.cols, CV_8U, cv::Scalar(0));
                cv::drawContours(foundedContours, gtContours, -1, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
                aia::imshow("GT Contours", foundedContours);
            }
#endif
            /** Data Set Preprocessing */
            // Load Original Image
            cv::Mat retina = cv::imread(img_path += "/IDRiD_" + imageNumber + ".jpg");
            cv::resize(retina, retina, cv::Size(0, 0), 0.15, 0.15, cv::INTER_AREA);

            // Cropping for faster computation
            retina = retina(FOV_Rect);
            cv::Mat c = retina.clone();

#ifdef SHOW_INTERMEDIATE
            aia::imshow("Retina mean sub", retina);
#endif

            // Mean Subtraction using original image
            retina -= cv::mean(retina);

            //Converting into Gray Scale for GLCM features
            cv::Mat retinaGS;
            cv::cvtColor(retina, retinaGS, cv::COLOR_BGR2GRAY);

            // Color conversion from BGR to Lab Color Space
            cv::Mat retinaLab;
            cv::cvtColor(retina, retinaLab, cv::COLOR_BGR2Lab);

            //New Color Space channels
            channels Lab; 
            cv::split(retinaLab, Lab);

            // Some aliases for better notation
            cv::Mat retinaL = Lab[0];
            cv::Mat retinaA = Lab[1];
            cv::Mat retinaB = Lab[2];

#ifdef SHOW_INTERMEDIATE
            aia::imshow("Retina L", retinaL);
            aia::imshow("Retina A", retinaA);
            aia::imshow("Retina B", retinaB);
#endif
            /** Segmentation */
            //Measuring the biggest Soft Exudate, a 41 neighbourhood window is chosen
            cv::Mat retinaLBIN;
            cv::adaptiveThreshold(retinaL,
                                  retinaLBIN,
                                  255,
                                  cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv::THRESH_BINARY,
                                  41,
                                  0);

#ifdef SHOW_INTERMEDIATE
            cv::imshow("Retina L TOPHAT BIN", retinaLBIN);
#endif
            //Morphology Opening in order to remove artifacts and retrieve better contours
            cv::morphologyEx(retinaLBIN,
                             retinaLBIN,
                             cv::MORPH_OPEN,
                             cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

#ifdef SHOW_INTERMEDIATE
            aia::imshow("Retina L TOPHAT ERODE", retinaLBIN);
#endif
            // Candidate extraction with no contours approximation
            contours candidateSE;
            cv::findContours(retinaLBIN, candidateSE, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

            /** Writing files for binary Classification: 
             * ---> BG: is a class defined for no Cotton Wool Spots 
             * ---> SE: is a class defined for Cotto Wool Spots
             * 
             * Every no GLCM feature is extracted by considering the L-Channel cropped
             * at the Candidate Object bounding rectangle
            */

            // Calculating IOU ratio for distinguish extracted objects 
            for (auto &cSE : candidateSE)
            {
                int count = 0;
                for (auto &gtCont : gtContours)
                {
                    //For each Predicted Contour and GroundTruth Contour calculating BRect
                    cv::Rect bRect  = cv::boundingRect(cSE);
                    cv::Rect gtRect = cv::boundingRect(gtCont);

                    //GLCM-based Features
                    float uniformity  = 0,
                          contrast    = 0,
                          homogeneity = 0,
                          IDM         = 0,
                          entropy     = 0;

                    glcm(retinaGS(bRect), uniformity, contrast, homogeneity, IDM, entropy);

                    //Color features
                    cv::Scalar midL,
                               midA,
                               midB,
                               stdDevL,
                               stdDevA,
                               stdDevB;

                    cv::meanStdDev(retinaL(bRect), midL, stdDevL);
                    cv::meanStdDev(retinaA(bRect), midA, stdDevA);
                    cv::meanStdDev(retinaB(bRect), midB, stdDevB);

                    //Shape features
                    double area  = cv::contourArea(cSE);
                    double p     = cv::arcLength(cSE, true);
                    double circ  = (4 * ucas::PI * area) / (p * p);
                    float extent = float(area) / float(bRect.area());

                    // Handcrafted threshold chosen by measuring the largest and the smallest SE area
                    if (area > 5 && area < 1500)
                    {
                        // Intersection Over Union
                        int rectIntersect = (gtRect & bRect).area();
                        int rectUnion = (gtRect.area() + bRect.area()) - rectIntersect;
                        float IOU = float(rectIntersect) / float(rectUnion);
#ifdef DEBUG
                        /**
                         * This section shows on the right groundtruth image,
                         * then on the left the orginal retina fundus image.
                         * If the IOU rate it's over 0.5 then a white rect is drawn
                         * on the groundtruth image, while a yellow one on the original image.
                         * On the contrary, if the rect is not being white filled, that SE is not
                         * being found.
                        **/
                        cv::rectangle(retina, gtRect, cv::Scalar(0, 255, 0));
                        cv::rectangle(GTmask, gtRect, cv::Scalar(255));
                        if (IOU > 0.5)
                        {
                            cv::rectangle(retina, gtRect & bRect, cv::Scalar(0, 255, 255));
                            cv::rectangle(GTmask, (gtRect & bRect), cv::Scalar(255), cv::FILLED);
                            aia::imshow("Retina", retina, false);
                            aia::imshow("GroundTruth", GTmask, false);
                            cv::waitKey(1000);
                        }
#endif

                        if (IOU > 0.5 && k <= 54)
                        {
                           // Print on file features vector
                            fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,SE\n",
                                    midL[0],
                                    midA[0],
                                    midB[0],
                                    stdDevL[0],
                                    stdDevA[0],
                                    stdDevB[0],
                                    area,
                                    circ,
                                    extent,
                                    uniformity,
                                    contrast,
                                    homogeneity,
                                    IDM,
                                    entropy);
                            TP++;
                        }
                        if (IOU > 0.5 && k > 54)
                        {
                            // Print on file features vector
                            fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,SE\n",
                                    midL[0],
                                    midA[0],
                                    midB[0],
                                    stdDevL[0],
                                    stdDevA[0],
                                    stdDevB[0],
                                    area,
                                    circ,
                                    extent,
                                    uniformity,
                                    contrast,
                                    homogeneity,
                                    IDM,
                                    entropy);
                            TP++;
                        }
                        if (k <= 54 && IOU < 0.5 && count == 0)
                        {
                            // Print on file features vector
                            fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,BACKGROUND\n",
                                    midL[0],
                                    midA[0],
                                    midB[0],
                                    stdDevL[0],
                                    stdDevA[0],
                                    stdDevB[0],
                                    area,
                                    circ,
                                    extent,
                                    uniformity,
                                    contrast,
                                    homogeneity,
                                    IDM,
                                    entropy);
                            BG++;
                            count++;
                        }
                        if (k > 54 && IOU < 0.5 && count == 0)
                        {
                            // Print on file features vector
                            fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,BACKGROUND\n",
                                    midL[0],
                                    midA[0],
                                    midB[0],
                                    stdDevL[0],
                                    stdDevA[0],
                                    stdDevB[0],
                                    area,
                                    circ,
                                    extent,
                                    uniformity,
                                    contrast,
                                    homogeneity,
                                    IDM,
                                    entropy);
                            BG++;
                            count++;
                        }
                    }
                }
            }

            // Clear Vectors from previous computations
            gtContours.clear();
            candidateSE.clear();
        }

        fclose(fp);

        printf("TOTAL OF SE: 147\n");
        printf("TOTAL OF SE FOUNDED: %d\n", TP);
        printf("TOTAL BACKGROUND PARTICLES: %d\n", BG);
        printf("FOUNDED %.3f%% OF SE\n", TP * 100 / (float)147);
    }
    catch (aia::error &ex)
    {
        std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
    }
    catch (ucas::Error &ex)
    {
        std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
    }

    return EXIT_SUCCESS;
}
void glcm(cv::Mat img, float &uniformity, float &contrast, float &homogeneity, float &IDM, float &entropy)
{

    int row = img.rows, col = img.cols;

    cv::Mat gl = cv::Mat::zeros(256, 256, CV_32FC1);

    //creating glcm matrix with 256 levels,radius=1 and in the horizontal direction
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col - 1; j++)
            gl.at<float>(img.at<uchar>(i, j), img.at<uchar>(i, j + 1)) = gl.at<float>(img.at<uchar>(i, j), img.at<uchar>(i, j + 1)) + 1;

    // normalizing glcm matrix for parameter determination
    gl = gl + gl.t();
    gl = gl / sum(gl)[0];

    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
        {
            // Calulating Uniformity
            uniformity += gl.at<float>(i, j) * gl.at<float>(i, j); //Pixel value squared

            // Finding Contrast and homogeneity
            contrast += (float)(i - j) * (i - j) * gl.at<float>(i, j);
            homogeneity += gl.at<float>(i, j) / (float)(1 + abs(i - j));

            // Finding Inverse Difference Moment
            if (i != j)
                IDM += gl.at<float>(i, j) / (float)((i - j) * (i - j));

            // Finding Entropy
            if (gl.at<float>(i, j) != 0)
                entropy += -gl.at<float>(i, j) * std::log2(gl.at<float>(i, j));
        }
}
