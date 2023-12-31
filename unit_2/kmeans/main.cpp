#include <opencv2/opencv.hpp>
#include <cstdlib>

using namespace cv;

int main(int argc, char** argv){
  int nClusters = 8;
  Mat rotulos;
  int nRodadas = 5;
  Mat centros;

  if(argc!=3){
    exit(0);
  }

  Mat img = imread("C:/Users/Eduardo Soares/Desktop/Projetos Pessoais/projects-open-cv/images/sushi.jpg", IMREAD_COLOR);
  Mat samples(img.rows * img.cols, 3, CV_32F);

  for( int y = 0; y < img.rows; y++ ){
    for( int x = 0; x < img.cols; x++ ){
      for( int z = 0; z < 3; z++){
        samples.at<float>(y + x*img.rows, z) = img.at<Vec3b>(y,x)[z];
      }
    }
  }

  kmeans(samples,
     nClusters,
     rotulos,
     TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS, 10000, 0.0001),//CV_TERMCRIT_ITER|CV_TERMCRIT_EPS
     nRodadas,
     KMEANS_PP_CENTERS,
     centros );


  Mat rotulada( img.size(), img.type() );
  for( int y = 0; y < img.rows; y++ ){
    for( int x = 0; x < img.cols; x++ ){
    int indice = rotulos.at<int>(y + x*img.rows,0);
    rotulada.at<Vec3b>(y,x)[0] = (uchar) centros.at<float>(indice, 0);
    rotulada.at<Vec3b>(y,x)[1] = (uchar) centros.at<float>(indice, 1);
    rotulada.at<Vec3b>(y,x)[2] = (uchar) centros.at<float>(indice, 2);
    }
  }
  imshow( "clustered image", rotulada );
  imwrite("sushi-kmeans.jpg", rotulada);
  waitKey( 0 );
}