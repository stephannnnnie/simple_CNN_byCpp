#include <opencv2/opencv.hpp>
#include <iostream>
#include "face_binary_cls.h"

using namespace cv;
using namespace std;

void conv0(float* m1, float* m2, float* m3, int width, int height, int channle, float* result);
void conv1(int out_channel, int in_channel, int height, int width, float* input, float* output);
void conv2(int out_channel, int in_channel, int height, int width, float* input, float* output);
Mat BGRToRGB(Mat img);

int main() {
	Mat image = imread("D:\\shidengheng\\C语言\\project2\\Project2\\SimpleCNNbyCPP-main\\samples\\bg.jpg");
	if (image.empty())
	{
		cout << "读入失败" << endl;
		return 0;
	}
	//cv::cvtColor(image, image, cv::COLOR_BGR2RGB);//转换图片格式
	image = BGRToRGB(image);
	int height = image.size().height;
	int width = image.size().width;
	int channle = 3;
	
	//将mat型的矩阵转为float*型的一维数组,并padding
	float* m1 = new float[(height+2) * (width+2)];
	float* m2 = new float[(height+2) * (width+2)];
	float* m3 = new float[(height+2) * (width+2)];
    /*if (m1 != nullptr)
	{
		cout << 1 << endl;
	}
    else {
	cout << 0 << endl;
    }
	if (m3 != nullptr)
	{
		cout << 1 << endl;
	}
	else {
		cout << 0 << endl;
	}*/
	for (int i = 0; i < height+2; i++)
	{
		for (int j = 0; j < width+2; j++)
		{
			m1[i * (width + 2) + j] = 0.0f;
			m2[i * (width + 2) + j] = 0.0f;
			m3[i * (width + 2) + j] = 0.0f;
		}
	}
	for (int  i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			
			m1[(i+1) * (width + 2) + j + 1] = (float)image.at<Vec3b>(i, j)[0]/255.0f;
			m2[(i+1) * (width + 2) + j + 1] = (float)image.at<Vec3b>(i, j)[1]/255.0f;
			m3[(i+1) * (width + 2) + j + 1] = (float)image.at<Vec3b>(i, j)[2]/255.0f;
			//cout << m1[(i + 1) * (height + 1) + j + 1] << "  "<< m2[(i + 1) * (height + 1) + j + 1]<<"  "<< m3[(i + 1) * (height + 1) + j + 1]<<"  ";
		}
		//cout << endl;
	}
	
	
//第一次进行卷积

	float* out1 = new float[16*(height/2) * (width/2)];

	conv0(m1, m2, m3, width, height,16,(out1));
	height = height / 2;
	width = width / 2;
	
		/*for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				cout << out1[ i * (width) + j] << "  ";
			}
			cout << endl;
		}*/
	
	
	
	//ReLU
	for (int i = 0; i < 16* (height)*(width); i++)
	{
		if (out1[i] < 0)
			out1[i] = 0;
	}
	//MaxPool
	int out_row = (height) / 2;
	int out_col = (width) / 2;
	float* max1 = new float[16*out_row*out_col];
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < out_row; j++)
		{
			for (int k = 0; k < out_col; k++)
			{
				max1[i * out_row * out_col + j * out_col + k] = 0.0f;
			}
		}
	}

	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < out_row; j++)
		{
			for (int k = 0; k < out_col; k++)
			{ 
				vector<float> temp;
				for (int ii = 2*j; ii <2*j+2 ; ii++)
				{
					for (int jj =2* k; jj < 2*k+2; jj++) {
						temp.push_back(out1[i * width * height + ii * width + jj]);
					}
				}
				sort(temp.begin(), temp.end());
				max1[i * out_row * out_col + j * out_col + k] = temp[temp.size() - 1];
				
				//cout<< max1[i * out_row * out_col + j * out_col + k]<<"  ";
			}			
		}
	}
	
	/*for (int n = 0; n < 16; n++)
	{
		for (int i = 0; i < out_row; i++)
		{
			for (int j = 0; j < out_col; j++)
			{
				//cout << n * out_row * out_col + i * out_col + j << endl;
				cout << max1[n * out_row * out_col + i * out_col + j] <<" ";
			}
			cout << endl;
		}
		cout << endl;
	}*/


//second conv
	float* out2 = new float[32 * (out_row - 2) * (out_col - 2)];//32*30*30
	conv1(32,16,out_row,out_col,max1,out2);

	
	//relu
	for (int i = 0; i < 32 * (out_row - 2) * (out_col - 2); i++)
	{
		if (out2[i] < 0)
			out2[i] = 0;
	}
	/*for (int i = 0; i < out_row-2; i++)
	{
		for (int j = 0; j < out_col-2; j++)
		{
			cout << out2[i * (out_col-2)+j] << "  ";
		}
		cout << endl;
	}*/
	//maxpool  32*15*15
	int max_row = (out_row-2) / 2 ;
	int max_col = (out_col-2) / 2 ;
	float* max2 = new float[32 * max_row * max_col];
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < max_row; j++)
		{
			for (int k = 0; k < max_col; k++)
			{
				max2[i * max_row * max_col + j * max_col + k] = 0.0f;
			}
		}
	}
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < max_row; j++)
		{
			for (int k = 0; k < max_col; k++)
			{
				vector<float> temp;
				for (int ii =2* j; ii < 2*j + 2; ii++)
				{
					for (int jj =2* k; jj < (int)2*k + 2; jj++) {
						if (jj<max_col&&ii<max_row)
						{
							temp.push_back(out2[i * (out_col-2) * (out_row-2) + ii * (out_col-2) + jj]);
						}
						
					}
				}
				if (!temp.empty())
				{
                 sort(temp.begin(), temp.end());
				 max2[i * max_row * max_col + j * max_col + k] = temp[temp.size() - 1];
				}
				
				//cout << i * out_row * out_col + j * out_col + k << " ";
				//cout<< max2[i * out_row * out_col + j * out_col + k]<<"  ";
			}
		}
	}
	/*for (int n = 0; n < 32; n++)
	{
		for (int i = 0; i < max_row; i++)
		{
			for (int j = 0; j < max_col; j++)
			{
				//cout << n * out_row * out_col + i * out_col + j << endl;
				cout << max2[n * max_row * max_col + i * max_col + j] << endl;
			}
			cout << endl;
		}
		cout << endl;
	}*/

	//third conv--32*8*8
	
	float* pad3 = new float[32 * (max_col+2)*(max_row+2)];
	for (int i = 0; i < 32*(max_col+2)*(max_row+2); i++)
	{
		pad3[i] = 0.0f;
	}
		
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < max_row; j++)
		{
			for (int k = 0; k < max_col; k++)
			{
				pad3[i * (max_col + 2) * (max_row + 2) + (j + 1) * (max_col + 2) + k + 1] = max2[i * max_row * max_col + j * max_col + k];
			}
		}
	}
	int out_rows = max_row / 2 +1;
	int out_cols = max_col / 2 +1;
	
	float* out3 = new float[32 * out_rows * out_cols];
	/*for (int i = 0; i < 32 * out_rows * out_cols; i++)
	{
		out3[i] = 0.0f;
	}*/
	conv2(32, 32, max_row, max_col, pad3, out3);
	//relu
	for (int i = 0; i < 32 * (out_rows ) * (out_cols ); i++)
	{
		if (out3[i] < 0)
			out3[i] = 0;
	}
	for (int n = 0; n < 32; n++) {
		for (int i = 0; i < out_rows; i++)
		{
			for (int j = 0; j < out_cols; j++)
			{
				//cout << n * out_row * out_col + i * out_col + j << endl;
				cout << out3[n * out_rows * out_cols+i * out_cols + j] << "  ";
			}
			cout << endl;
		}
	}


	//fully-connected layer
	float* final = new float[2];
	final[0] = 0;
	final[1] = 0;
	for (int i = 0; i < 32*out_rows*out_cols; i++)
	{
		final[0] += out3[i] * fc0_weight[i];
		final[1] += out3[i] * fc0_weight[32*out_rows*out_cols + i];
	}
	final[0] += fc0_bias[0];
	final[1] += fc0_bias[1];
	//softmax

	float* per = new float[2];
	per[0] = exp(final[0] )/ (exp(final[0]) + exp(final[1]));
	per[1] = exp(final[1]) / (exp(final[0]) + exp(final[1]));
	cout << "confidence face score: " << per[1] << endl;
	cout << "background's confidence score: " <<per[0]<< endl;
	
    m1 = NULL;
	delete[] m1;
	m2 = NULL;
	m3 = NULL;
	delete[] m2;	
	delete[] m3;
    out1 = NULL;
	max1 = NULL;	
	delete[] out1;
	delete[] max1;
	out2 = NULL;
	delete[] out2;
	max2 = NULL;
	delete[] max2;
	out3 = NULL;
	delete[] out3;
	pad3 = NULL;
	delete[] pad3;
	final = NULL;
	delete[] final;
	delete[] per;
	return 0;
}
void conv0(float* m1,float* m2,float* m3,int width,int height,int channle,float * result)
{
	int point = 0;
	for (int ch = 0; ch < channle; ch++)
	{
		for (int i = 0; i < height; i=i+2)
		{
			for (int j = 0; j < width; j = j + 2)
			{
				float sum = 0.0f;		
				bool flag = false;
				for (int kh = 0; kh < 3; kh++)
				{
					for (int kw = 0; kw < 3; kw++)
					{
						if (i + 2 < height+2 && j + 2 < width+2)
						{
							flag = true;
							sum += (float)m1[(width + 2) * (i + kh) + j + kw] * conv0_weight[27 * ch + 3 * kh + kw];//
							sum += (float)m2[(width + 2) * (i + kh) + j + kw] * conv0_weight[27 * ch + 9 + 3 * kh + kw];//
							sum += (float)m3[(width + 2) * (i + kh) + j + kw] * conv0_weight[27 * ch + 9 * 2 + 3 * kh + kw];//
						}
					}
				}
				if (flag)
				{
					sum += conv0_bias[ch];
					result[point] = sum;
					point++;
				}
			}			
		}
	}
}

void conv1(int out_channel,int in_channel,int height,int width,float* input,float* output) {
	int point=0;
	
	for (int ch = 0; ch < out_channel; ch++)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float sum = 0.0f;
				bool flag=false;
				for (int  kc = 0; kc < in_channel; kc++)
				{
					for (int kh = 0; kh < 3; kh++) 
					{
						for (int kw = 0; kw < 3; kw++) 
						{
							if (i+2<height&&j+2<width)
							{
							   flag = true;
			                   sum += input[kc * ((height ) * (width)) + (width) * (i + kh) + j + kw] * conv1_weight[144 * ch + 9 * kc + 3* kh + kw];//
							}
				
						}
					}
				}
				if (flag)
				{
					sum += conv1_bias[ch];				
					output[point] = sum;				
					point++;
				}
				
			}
		}
	}
}
void conv2(int out_channel, int in_channel, int height, int width, float* input, float* output) {
	int point=0;
	for (int ch = 0; ch < out_channel; ch++)
	{
		for (int i = 0; i < height; i=i+2)
		{
			for (int j = 0; j < width; j=j+2)
			{
				float sum = 0.0f;
				bool flag = false;
				for (int kc = 0; kc < in_channel; kc++)
				{
					for (int kh = 0; kh < 3; kh++)
					{
						for (int kw = 0; kw < 3; kw++)
						{
							if (i + 2 < height+2 && j + 2 < width+2)
							{
								flag = true;							
								sum += input[kc * ((height + 2) * (width + 2)) + (width + 2) * (i + kh) + j + kw] * conv2_weight[288 * ch + 9 * kc + 3 * kh + kw];//
							}
						}
					}
				}
				if (flag)
				{
					sum += conv2_bias[ch];
					output[point] = sum;
					point++;
				}
			}
		}
	}
}

Mat BGRToRGB(Mat img)
{
	Mat image(img.rows, img.cols, CV_8UC3);
	for (int i = 0; i < img.rows; ++i)
	{
		//获取第i行首像素指针
		Vec3b* p1 = img.ptr<Vec3b>(i);
		Vec3b* p2 = image.ptr<Vec3b>(i);
		for (int j = 0; j < img.cols; ++j)
		{
			//将img的bgr转为image的rgb 
			p2[j][2] = p1[j][0];
			p2[j][1] = p1[j][1];
			p2[j][0] = p1[j][2];
		}
	}
	return image;
}
