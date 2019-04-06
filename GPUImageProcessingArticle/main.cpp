#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
	#include "CL/cl.h"
#endif


std::string GetPlatformName (cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

std::string GetDeviceName (cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo (id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetDeviceInfo (id, CL_DEVICE_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

void CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit (1);
	}
}

std::string LoadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	return result;
}

cl_program CreateProgram (const std::string& source,
	cl_context context)
{
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateProgramWithSource.html
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	CheckError (error);

	return program;
}

Mat Apply_filter (int DilateorErode)
{
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetPlatformIDs.html
	cl_uint platformIdCount = 0;
	clGetPlatformIDs (0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
		// return 1;
	} else {
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	}

	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);

	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetDeviceIDs.html
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_GPU, 0, nullptr,
		&deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
		// return 1;
	} else {
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	}

	std::vector<cl_device_id> deviceIds (deviceIdCount);
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_GPU, deviceIdCount,
		deviceIds.data (), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateContext.html
	const cl_context_properties contextProperties [] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds [0]),
		0, 0
	};

	cl_int error = CL_SUCCESS;
	cl_context context = clCreateContext (contextProperties, deviceIdCount,
		deviceIds.data (), nullptr, nullptr, &error);
	CheckError (error);

	std::cout << "Context created" << std::endl;

	// Create a program from source
	cl_program program = CreateProgram (LoadKernel ("kernel.cl"),
		context);
	CheckError (clBuildProgram (program, deviceIdCount, deviceIds.data (), 
		nullptr, nullptr, nullptr));
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateKernel.html
	cl_kernel kernel = clCreateKernel (program, "morphOpKernel", &error);
	CheckError (error);
	// Reading the input image
	Mat image = imread("in.png", IMREAD_GRAYSCALE);
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateImage2D.html
	static const cl_image_format format = { CL_LUMINANCE, CL_UNORM_INT8 };
	char *buffer = reinterpret_cast<char *>(image.data);
	cl_mem inputImage = clCreateImage2D (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,
		image.cols, image.rows, 0,
		// This is a bug in the spec
		// const_cast<char*> (image.data),
		buffer,
		&error);
	CheckError (error);
	cl_mem outputImage = clCreateImage2D (context, CL_MEM_WRITE_ONLY, &format,
		image.cols, image.rows, 0,
		nullptr, &error);
	CheckError (error);

	// Create a buffer for the filter weights
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateBuffer.html
	// cl_mem myFilter = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	// 	sizeof (int) * 1, &DilateorErode, &error);
	CheckError (error);
	// Setup the kernel arguments
	clSetKernelArg (kernel, 0, sizeof (cl_mem), &inputImage);
	clSetKernelArg (kernel, 1, sizeof (int), &DilateorErode);
	clSetKernelArg (kernel, 2, sizeof (cl_mem), &outputImage);
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateCommandQueue.html
	cl_command_queue queue = clCreateCommandQueue (context, deviceIds [0],
		0, &error);
	CheckError (error);
	// Run the processing
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueNDRangeKernel.html
	// const auto size = image.cols*image.rows;
	std::size_t offset [3] = { 0 };
	std::size_t size [3] = { image.cols, image.rows, 1 };
	CheckError (clEnqueueTask (queue, kernel, 0, NULL, NULL));
	CheckError (clEnqueueNDRangeKernel (queue, kernel, 2, offset, size, nullptr,
		0, nullptr, nullptr));
	// Prepare the result image, set to original image. We're changing it later
	Mat result = image;
	// Mat result = imread("in.png", CV_LOAD_IMAGE_GRAYSCALE);
	// Get the result back to the host
	std::size_t origin [3] = { 0 };
	std::size_t region [3] = { result.cols, result.rows, 1 };

	clEnqueueReadImage (queue, outputImage, CL_TRUE,
		origin, region, 0, 0,
		(void*)result.data, 0, nullptr, nullptr);

	
	clReleaseMemObject (outputImage);
	clReleaseMemObject (inputImage);

	clReleaseCommandQueue (queue);
	
	clReleaseKernel (kernel);
	clReleaseProgram (program);

	clReleaseContext (context);
	return result;
}

int main()
{	
	// Dilating the image
	
	int dilate = 0;
	Mat result_dilate = Apply_filter (dilate);
	imwrite("dilate.png", result_dilate);
	// Eroding the image

	int erode = 1;
	Mat result_erode = Apply_filter (erode);
	imwrite("erode.png", result_erode);
	return 0;
}
