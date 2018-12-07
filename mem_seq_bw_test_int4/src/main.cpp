/**
 * File              : host/src/main.cpp
 * Author            : Cheng Liu <st.liucheng@gmail.com>
 * Date              : 11.12.2017
 * Last Modified Date: 11.12.2017
 * Last Modified By  : Cheng Liu <st.liucheng@gmail.com>
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
// ACL specific includes
#include "CL/opencl.h"
//#include "ACLHostUtils.h"
#include "AOCLUtils/aocl_utils.h"
#define LEN (4)
using namespace aocl_utils;
static size_t vector_size = 256*1024*1024;
static unsigned int round_mask = (vector_size/LEN) - 1;
static unsigned int burst_base_addr = 0;
static unsigned int burst_num = 16*1024*1024;
static unsigned int stride = 128/LEN;
double bw;
double lat;

// ACL runtime configuration
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_kernel kernel;
static cl_kernel kernel_read;
static cl_kernel kernel_write;
static cl_program program;
static cl_int status;


// input and output vectors
static unsigned *hdatain, *hdataout;

static void initializeVector(unsigned* vector, int size) {
  for (int i = 0; i < size; ++i) {
    vector[i] = 0x32103210;
  }
}

static void initializeVector_seq(unsigned* vector, int size) {
  for (int i = 0; i < size; ++i) {
    vector[i] = i;
  }
}

static void dump_error(const char *str, cl_int status) {
  printf("%s\n", str);
  printf("Error code: %d\n", status);
}

// free the resources allocated during initialization
static void freeResources() {

  if(kernel) 
    clReleaseKernel(kernel);  
  if(kernel_read) 
    clReleaseKernel(kernel_read);  
  if(kernel_write) 
    clReleaseKernel(kernel_write);      
  if(program) 
    clReleaseProgram(program);
  if(queue) 
    clReleaseCommandQueue(queue);
  if(hdatain) 
   clSVMFreeAltera(context,hdatain);
  if(hdataout) 
   clSVMFreeAltera(context,hdataout);     
  if(context) 
    clReleaseContext(context);

}

cl_int setHardwareEnv(
		cl_uint &num_platforms,
		cl_uint &num_devices
		){
	// get the platform ID
	status = clGetPlatformIDs(1, &platform, &num_platforms);
	if(status != CL_SUCCESS) {
		dump_error("Failed clGetPlatformIDs.", status);
		freeResources();
		return 1;
	}

	if(num_platforms != 1) {
		printf("Found %d platforms!\n", num_platforms);
		freeResources();
		return 1;
	}

	// get the device ID
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
	if(status != CL_SUCCESS) {
		dump_error("Failed clGetDeviceIDs.", status);
		freeResources();
		return 1;
	}
	if(num_devices != 1) {
		printf("Found %d devices!\n", num_devices);
		freeResources();
		return 1;
	}

	// create a context
	context = clCreateContext(0, 1, &device, NULL, NULL, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateContext.", status);
		freeResources();
		return 1;
	}

	return CL_SUCCESS;
}

cl_int setKernelEnv(){
	// create a command queue
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateCommandQueue.", status);
		freeResources();
		return 1;
	}

	// create the program
	cl_int kernel_status;
	size_t binsize = 0;
	unsigned char * binary_file = loadBinaryFile("./mem_bandwidth.aocx", &binsize);

	if(!binary_file) {
		dump_error("Failed loadBinaryFile.", status);
		freeResources();
		return 1;
	}
	program = clCreateProgramWithBinary(
			context, 1, &device, &binsize, 
			(const unsigned char**)&binary_file, 
			&kernel_status, &status);

	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateProgramWithBinary.", status);
		freeResources();
		return 1;
	}
	delete [] binary_file;

	// build the program
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	if(status != CL_SUCCESS) {
		dump_error("Failed clBuildProgram.", status);
		freeResources();
		return 1;
	}
	return CL_SUCCESS;
}

void cleanup(){}

int main(int argc, char *argv[]) {
	cl_uint num_platforms;
	cl_uint num_devices;

	status = setHardwareEnv(num_platforms, num_devices);
	printf("Creating host buffers.\n");
	unsigned int buf_size =  vector_size * sizeof(unsigned int);

	// allocate and initialize the input vectors
	hdatain = (unsigned int*)clSVMAllocAltera(context, 0, buf_size, 1024); 
	hdataout = (unsigned int*)clSVMAllocAltera(context, 0, buf_size, 1024);
	if(!hdatain || !hdataout) {
		dump_error("Failed to allocate buffers.", status);
		freeResources();
		return 1;	
	}
	initializeVector_seq(hdatain, vector_size);
	initializeVector(hdataout, vector_size);
	status = setKernelEnv();


	printf("Creating memread kernel\n");
	{
		kernel_read = clCreateKernel(program, "memread", &status);
		if(status != CL_SUCCESS) {
			dump_error("Failed clCreateKernel.", status);
			freeResources();
			return 1;
		}

		// set the arguments
		cl_int arg_2 = burst_base_addr;
		cl_int arg_3 = burst_num;
		cl_int arg_4 = stride;
		cl_int arg_5 = round_mask;
		clSetKernelArgSVMPointerAltera(kernel_read, 0, (void*)hdatain);
		clSetKernelArgSVMPointerAltera(kernel_read, 1, (void*)hdataout);
		clSetKernelArg(kernel_read, 2, sizeof(cl_int), &(arg_2));
		clSetKernelArg(kernel_read, 3, sizeof(cl_int), &(arg_3));
		clSetKernelArg(kernel_read, 4, sizeof(cl_int), &(arg_4));
		clSetKernelArg(kernel_read, 5, sizeof(cl_int), &(arg_5));

		printf("Launching the memory read kernel...\n");
		status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
				(void *)hdatain, buf_size, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}

		status = clEnqueueSVMMap(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE, 
				(void *)hdataout, buf_size, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}	

		// launch kernel
		const double start_time = getCurrentTimestamp();
		status = clEnqueueTask(queue, kernel_read, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			dump_error("Failed to launch kernel.", status);
			freeResources();
			return 1;
		}

		status = clEnqueueSVMUnmap(queue, (void *)hdatain, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMUnmap", status);
			freeResources();
			return 1;
		}

		status = clEnqueueSVMUnmap(queue, (void *)hdataout, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMUnmap", status);
			freeResources();
			return 1;
		}
		clFinish(queue);
		const double end_time = getCurrentTimestamp();

		// Wall-clock time taken.
		double time = (end_time - start_time);
		bw = burst_num * sizeof(unsigned int) * LEN/ (time * 1000000.0);
		lat = (time * 1000000000.0) / (burst_num * LEN);
		printf("Average memory read latency = %.2f ns\n", lat);
		printf("Read Bandwidth = %.2f MB/s\n", bw);

	}

	freeResources();

	return 0;
}



