//-----------------------------------------------------------------------------
// Filename: bfs_host.cpp
// Version: 1.0
// Description: Breadth-first search OpenCL benchmark.
//
// Author:      Cheng Liu
// Email:       liucheng@ict.ac.cn, st.liucheng@gmail.com
// Affiliation: Institute of Computing Technology, Chinese Academy of Sciences
//
// Acknowledgement:
//
//-----------------------------------------------------------------------------

#include "graph.h"
#include "opencl_utils.h"
#include <stdlib.h>
#include <malloc.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "graph.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

//#define SW_EMU
#define REPEAT 64
#define BATCH 16
#define BW 16
#define VMAX (4 * 1024 * 1024)

#ifdef SW_EMU
#define BMAP_DEPTH (VMAX / (BATCH * BW)) 
#endif
typedef unsigned short bmap_dt;

#define HERE do {std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;} while(0)

#define ERROR(FMT, ARG...) do {fprintf(stderr,"File=%s, Line=%d  \
		" FMT " \n",__FILE__, __LINE__, ##ARG); exit(-1);} while(0)

#define PRINT(FMT, ARG...) do {fprintf(stdout,"File=%s, Line=%d  \
		" FMT " \n",__FILE__, __LINE__, ##ARG);} while(0)

#define AOCL_ALIGNMENT 64

template<class T>
struct AlignedArray{
	AlignedArray(size_t numElts){ data = (T*) memalign( AOCL_ALIGNMENT, sizeof(T) * numElts ); }
	~AlignedArray(){ free(data); }

	T& operator[](size_t idx){ return data[idx]; }
	const T& operator[](size_t idx) const { return data[idx]; }

	T* data;
};

Graph* createGraph(const std::string &gName){
	std::string dir = "../../graph-data/";
	//std::string dir = "/data/DATA/liucheng/graph-data/";
	std::string fName;

	if     (gName == "dblp")        fName = "dblp.ungraph.txt";
	else if(gName == "youtube")     fName = "youtube.ungraph.txt";
	else if(gName == "lj")          fName = "lj.ungraph.txt";
	else if(gName == "pokec")       fName = "pokec-relationships.txt";
	else if(gName == "wiki-talk")   fName = "wiki-Talk.txt";
	else if(gName == "lj1")         fName = "LiveJournal1.txt";
	else if(gName == "orkut")       fName = "orkut.ungraph.txt";
	else if(gName == "rmat-19-32")  fName = "rmat-19-32.txt";
	else if(gName == "rmat-21-32")  fName = "rmat-21-32.txt";
	else if(gName == "rmat-21-128") fName = "rmat-21-128.txt";
	else if(gName == "twitter")     fName = "twitter_rv.txt";
	else if(gName == "friendster")  fName = "friendster.ungraph.txt";
	else ERROR(" Unknown graph name %s .", gName.c_str());

	std::string fpath = dir + fName;
	return new Graph(fpath.c_str());
}

void swBfsInit(int vertexNum, char* depth, const int &vertexIdx){
	for(int i = 0; i < vertexNum; i++){
		depth[i] = -1;
	}
	depth[vertexIdx] = 0;
}

void swBfs(CSR* csr, char* depth, const int &vertexIdx){
	std::vector<int> frontier;
	char level = 0;
	while(true){
		for(int i = 0; i < csr->vertexNum; i++){
			if(depth[i] == level){
				frontier.push_back(i);
				int start = csr->rpao[2 * i];
				int num   = csr->rpao[2 * i + 1];
				for(int cidx = 0; cidx < num; cidx++){
					int ongb = csr->ciao[start + cidx];
					if(ongb != -1){
						if(depth[ongb] == -1){
							depth[ongb] = level + 1;
						}
					}
				}
			}
		}

		if(frontier.empty()){
			break;
		}
		//std::cout << "swBfs iteration: " << (int)level  << ", frontier size: " << frontier.size() << std::endl;

		level++;
		frontier.clear();
	}
}

void hwBfsInit(int vertexNum, char* depth, int rootVidx){
	for(int i = 0; i < vertexNum; i++){
		if(i == rootVidx){
			depth[i] = 0;
		}
		else{
			depth[i] = -1;
		}
	}
}

int verify(char* swDepth, char* hwDepth, const int &num){
	bool match = true;
	for (int i = 0; i < num; i++) {
		if (swDepth[i] != hwDepth[i]) {
			PRINT("swDepth[%d] = %d, hwDepth[%d] = %d\n", i, swDepth[i], i, hwDepth[i]);	
			match = false;
			break;
		} 
	}

	if (match){
		printf("TEST PASSED.\n");
		return EXIT_SUCCESS;
	} 
	else{
		printf("TEST FAILED.\n");
		return -1;
	}
}

int getStartVertexIdx(const std::string &gName){
	if(gName == "youtube")    return 320872;
	if(gName == "lj1")        return 3928512;
	if(gName == "pokec")      return 182045;
	if(gName == "rmat-19-32") return 104802;
	if(gName == "rmat-21-32") return 365723;
	return -1;
}

// Sum the array
int getSum(int *ptr, int num){	
	if(ptr == nullptr) return -1;
	int sum = 0;
	for(int i = 0; i < num; i++){
		sum += ptr[i];
	}
	return sum;
}

int align(int num, int dataWidth, int alignedWidth){
	if(dataWidth > alignedWidth){
		std::cout << "Aligning to smaller data width is not supported." << std::endl;
		return -1;
	}
	else{
		int wordNum = alignedWidth / dataWidth;
		int alignedNum = ((num - 1)/wordNum + 1) * wordNum;
		return alignedNum;
	}
}

void addKernelNames(std::vector<std::string> &kernelNames){
	kernelNames.push_back("read_frontier");
	kernelNames.push_back("read_rpa");
	kernelNames.push_back("read_cia");
	kernelNames.push_back("traverse_cia");
	for(int i = 0; i < BATCH; i++){
		std::stringstream ss;
		ss << i;
		std::string funName = "write_frontier" + ss.str();
		kernelNames.push_back(funName);
	}
}

void initRpaInfo(int* rpaInfo, CSR* csr, const int &rootVidx){
	for(int i = 0; i < 2 * csr->vertexNum; i++){
		rpaInfo[i] = csr->rpao[i];
	}
}

void setPlatformEnv(
		cl_device_id &device,
		cl_platform_id &platform,
		cl_uint &num_platforms,
		cl_uint &num_devices,
		cl_context &context
		)
{
	cl_int status = clGetPlatformIDs(1, &platform, &num_platforms);
	if(status != CL_SUCCESS) ERROR("Failed clGetPlatformIDs. Error code: %d.", status);
	if(num_platforms != 1) ERROR("Found %d platforms!", num_platforms);

	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
	if(status != CL_SUCCESS) ERROR("Failed clGetDeviceIDs. Error code: %d.", status);
	if(num_devices != 1) ERROR("Found %d devices!", num_devices);

	context = clCreateContext(0, 1, &device, NULL, NULL, &status);
	if(status != CL_SUCCESS) ERROR("Failed clCreateContext Error code: %d", status);

}

void setKernelEnv(
		cl_device_id &device, 
		cl_context &context, 
		std::vector<cl_command_queue> &queues, 
		std::vector<cl_kernel> &kernels,
		std::vector<std::string> &kernelNames,
		cl_program &program,
		std::string binFileName
		)
{
	cl_int status;
	for(size_t i = 0; i < queues.size(); i++){
		queues[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
		if(status != CL_SUCCESS) ERROR("Failed creating clCreateCommandQueue. Error code: %d", status);
	}

	size_t binsize = 0;
	unsigned char * binaryFile = aocl_utils::loadBinaryFile(binFileName.c_str(), &binsize);
	if(!binaryFile) ERROR("Failed to load binary file.");

	cl_int kernelStatus;
	program = clCreateProgramWithBinary(
			context, 
			1, 
			&device, 
			&binsize, 
			(const unsigned char**)&binaryFile, 
			&kernelStatus, 
			&status);

	if(status != CL_SUCCESS){
		delete [] binaryFile;
		ERROR("Failed clCreateProgramWithBinary, ERROR code: %d", status);
	}

	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	if(status != CL_SUCCESS) ERROR("Failed on clBuildProgram, Error code: %d", status);

	for(size_t i = 0; i < kernels.size(); i++){
		kernels[i] = clCreateKernel(program, kernelNames[i].c_str(), &status);
		if(status != CL_SUCCESS) ERROR("Failed to create kernel %s.", kernelNames[i].c_str());
	}
}

void cleanup(){}

int main(int argc, char ** argv){
	////////////////////////////////////////
	// Platform definition
	////////////////////////////////////////
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	std::vector<cl_command_queue> queues;
	std::vector<cl_kernel> kernels;
	cl_program program;
	cl_int status;
	cl_uint num_platforms;
	cl_uint num_devices;
	setPlatformEnv(device, platform, num_platforms, num_devices, context);

	////////////////////////////////
	// Env init
	///////////////////////////////
	std::vector<std::string> kernelNames;
	addKernelNames(kernelNames);
	kernels.resize(kernelNames.size());
	queues.resize(kernelNames.size());
	std::string binFileName = "./bfs_fpga.aocx";
	setKernelEnv(device, context, queues, kernels, kernelNames, program, binFileName);

	////////////////////////////////
	// Graph init
	////////////////////////////////
	int         batch   = BATCH;
	std::string gName = "youtube";
	Graph* gptr = createGraph(gName);
	int millionEdges = gptr->edgeNum/(1000000.0);
	CSR* csr = new CSR(*gptr, batch);
	free(gptr);
	int vertexNum = csr->vertexNum; 
	//int rpaoSize = (int)(csr->rpao.size());
	int ciaoSize = (int)(csr->ciao.size());
	int segSize  = (vertexNum + batch - 1) / batch;
	int rootVidx = getStartVertexIdx(gName);
	char* hwDepth = (char*)malloc(vertexNum * sizeof(char));
	//char* swDepth = (char*)malloc(vertexNum * sizeof(char));
	std::cout << "Graph is loaded." << std::endl;

	int* frontier = (int*) clSVMAllocAltera(context, 0, sizeof(int) * vertexNum, 1024);
	if(!frontier) ERROR("frontier allocation error.");
	int* rpaInfo = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 2 * vertexNum, 1024); 
	if(!rpaInfo) ERROR("rpaInfo allocation error!"); 
	int* cia = (int*) clSVMAllocAltera(context, 0, sizeof(int) * ciaoSize, 1024);
	if(!cia) ERROR("cia allocation error!");
	for(int i = 0; i < ciaoSize; i++){
		cia[i] = csr->ciao[i];
	}
	char* level  = (char*)clSVMAllocAltera(context, 0, sizeof(char), 1024);
	if(!level) ERROR("level allocation error.");
	int* frontierSize  = (int*)clSVMAllocAltera(context, 0, sizeof(int), 1024);
	if(!frontierSize) ERROR("frontierSize allocation error.");


	std::vector<int*> nextFrontier(batch);
	for(int i = 0; i < batch; i++){
		nextFrontier[i] = (int*) clSVMAllocAltera(context, 0, sizeof(int) * segSize, 1024);
		if(!nextFrontier[i]) ERROR("nextFrontier allocation error!");
	}
	int* nextFrontierSize = (int*) clSVMAllocAltera(context, 0, sizeof(int) * batch, 1024);
	if(!nextFrontierSize) ERROR("nextFrontierSize allocation error!");

#ifdef SW_EMU
	std::vector<bmap_dt*> bmap(batch);
	for(int i = 0; i < batch; i++){
		bmap[i] = (bmap_dt*) clSVMAllocAltera(context, 0, sizeof(bmap_dt) * BMAP_DEPTH, 1024);
	}
#endif

	// Map memory objects to the kernel
	status = clEnqueueSVMMap(queues[0], CL_TRUE, CL_MAP_READ, (void *)frontier, sizeof(int) * vertexNum, 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Failed clEnqueueSVMMap, error code: %d", status);
	clSetKernelArgSVMPointerAltera(kernels[0], 0, (void*)frontier);
	status = clEnqueueSVMMap(queues[0], CL_TRUE, CL_MAP_READ, (void *)frontierSize, sizeof(int), 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Failed clEnqueueSVMMap, error code: %d", status);
	clSetKernelArgSVMPointerAltera(kernels[0], 1, (void*)frontierSize);


	status = clEnqueueSVMMap(queues[1], CL_TRUE, CL_MAP_READ, (void *)rpaInfo, sizeof(int) * 2 * vertexNum, 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Failed clEnqueueSVMMap, error code: %d", status);
	clSetKernelArgSVMPointerAltera(kernels[1], 0, (void*)rpaInfo);
	status = clEnqueueSVMMap(queues[1], CL_TRUE, CL_MAP_READ, (void *)frontierSize, sizeof(int), 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Failed clEnqueueSVMMap, error code: %d", status);
	clSetKernelArgSVMPointerAltera(kernels[1], 1, (void*)frontierSize);


	status = clEnqueueSVMMap(queues[2], CL_TRUE, CL_MAP_READ, (void *)cia, sizeof(int) * ciaoSize, 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Failed clEnqueueSVMMap, error code: %d", status);
	clSetKernelArgSVMPointerAltera(kernels[2], 0, (void*)cia);
	status = clEnqueueSVMMap(queues[2], CL_TRUE, CL_MAP_READ, (void *)frontierSize, sizeof(int), 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Failed clEnqueueSVMMap, error code: %d", status);
	clSetKernelArgSVMPointerAltera(kernels[2], 1, (void*)frontierSize);


	status = clEnqueueSVMMap(queues[3], CL_TRUE, CL_MAP_WRITE, (void *)nextFrontierSize, sizeof(int) * batch, 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Failed clEnqueueSVMMap, error code: %d", status);
	clSetKernelArgSVMPointerAltera(kernels[3], 0, (void*)nextFrontierSize);
	status = clEnqueueSVMMap(queues[3], CL_TRUE, CL_MAP_READ, (void *)level, sizeof(char), 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Failed clEnqueueSVMMap, error code: %d", status);
	clSetKernelArgSVMPointerAltera(kernels[3], 2, (void*)level);

#ifdef SW_EMU
	for(int i = 0; i < batch; i++){
		status = clEnqueueSVMMap(queues[3], 
				CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
				(void *)(bmap[i]), 
				sizeof(bmap_dt) * BMAP_DEPTH, 
				0, 
				NULL, 
				NULL); 
		if(status != CL_SUCCESS) ERROR("Failed clEnqueueSVMMap, bitmap[%d], error code: %d", i, status);
		clSetKernelArgSVMPointerAltera(kernels[3], 3 + i, (void*)(bmap[i]));
	}
#endif

	for(int i = 0; i < batch; i++){
		status = clEnqueueSVMMap(queues[4 + i], CL_TRUE, CL_MAP_WRITE, (void *)(nextFrontier[i]), sizeof(int) * segSize, 0, NULL, NULL); 
		if(status != CL_SUCCESS) ERROR("Failed clEnqueueSVMMap, error code: %d", status); 
		clSetKernelArgSVMPointerAltera(kernels[4 + i], 0, (void*)(nextFrontier[i]));
	}

	double avgFPGARuntime = 0;
	double avgBFSRuntime = 0;
	int it = 0;
	int realIt = 0;
	while(it < REPEAT){
		realIt++;
		rootVidx = rand()%vertexNum;

		///////////////////////////////////////////
		// software bfs
		//////////////////////////////////////////
		//swBfsInit(vertexNum, swDepth, rootVidx);
		auto begin = std::chrono::high_resolution_clock::now();
		//swBfs(csr, swDepth, rootVidx);
		auto end = std::chrono::high_resolution_clock::now();
		double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

		////////////////////////////////////
		// hardware bfs
		////////////////////////////////////
		initRpaInfo(rpaInfo, csr, rootVidx);
		hwBfsInit(vertexNum, hwDepth, rootVidx);
		level[0] = 0;
		frontier[0] = rootVidx;
		frontierSize[0] = 1;
		clSetKernelArg(kernels[3], 1, sizeof(int), (void*)&rootVidx);

		std::vector<double> fpgaExeTime;
		begin = std::chrono::high_resolution_clock::now();
		while(frontierSize[0] > 0){
			auto t0 = std::chrono::high_resolution_clock::now();
			status = clEnqueueTask(queues[0], kernels[0], 0, NULL, NULL);                               
			if(status != CL_SUCCESS) ERROR("Failed to enqueue task.");

			status = clEnqueueTask(queues[1], kernels[1],  0, NULL, NULL);
			if(status != CL_SUCCESS) ERROR("Failed to enqueue task."); 

			status = clEnqueueTask(queues[2], kernels[2],  0, NULL, NULL);
			if(status != CL_SUCCESS) ERROR("Failed to enqueue task.");

			status = clEnqueueTask(queues[3], kernels[3],  0, NULL, NULL);
			if(status != CL_SUCCESS) ERROR("Failed to enqueue task.");

			for(int i = 0; i < batch; i++){
				status = clEnqueueTask(queues[4 + i], kernels[4 + i],  0, NULL, NULL);
				if(status != CL_SUCCESS) ERROR("Failed to enqueue task.");
			}

			for(auto q : queues){
				clFinish(q);
			}

			auto t1 = std::chrono::high_resolution_clock::now();
			double t = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
			fpgaExeTime.push_back(t);

			int sum = 0;
			for(int i = 0; i < batch; i++){
				sum += nextFrontierSize[i];
			}
			frontierSize[0] = sum;

			std::vector<int> baseVec(batch, 0);
			for(int i = 1; i < batch; i++){
				baseVec[i] = baseVec[i - 1] + nextFrontierSize[i - 1];
			}

			//if(frontierSize < (vertexNum * 0.1)){
			//#pragma omp parallel for
			for(int i = 0; i < batch; i++){
				int id = 0;
				for(int j = 0; j < nextFrontierSize[i]; j++){
					int vidx = nextFrontier[i][j];
					frontier[baseVec[i] + id] = vidx;
					//#ifdef SW_EMU
					//hwDepth[vidx] = level[0] + 1;
					//#endif
					//rpaInfo[baseVec[i] + 2 * id] = csr->rpao[2 * vidx];
					//rpaInfo[baseVec[i] + 2 * id + 1] = csr->rpao[2 * vidx + 1];
					id++;
				}
			}
			//}
			//else{
			//	#pragma omp parallel for
			//	for(int i = 0; i < batch; i++){
			//		int id = 0;
			//		for(int j = 0; j < nextFrontierSize[i]; j++){
			//			int vidx = nextFrontier[i][j];
			//			//hwDepth[vidx] = level + 1;
			//			rpaInfo[baseVec[i] + 2 * id] = csr->rpao[2 * vidx];
			//			rpaInfo[baseVec[i] + 2 * id + 1] = csr->rpao[2 * vidx + 1];
			//			id++;
			//		}
			//	}
			//}

			//std::cout << "hwBfs iteration: " << (int)(level[0]) << ", frontier size: " << frontierSize[0] << std::endl;
			level[0]++;
		}
		end = std::chrono::high_resolution_clock::now();
		elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
		if(level[0] < 3) continue;
		avgBFSRuntime += elapsedTime;

		double sum = 0;
		for(auto t : fpgaExeTime){
			sum += t;
		}
		avgFPGARuntime += sum;

		//#ifdef SW_EMU
		//verify(swDepth, hwDepth, vertexNum);
		//#endif
		it++;
	}
	std::cout << "After " << realIt << " runs," << it << " non-trivial searches are evaluated." << std::endl;
	std::cout << "Average hardware bfs time: " << avgBFSRuntime/REPEAT << " ms" << std::endl;
	std::cout << "Average MTEPS: " << 1000 * millionEdges * REPEAT / avgBFSRuntime << std::endl;
	std::cout << "Average FPGA exeuction time: " << avgFPGARuntime/REPEAT << " ms" << std::endl;
	std::cout << "Average CPU exeuction time: " << (avgBFSRuntime - avgFPGARuntime)/REPEAT << " ms" << std::endl;

	// Cleanup memory
	status = clEnqueueSVMUnmap(queues[0], (void *)rpaInfo, 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Fail to unmap shared memory object."); 
	status = clEnqueueSVMUnmap(queues[1], (void *)cia, 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Fail to unmap shared memory object.");
	status = clEnqueueSVMUnmap(queues[2], (void *)nextFrontierSize, 0, NULL, NULL); 
	if(status != CL_SUCCESS) ERROR("Fail to unmap shared memory object.");

#ifdef SW_EMU
	for(int i = 0; i < batch; i++){
		status = clEnqueueSVMUnmap(queues[2], (void *)(bmap[i]), 0, NULL, NULL); 
		if(status != CL_SUCCESS) ERROR("Fail to unmap shared memory object.");
	}
#endif

	for(int i = 0; i < batch; i++){
		status = clEnqueueSVMUnmap(queues[3 + i], (void *)(nextFrontier[i]), 0, NULL, NULL); 
		if(status != CL_SUCCESS) ERROR("Fail to unmap shared memory object.");
	}

	// Release kernels.
	for(auto k : kernels){
		clReleaseKernel(k);
	}
	clReleaseProgram(program);
	for(auto q : queues){
		clReleaseCommandQueue(q);
	}
	clReleaseContext(context);

	return 0;
}
