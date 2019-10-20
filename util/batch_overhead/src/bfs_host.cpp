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
// This work is mostly done in School of Computing, National University of 
// Singapore. I get a lot of support and suggestions from Xtra group lead by 
// Prof. Bingsheng He. Particularly, Xinyu Chen and Shengliang Lu in Xtra group 
// made many contributions to this work.
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
#define BATCH 4
#define BW 32
#define VMAX (16 * 1024 * 1024)

#ifdef SW_EMU
typedef unsigned int bmap_dt;
#endif

#define BMAP_DEPTH (VMAX / (BATCH * BW)) 
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
	else if(gName == "rmat-21-32")  fName = "rmat-21-32.txt";
	else if(gName == "rmat-19-32")  fName = "rmat-19-32.txt";
	else if(gName == "rmat-21-128") fName = "rmat-21-128.txt";
	else if(gName == "twitter")     fName = "twitter_rv.txt";
	else if(gName == "friendster")  fName = "friendster.ungraph.txt";
	else ERROR(" Unknown graph name %s .", gName.c_str());

	std::string fpath = dir + fName;
	return new Graph(fpath.c_str());
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

void cleanup(){}

int main(int argc, char ** argv){
	////////////////////////////////
	// Graph init
	////////////////////////////////
	int         batch   = std::stoi(argv[1]);
	std::string gName   = argv[2];
	std::cout << "batch = " << batch << std::endl;
	std::cout << "Graph is " << gName << std::endl;

	Graph* gptr = createGraph(gName);
	int originalVNum = gptr->vertexNum;
	int originalENum = gptr->edgeNum;

	CSR* csr = new CSR(*gptr, batch);
	free(gptr);
	int batchedENum = (int)(csr->ciao.size());

	double replRate = 0;
	// Edge number doubles as undirectional graph is converted to directional graphs
	if(gName == "youtube") 
		replRate = (originalVNum + batchedENum) * 1.0 / (originalVNum + originalENum * 2);
	else
		replRate = (originalVNum + batchedENum) * 1.0 / (originalVNum + originalENum);

	std::cout << "replciation rate is: " << replRate << std::endl;

	return 0;
}
