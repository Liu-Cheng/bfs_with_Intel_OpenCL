#include "graph.h"

void Graph::loadFile(
        const std::string& gName, 
        std::vector<std::vector<int>> &data
        )
{
    std::ifstream fhandle(gName.c_str());
    if(!fhandle.is_open()){
        HERE;
        std::cout << "Failed to open " << gName << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    while(std::getline(fhandle, line)){
        std::istringstream iss(line);
        data.push_back(
            std::vector<int>(std::istream_iterator<int>(iss),
            std::istream_iterator<int>())
        );
    }
    fhandle.close();
}

int Graph::getMaxIdx(const std::vector<std::vector<int>> &data){
    int maxIdx = data[0][0]; 
    for(auto it1 = data.begin(); it1 != data.end(); it1++){
        for(auto it2 = it1->begin(); it2 != it1->end(); it2++){            
            if(maxIdx <= (*it2)){
                maxIdx = *it2;
            }
        }
    }
    return maxIdx;
}

int Graph::getMinIdx(const std::vector<std::vector<int>> &data){
    int minIdx = data[0][0]; 
    for(auto it1 = data.begin(); it1 != data.end(); it1++){
        for(auto it2 = it1->begin(); it2 != it1->end(); it2++){            
            if(minIdx >= (*it2)){
                minIdx = *it2;
            }
        }
    }
    return minIdx;
}

Graph::Graph(const std::string& gName){

    // Check if it is undirectional graph
    auto found = gName.find("ungraph", 0);
    if(found != std::string::npos)
        isUgraph = true;
    else
        isUgraph = false;

    std::vector<std::vector<int>> data;
    loadFile(gName, data);
    vertexNum = getMaxIdx(data) + 1;
    edgeNum = (int)data.size();
    std::cout << "vertex num: " << vertexNum << std::endl;
    std::cout << "edge num: " << edgeNum << std::endl;

    for(int i = 0; i < vertexNum; i++){
        Vertex* v = new Vertex(i);
        vertices.push_back(v);
    }

    for(auto it = data.begin(); it != data.end(); it++){
        int srcIdx = (*it)[0];
        int dstIdx = (*it)[1];
        vertices[srcIdx]->outVid.push_back(dstIdx);
        vertices[dstIdx]->inVid.push_back(srcIdx);
        if(isUgraph && srcIdx != dstIdx){
            vertices[dstIdx]->outVid.push_back(srcIdx);
            vertices[srcIdx]->inVid.push_back(dstIdx);
        }
    }

    for(auto it = vertices.begin(); it != vertices.end(); it++){
        (*it)->inDeg = (int)(*it)->inVid.size();
        (*it)->outDeg = (int)(*it)->outVid.size();
    }
}

CSR::CSR(const Graph &g) 
	: vertexNum(g.vertexNum), edgeNum(g.edgeNum), padLen(0)
{
    rpao.resize(2 * vertexNum);
    rpai.resize(2 * vertexNum);
    for(int i = 0; i < vertexNum; i++)
	{
		int outDeg = g.vertices[i]->outDeg;
		int inDeg  = g.vertices[i]->inDeg;
		if(i == 0)
		{
			rpao[2 * i] = 0;
			rpai[2 * i] = 0;
		}
		else
		{
			rpao[2 * i] = rpao[2 * (i - 1)] + rpao[2 * (i - 1) + 1];
			rpai[2 * i] = rpai[2 * (i - 1)] + rpai[2 * (i - 1) + 1];
		}

		rpao[2 * i + 1] = outDeg;
		rpai[2 * i + 1] = inDeg;

		for(auto vid : g.vertices[i]->outVid){
            ciao.push_back(vid);
        }
        for(auto vid : g.vertices[i]->inVid){
            ciai.push_back(vid);
        }

    }
}


CSR::CSR(const Graph &g, const int &_padLen) : 
	vertexNum(g.vertexNum), edgeNum(g.edgeNum), padLen(_padLen)
{
	rpao.resize(2 * vertexNum);
	rpai.resize(2 * vertexNum);
	for(int i = 0; i < vertexNum; i++)
	{
		if(i == 0)
		{
			rpao[2 * i] = 0;
			rpai[2 * i] = 0;
		}
		else
		{
			rpao[2 * i] = rpao[2 * (i - 1)] + rpao[2 * (i - 1) + 1];
			rpai[2 * i] = rpai[2 * (i - 1)] + rpai[2 * (i - 1) + 1];
		}

		// update rpao and ciao
		std::vector<std::vector<int>> paddingVec;
		paddingVec.resize(padLen);
        for(auto vidx : g.vertices[i]->outVid)
		{
			int bankIdx = vidx % padLen;
			paddingVec[bankIdx].push_back(vidx);
		}

		int maxSize = 0;
		for(int j = 0; j < padLen; j++){
			int vecSize = (int)(paddingVec[j].size()); 
			if(vecSize > maxSize) maxSize = vecSize;
		}
		rpao[2 * i + 1] = maxSize * padLen;

		for(int j = 0; j < maxSize; j++){
			for(int k = 0; k < padLen; k++){
				if((int)(paddingVec[k].size()) > j){
					ciao.push_back(paddingVec[k][j]);
				}
				else{
					ciao.push_back(-1);
				}
			}
		}

		//ciai needs to be added later.	
	}
} 

