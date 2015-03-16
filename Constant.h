#pragma once;
#include <string>
using namespace std;

class Constant{
    //movielens
    public:
//        const static int userNum ;//= 943;
//        const static int itemNum ;//= 1682;
//        const static char* inFileName;// = "../recommendationreview/Data/movielens/movielens.dat";//polblogs1  realPolblogbeginfrom1
//        const static std::string DATAPATH;// = "../recommendationreview/Data/movielens/";
};
	const int g_nLength = 10000;
	//const int TOPL_CONFIG=20;
	//const int TOPL=50;
	const int HASHLENGTH= 3000;
	const static int notFound_number = -333;

    //movielens
    const static int userNum = 943;
    const static int itemNum = 1682;
    const static char* inFileName = "../recommendationreview/Data/movielens/movielens.dat";//polblogs1  realPolblogbeginfrom1
	const static std::string DATAPATH = "../recommendationreview/Data/movielens/";
	const static string resultfilepath = "F:/Documents/Dropbox/program/project/recomdationReviewVS/Result/movielens/";

	//netflix
	//const static int userNum = 10000;
	//const static int itemNum = 6000;
	////const static char* inFileName = "../recommendationreview/Data/netflix_pnas/netflix_pnas.txt";//
	////const static std::string DATAPATH = "../recommendationreview/Data/netflix_pnas/";
	//const static char* inFileName = "../recommendationreview/Data/netflix_pnas/netflix_pnas.txt";//
	//const static std::string DATAPATH = "../recommendationreview/Data/netflix_pnas/";
	//const static string resultfilepath = "F:/Documents/Dropbox/program/project/recomdationReviewVS/Result/netflix/";

    //amazon
	//const static int userNum = 9999;
	//const static int itemNum = 24403;//\recommendationreview\Data\amazon
	//const static char* inFileName = "../recommendationreview/Data/amazon1w/Amazon_reindex_1w.txt";//
	//const static std::string DATAPATH = "../recommendationreview/Data/amazon1w/";
	//const static string resultfilepath = "F:/Documents/Dropbox/program/project/recomdationReviewVS/Result/amazon1w/";

	//rym
	//const static int userNum = 33786;
	//const static int itemNum = 5381;
	//const static char* inFileName = "../recommendationreview/Data/RYM/RYMhigherrate.txt";//
	//const static std::string DATAPATH = "../recommendationreview/Data/RYM/";

	const static char* outFileName = "../answer/answer.txt";
	const static char* smartoutFileName = "D:/smartpolanswer.txt";
	//const static std::string configFile = "../recommendationreview/config.ini";
	const static std::string configFile = "config.ini";

	const bool doPrint = true;
	/*int topoNetwork[PointNum][PointNum];
	int oriTopoNetwork[PointNum][PointNum];*/
	//#define GETSIZE(c) (sizeof(c)/sizeof(c[0]));
