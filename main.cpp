#include "SimpleNetwork.h"
#include "operate_config.h"
#include "interSection.hpp"
#include <sstream>
#include <iterator>
using namespace std;

//int userNum = 943;
//int itemNum = 1682;
//char* inFileName = "../recomdationReviewVS/Data/movielens/movielens.dat";//polblogs1  realPolblogbeginfrom1
//std::string DATAPATH = "../recomdationReviewVS/Data/movielens/";

typedef Eigen::Triplet<double> TRIPLET;
typedef Eigen::SparseMatrix<double> SpMatrix;
vector<TRIPLET> tripletList;

double unSortedScore[itemNum];
//double lamada = 1;
double HybirditemScore[itemNum];

double minlamada, maxlamada, minsita, maxsita, mingama, maxgama,lamadaIncremental,gamaIncremental,sitaIncremental,totalstepts,TOPL = 50;
double RANKINGSCORE = 0.0,LOCALRS = 0.0,globalSeed = 1.0;
//string resultfilepath = "../recomdationReviewVS/Result/";
string funcName;
string resultfile;
int runingMode = 1;//runingMode = 0 is train mode, runingMode = 1 is test mode; only affect function RankingScoreNotCollect

int degreeLimited = 10;//local rankingscore 的degree限制
int fullcounter = 0, localcounter=0;
bool isOtherMathodRunning = false; 

SimpleNetwork oldNet(false);
//2303
SimpleNetwork trainingSet(false);
SimpleNetwork learningSet(false);
SimpleNetwork combainTrainAndLearningSet(false);
SimpleNetwork probSet(false);
//int userNum = 940,itemNum = 1430;
//int HeatSscore[userNum][itemNum];
//int ProbSscore[userNum][itemNum];
//int oriTopoNetwork[userNum][itemNum];
//int Hybirdscore[userNum][itemNum];//这是每个用户i,给他推荐的item列表，Hybirdscore[i][j]是对用户i来说排名j的的产品编号
////int UserRecommondatedItem[userNum][itemNum];
//int itemRankForUser[userNum][itemNum];//对于每个用户i,每个item（这里是index_array[itemIndex]）的排名就是itemindex,比如用户i来说产品j的排名就是itemRankForUser[i][j]
//double userRecommondatedItemScore[userNum][itemNum];//这是每个用户i,给他推荐的item分数，userRecommondatedItemScore[i][j]，是对用户i来说其推荐列表里产品j的产品的分数

//vector<vector <int> > oriTopoNetwork(userNum,vector<int>(itemNum,0));
//vector<vector <int> > Hybirdscore(userNum,vector<int>(itemNum,0));//这是每个用户i,给他推荐的item列表，Hybirdscore[i][j]是对用户i来说排名j的的产品编号
//vector<vector <int> > itemRankForUser(userNum,vector<int>(itemNum,0));//对于每个用户i,每个item（这里是index_array[itemIndex]）的排名就是itemindex,比如用户i来说产品j的排名就是itemRankForUser[i][j]
//vector<vector <double> > userRecommondatedItemScore(userNum,vector<double>(itemNum,0.0));//这是每个用户i,给他推荐的item分数，userRecommondatedItemScore[i][j]，是对用户i来说其推荐列表里产品j的产品的分数
//vector<vector <double> > MCFUserUserMatrix(userNum,vector<double>(userNum,0.0));//UCF里面用于更新用户的分数的矩阵。

//vector<vector <int> > itemRankForUser(userNum,vector<int>(itemNum,0));
vector<vector <int> > Hybirdscore(userNum,vector<int>(TOPL,0));
vector<vector <int> > itemRankForUser(userNum,vector<int>(itemNum,0));
//vector<int> itemRankForUser(itemNum,0);
vector<vector <double> > userRecommondatedItemScore;//(userNum,vector<double>(itemNum,0.0));
//vector<vector <double> > MCFUserUserMatrix;//(userNum,vector<double>(itemNum,0.0));

vector<double> sngleLamadaArray(userNum);
vector<int> itemsCommonNeighbor;
//DynamicSparseMatrix<double>;
//Matrix<double,userNum,itemNum> exchangeMatrix;
//Matrix<double,Dynamic,Dynamic> transformationMatrix;

//-----------------------------------------------------------------------------
//Matrix< double , Dynamic , Dynamic > transformationMatrix(0,0);
//Matrix< double , Dynamic , Dynamic > transformationMatrix(itemNum,itemNum);
//------------------------------------------------------------------------------

//Matrix<double,itemNum,itemNum> transformationMatrix(itemNum,itemNum);

//extern int Hybirdscore[userNum][itemNum];
//extern int UserRecommondatedItem[userNum][itemNum];
//extern int itemRankForUser[userNum][itemNum];
//extern double userRecommondatedItemScore[userNum][itemNum];
//vector<vector<int>> resultM(m, n);
//extern vector<vector <int> > Hybirdscore; //m*n的二维vector，所有元素初始化为0
//extern vector<vector <int> > itemRankForUser;
//extern vector<int> itemRankForUser;
//extern vector<vector <double> > userRecommondatedItemScore;
//extern vector<double> sngleLamadaArray;
//extern vector<int> itemsCommonNeighbor;

//double randScoreArray[PointNum];
//int newNetMatrix[PointNum][PointNum];
struct paraGroup
{
	int times;
	double lamada;
	double sita;
	double gama;
	int isTheBestTimes;
	double oldRS;
	double rs;
	double lrs;
	double pricis;
	double recall;
	double intrSim;
	double hamdis;
	double Popul;
};

struct Ranker
{
	long int id;
	double value;
	bool operator ==(const Ranker &d1)
	{
		return (d1.id == this->id);
	}
};

vector<paraGroup> paraGroupVector;//(10);

bool Greater(const Ranker & d1, const Ranker & d2)
{
	return d1.value < d2.value;
};
bool Smaller(const Ranker & d1, const Ranker & d2)
{
	return d1.value > d2.value;
};
string generateFilename(string filename = resultfilepath)
{
	time_t curtime=time(0);
	//tm tim = *localtime(&curtime);
	tm tim;
	localtime_s(&tim,&curtime);
	int day,mon,year,hour,minite;
	day=tim.tm_mday;
	mon=tim.tm_mon;
	year=tim.tm_year;
	hour = tim.tm_hour;
	minite = tim.tm_min;
	stringstream path;
	path<<filename<<mon+1<<"_"<<day<<" "<<hour<<"."<<minite<<".txt";
	return path.str();
}

string generateFiledate()
{
	time_t curtime=time(0);
	//tm tim = *localtime(&curtime);
	tm tim;
	localtime_s(&tim, &curtime);
	int day,mon,year,hour,minite;
	day=tim.tm_mday;
	mon=tim.tm_mon;
	year=tim.tm_year;
	hour = tim.tm_hour;
	minite = tim.tm_min;
	stringstream path;
	path<<mon+1<<"_"<<day<<" "<<hour<<"."<<minite;
	return path.str();
}

void dividetotwoset(SimpleNetwork &network = oldNet)
{
	oldNet.countDegree();
	srand( time(NULL) );        // init the random generator
	double number;
	int item = 0;
	for (int i = 0; i< userNum; i++)
	{
		for (unsigned int j = 0; j < network.user_item_relation[i].size(); j++)
		{
			number = (rand() % 100 )/100.0;
			item = network.user_item_relation[i][j];
			if (number<0.9 || network.user_item_relation[i].size()<=1 || network.item_user_relation[item].size()<=1)
			{
				trainingSet.addEdge(i,item);
				//cout<<"add to train: "<<i<<" "<<item<<endl;
			}
			else
			{
				learningSet.addEdge(i,item);
				//cout<<"add to learn: "<<i<<" "<<item<<endl;
			}
		}
	}
	trainingSet.countDegree();
	learningSet.countDegree();
	probSet.countDegree();
}

void divideto3set(SimpleNetwork &network = oldNet)
{
	srand( time(NULL) );        // init the random generator
	double number;
	int item = 0;
	for (int i = 0; i< userNum; i++)
	{
		for (unsigned int j = 0; j < network.user_item_relation[i].size(); j++)
		{
			number = (rand() % 100 )/100.0;
			item = network.user_item_relation[i][j];
			if (number<0.8 || network.user_item_relation[i].size()<=1 || network.item_user_relation[item].size()<=1)
			{
				trainingSet.addEdge(i,item);
				combainTrainAndLearningSet.addEdge(i,item);
			}
			else
			{
				if (number>=0.9)
				{
					probSet.addEdge(i,item);
				}
				else// if (number>0.8)
				{
					learningSet.addEdge(i,item);
					combainTrainAndLearningSet.addEdge(i,item);
				}
			}
		}
	}
	trainingSet.countDegree();
	trainingSet.printNetwork("D:/spreading/network/movielens/train.txt");
	probSet.countDegree();
	probSet.printNetwork("D:/spreading/network/movielens/prob.txt");
	learningSet.countDegree();
	learningSet.printNetwork("D:/spreading/network/movielens/learning.txt");
	combainTrainAndLearningSet.countDegree();
	combainTrainAndLearningSet.printNetwork("D:/spreading/network/movielens/newNet.txt");
}

void divideto3set10Times(SimpleNetwork &network = oldNet)
{
	srand( time(NULL) );        // init the random generator
	double number;
	int item = 0;
	//char filenamenumbers[5];
	//char buffer[80];
	//string path = DATAPATH;
	oldNet.loadNetworkFromFile(inFileName);
	//string path = "D:/spreading/network/delicious_pnas/";
	for (int times = 0; times<10; times++)
	{
		trainingSet.empty();
		probSet.empty();
		learningSet.empty();
		combainTrainAndLearningSet.empty();
		for (int i = 0; i< userNum; i++)
		{
			for (unsigned int j = 0; j < network.user_item_relation[i].size(); j++)
			{
				number = (rand() % 100 )/100.0;
				item = network.user_item_relation[i][j];
				if (number<0.8 || network.user_item_relation[i].size()<=1 ||network.item_user_relation[item].size()<=1)
				{
					trainingSet.addEdge(i,item);
					combainTrainAndLearningSet.addEdge(i,item);
				}
				else
				{
					if (number>=0.9)
					{
						probSet.addEdge(i,item);
					}
					else// if (number>0.8)
					{
						learningSet.addEdge(i,item);
						combainTrainAndLearningSet.addEdge(i,item);
					}
				}
			}
		}
		//trainingSet.printNetwork("D:/spreading/network/movielens/train.txt");
		//itoa(times,filenamenumbers,10);

		trainingSet.countDegree();
		stringstream path;
		path<<DATAPATH<<"train"<<times<<".txt";
		trainingSet.printNetwork(path.str().c_str());
		path.str("");

		probSet.countDegree();
		path<<DATAPATH<<"prob"<<times<<".txt";
		//sprintf(buffer, "%s%s%d%s", DATAPATH, "prob", times,".txt");
		probSet.printNetwork(path.str().c_str());
		path.str("");

		learningSet.countDegree();
		path<<DATAPATH<<"learning"<<times<<".txt";
		learningSet.printNetwork(path.str().c_str());
		path.str("");

		combainTrainAndLearningSet.countDegree();
		path<<DATAPATH<<"newNet"<<times<<".txt";
		combainTrainAndLearningSet.printNetwork(path.str().c_str());
		path.str("");
	}
}

void intiPramate()
{
	operate_config opc(configFile);
	minlamada = opc.getNumber("minlamada");
	maxlamada = opc.getNumber("maxlamada");
	lamadaIncremental = opc.getNumber("lamadaincremental");
	minsita = opc.getNumber("minsita");
	maxsita = opc.getNumber("maxsita");
	sitaIncremental = opc.getNumber("sitaincremental");
	mingama = opc.getNumber("mingama");
	maxgama = opc.getNumber("maxgama");
	gamaIncremental = opc.getNumber("gamaincremental");
	if (opc.getNumber("degreelimited")!=notFound_number)
	{
		degreeLimited = opc.getNumber("degreelimited");
	}
	cout<<"degreeLimited: "<<degreeLimited<<endl;
	cout<<"minlamada: "<<minlamada<<" maxlamada: "<<maxlamada<<" lamadaIncremental: "<<lamadaIncremental<<endl;
	cout<<"minsita: "<<minsita<<" maxsita: "<<maxsita<<" sitaIncremental: "<<sitaIncremental<<endl;
	cout<<"mingama: "<<mingama<<" maxgama: "<<maxgama<<" gamaIncremental: "<<gamaIncremental<<endl;
}

void init()
{
	//divideto3set();
	//dividetotwoset();
	//intiPramate();
	oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem
	string datapath = DATAPATH;
	trainingSet.loadNetworkFromFile(datapath+"train0.txt");
	probSet.loadNetworkFromFile(datapath+"prob0.txt");
	learningSet.loadNetworkFromFile(datapath+"learning0.txt");
	combainTrainAndLearningSet.loadNetworkFromFile(datapath+"newNet0.txt");
	/*cout<<oldNet.getUserMaxDegree()<<"	old	"<<oldNet.getItemMaxDegree()<<endl;
	cout<<trainingSet.getUserMaxDegree()<<"	trainingSet	"<<trainingSet.getItemMaxDegree()<<endl;
	cout<<probSet.getUserMaxDegree()<<"	probSet	"<<probSet.getItemMaxDegree()<<endl;
	cout<<combainTrainAndLearningSet.getUserMaxDegree()<<"	combainTrainAndLearningSet	"<<combainTrainAndLearningSet.getItemMaxDegree()<<endl;*/
	//trainingSet.MatrixHybird(0.3);
}

void init(int times)
{
	trainingSet.empty();
	probSet.empty();
	learningSet.empty();
	combainTrainAndLearningSet.empty();
	//oldNet.empty();
	//oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem
	//char* path = "../recomdation/Data/movielens/";//../recomdation/Data/movielens/
	string path;
	//oldNet.loadNetworkFromFile(inFileName);
	stringstream ss;

	ss<<DATAPATH<<"train"<<times<<".txt";
	path = ss.str();
	trainingSet.loadNetworkFromFile(path);
	ss.clear();
	ss.str("");

	ss<<DATAPATH<<"prob"<<times<<".txt";
	path = ss.str();
	probSet.loadNetworkFromFile(path);
	ss.clear();
	ss.str("");

	ss<<DATAPATH<<"learning"<<times<<".txt";
	path = ss.str();
	learningSet.loadNetworkFromFile(path);
	ss.clear();
	ss.str("");

	ss<<DATAPATH<<"newNet"<<times<<".txt";
	path = ss.str();
	combainTrainAndLearningSet.loadNetworkFromFile(path);
	ss.clear();
	ss.str("");
	//intiPramate();
}
static void debugMeg(string meg)
{
	if(doPrint)
		cout<<meg;
}
int gezb_bsearch(double Source[], int Low, int Len, double Key)
{
	int low, high=Len-1;
	low=Low;
	if (Key<Source[low])
	{
		return low;
		cout<<"low "<<low<<endl;
	}
	if (Key>Source[high-1])
	{
		//std::cout<<"can't find ! key="<<Key<<",low["<<low<<"]="<<Source[low]<<",high["<<high<<"]="<<Source[high]<<std::endl;
		return high;
	}
	//if (Key<Source[low] || Key>Source[high])
	//{
	//	//std::cout<<"can't find ! key="<<Key<<",low["<<low<<"]="<<Source[low]<<",high["<<high<<"]="<<Source[high]<<std::endl;
	//	return -1;
	//}
	while(low<=high)
	{
		//std::cout<<"low["<<low<<"]="<<Source[low]<<",high["<<high<<"]="<<Source[high]<<std::endl;
		int middle = (low + high)/2;
		if (Source[middle]<=Key&&Source[middle+1]>=Key)
		{
			//std::cout<<"ok get it. \ncurIndex="<<middle<<", key="<<Key<<std::endl;
			return middle+1;
		}
		else if (Key>Source[middle])
		{
			low = middle+1;
		}
		else if (Key<Source[middle])
		{
			high = middle-1;
		}
	}
	return -1;
}

int gezb_bsearch_recurse(double Source[],int Low,int High,double Key)
{
	/*if (Key<Source[Low] || Key>Source[High])
	{
		std::cout<<"can't find ! key="<<Key<<",low["<<Low<<"]="<<Source[Low]<<",high["<<High<<"]="<<Source[High]<<std::endl;
		return -1;
	}*/
	if (Key<Source[Low])
	{
		return Low;
	}
	if (Key>Source[High-1])
	{
		//std::cout<<"can't find ! key="<<Key<<",low["<<low<<"]="<<Source[low]<<",high["<<high<<"]="<<Source[high]<<std::endl;
		return High;
	}
	int middle;
	if (Low<=High)
	{
		//std::cout<<"low["<<Low<<"]="<<Source[Low]<<",high["<<High<<"]="<<Source[High]<<std::endl;
		middle = (Low+High)/2;
		if (Source[middle]<=Key&&Source[middle+1]>=Key)
		{
			//std::cout<<"ok get it. \ncurIndex="<<middle<<", key="<<Key<<std::endl;
			return middle+1;
		}
		else if (Source[middle]>Key)
		{
			return gezb_bsearch_recurse(Source,Low,middle-1,Key);
		}
		else if (Source[middle]<Key)
		{
			return gezb_bsearch_recurse(Source,middle+1,High,Key);
		}
	}
	return -1;
}

//friends function implements
int hashInterSection(SimpleHashSet<int> *pHashSet, int a[], int m, int b[], int n)
{
	itemsCommonNeighbor.clear();
	int commonItemNumber = 0;
	for(int i = 0; i < m; i++)
	{
		if (!pHashSet->InsertHash(a[i]))
		{
			++commonItemNumber;
			itemsCommonNeighbor.push_back(a[i]);
		}
	}

	for(int j = 0; j < n; j++)
	{
		if (!pHashSet->InsertHash(b[j]))
		{
			++commonItemNumber;
			itemsCommonNeighbor.push_back(b[j]);
		}
	}
	//cout<<commonItemNumber<< endl;
	return commonItemNumber;
}
int hashInterSection(SimpleHashSet<int>* pHashSet, int a[], int m, const vector<int> &b, int n)
{
	itemsCommonNeighbor.clear();
	int commonItemNumber = 0;
	for(int i = 0; i < m; i++)
	{
		if (!pHashSet->InsertHash(a[i]))
		{
			++commonItemNumber;
			itemsCommonNeighbor.push_back(a[i]);
		}
	}

	for(int j = 0; j < n; j++)
	{
		if (!pHashSet->InsertHash(b[j]))
		{
			++commonItemNumber;
			itemsCommonNeighbor.push_back(b[j]);
		}
	}
	//cout<<commonItemNumber<< endl;
	return commonItemNumber;
}
int hashInterSection(SimpleHashSet<int>* pHashSet, const vector<int> &a, int m, const vector<int> &b, int n)
{
	itemsCommonNeighbor.clear();
	int commonItemNumber = 0;
	for(int i = 0; i < m; i++)
	{
		if (!pHashSet->InsertHash(a[i]))
		{
			++commonItemNumber;
			itemsCommonNeighbor.push_back(a[i]);
		}
	}
	for(int j = 0; j < n; j++)
	{
		//cout<<"b[j] is:	"<<b[j]<<"	n is:	"<<n<<endl;
		if (!pHashSet->InsertHash(b[j]))
		{
			++commonItemNumber;
			itemsCommonNeighbor.push_back(b[j]);
		}
	}
	//cout<<commonItemNumber<< endl;
	return commonItemNumber;
}
int hashInterSection(SimpleHashSet<int>* pHashSet, const vector<int> &a, int m, int b[], int n)
{
	itemsCommonNeighbor.clear();
	int commonItemNumber = 0;
	for(int i = 0; i < m; i++)
	{
		if (!pHashSet->InsertHash(a[i]))
		{
			++commonItemNumber;
			itemsCommonNeighbor.push_back(a[i]);
		}
	}

	for(int j = 0; j < n; j++)
	{
		if (!pHashSet->InsertHash(b[j]))
		{
			++commonItemNumber;
			itemsCommonNeighbor.push_back(b[j]);
		}
	}
	//cout<<commonItemNumber<< endl;
	return commonItemNumber;
}
int getCommonObjectsFromSortedarray(const  int * pArray1 , int iSize1 , const int * pArray2 , int iSize2 )
{
	const int * piFirEnd = pArray1+iSize1 , *piSecEnd = pArray2+iSize2;
	int iRet = 0;

	while ( pArray1 < piFirEnd && pArray2 < piSecEnd )
	{
		if ( *pArray1 ==  *pArray2)
		{
			printf("%d ", *pArray1);

			pArray1++;
			pArray2++;
			iRet++;
		}
		else if (  *pArray1 <  *pArray2 )
		{
			pArray1++;
		}
		else
		{
			pArray2++;
		}
	}
	return iRet;
}

int getCommonObjects(const  int pArray1[] , int iSize1 , const int pArray2[] , int iSize2 )
{
	vector<int> v1(pArray1, pArray1 + iSize1);
	vector<int> v2(pArray2, pArray2 + iSize2);
	vector<int> v(50);
	vector<int>::iterator it;
	sort(v1.begin(), v1.end());
	sort(v2.begin(), v2.end());
	it = set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), v.begin());
	//cout <<"The number of the same elements is: "<<(it - vr_less.begin()) <<endl;
	return int(it - v.begin());
}

double CosinSimilarity(int i, int j,SimpleNetwork &testSet = combainTrainAndLearningSet)
{
	int sizei,sizej, intersection;
	sizei = testSet.item_user_relation[i].size();
	if (sizei == 0)
	{
		return 0;
	}
	sizej = testSet.item_user_relation[j].size();
	if (sizej == 0)
	{
		return 0;
	}
	SimpleHashSet<int> SimpleHashSet((sizei+sizej)*10);
	intersection = hashInterSection(&SimpleHashSet,testSet.item_user_relation[i],sizei,testSet.item_user_relation[j],sizej);
	SimpleHashSet.SetLength(0);
	//cout<<"intersection : "<<sizei*sizej*1.0<<endl;
	return intersection/sqrt(sizei*sizej*1.0);
}
int cmp_by_index(const void * a, const void * b)
{
	double* ptsrcbuf = unSortedScore;

	int * _a = (int*)a, *_b = (int*)b;
	//如果该item对应Score过大(比如大于itemNum)，也许程序有问题
	//if (ptsrcbuf[*_b]>itemNum)
	//{
	//	cout<<"item "<<* _b<<"'s score is: "<<ptsrcbuf[*_b]<<endl;
	//}

	/*cout<<" * _a "<<* _a<<" = "<<ptsrcbuf[*_a]<<endl;
	cout<<" * _b "<<* _b<<" = "<<ptsrcbuf[*_b]<<endl;*/
	if (ptsrcbuf[*_b] > ptsrcbuf[*_a])
	{
		return 1;
	}else
		return -1;
}

int sortItemIdByScore(double *arr123)
{
	//std::copy(std::begin(unSortedScore), std::end(unSortedScore), std::begin(unSortedScore));
	//std::copy(arr123->begin(), arr123->end(), std::begin(unSortedScore));
	int index_array[userNum];
	for (int j1 = 0; j1 < userNum; ++j1)
		index_array[j1] = j1;
	qsort(index_array, userNum, sizeof(index_array[0]), cmp_by_index);
	return 0;
}

template<typename T>
int emptyVector(vector< vector<T> >* vec)
{
	for (typename vector< vector<T> >::iterator itr = vec->begin(); itr!=vec->end();++itr)
	{
		(itr)->clear();
	}
	return 0;
}

double RankingScoreNotCollect(int i, vector<int> &rankForUser,int recomListSize,SimpleNetwork &inputSet = probSet)
{
	//cout<<__FUNCTION__<<" i "<<i<<endl;
	SimpleNetwork *testSet;
	SimpleNetwork *xunlianSet;//这里是用来选择localrankingscore的时候选择度小于某的点的，这个度小于多少是指训练集
	if (runingMode == 0)
	{
		testSet = &learningSet;
		xunlianSet = &trainingSet;
	}else
	{
		testSet = &inputSet;
		xunlianSet = &combainTrainAndLearningSet;
	}
	//cout<<testSet->countDegree()<<"	";
	double temprankscore = 0.0,templocalrs = 0.0;
	int tempItem=0,size=0,userRemedSmallerDegreeItmeCounter = 0,localItemSize = 0;
	double notCollectSize = itemNum-oldNet.user_item_relation[i].size();
	double notRecommdatedPos = (( notCollectSize - recomListSize ) / 2 + recomListSize);
	size = testSet->user_item_relation[i].size();
	if(size>0)
	{
		fullcounter++;
		for (int j = 0; j < size; j++)
		{
			tempItem = testSet->user_item_relation[i][j];
			localItemSize = xunlianSet->item_user_relation[tempItem].size();
			double tempPosition=0.0;
			if (rankForUser[tempItem] == 0)
			{
				//不在推荐列表里面，那么他的位置就是所的uncollected的item减去对齐推荐的列表长度，剩下的就是得分为0的item了，取个平均值在加上ComputeRankScore.size
				tempPosition = (notRecommdatedPos)/notCollectSize;
				//cout<< rankForUser[tempItem]<<"	"<<notRecommdatedPos<<" notCollectSize	"<<notCollectSize<<"	"<<tempPosition<<" localItemSize	"<<localItemSize<<endl;
			}else
			{
				tempPosition = (rankForUser[tempItem])/(notCollectSize);//排名从1开始的，所以不用+1
			}
			if (tempPosition>1)
			{
				tempPosition = 1;
			}
			temprankscore += tempPosition;
			if(localItemSize<=degreeLimited)
			{
				userRemedSmallerDegreeItmeCounter++;
				//userRemedSmallerDegreeItmeCounter++;
				templocalrs += tempPosition;	
				//if (rankForUser[tempItem] == 0)
				//{
				//	//test why localrs alway smaller tham rs
				//	cout<< rankForUser[tempItem]<<"	"<<notRecommdatedPos<<" notCollectSize	"<<notCollectSize<<"	"<<tempPosition<<" localItemSize	"<<localItemSize<<endl;
				//}//cout<<temprankscore<<"	lR	"<<templocalrs<<" localcounter "<<localcounter<<endl;
			}
		}
		RANKINGSCORE += temprankscore/size;
		if (userRemedSmallerDegreeItmeCounter > 0)
		{
			localcounter++;
			LOCALRS += templocalrs / userRemedSmallerDegreeItmeCounter;//;
			//cout << RANKINGSCORE << "	lR	" << LOCALRS << " localcounter " << userRemedSmallerDegreeItmeCounter << " fullcounter " << size << endl;
	}
	}
	return RANKINGSCORE;
}

double RankingScoreNotCollect(SimpleNetwork &testSet = probSet)
{
	double rs = 0.0;
	double tempPosition=0;
	int tempItem=0,size=0,counter = 0,notCollectSize = 0;
	for (int i = 0; i< userNum; i++)
	{
		notCollectSize = itemNum-oldNet.user_item_relation[i].size();

		size = testSet.user_item_relation[i].size();
		if (size>0)
		{
			for (int j = 0; j < size; j++)
			{
				counter++;
				tempItem = testSet.user_item_relation[i][j];
				tempPosition = (itemRankForUser[i][tempItem]+1.0)/(notCollectSize);
				//tempPosition = (itemRankForUser[i][tempItem]+1.0)/(itemNum-combainTrainAndLearningSet.user_item_relation[i].size());
				//tempPosition = (itemRankForUser[i][tempItem]+1.0)/(itemNum);
				if (tempPosition>1)
				{
					//cout<<"tempPosition "<<itemRankForUser[i][tempItem]+1.0<<"	"<<(itemNum-oldNet.user_item_relation[i].size())<<"	"<<tempPosition<<"	"<<endl;
					tempPosition = 1;
				}

				/*if (trainingSet.itemRankForUser[i][tempItem]==0||trainingSet.itemRankForUser[i][tempItem]>=1681)
				{
					cout<<" user " << i <<" item  " << tempItem <<" rank is "<<trainingSet.itemRankForUser[i][tempItem]<<endl;
					cout<<"tempPosition is : "<<tempPosition<<endl;
				}*/

				//cout<<""<<trainingSet.itemRankForUser[i][tempItem]<<" "<<itemNum-learningSet.network.user_item_relation[i].size()<<endl;
				rs += tempPosition;
				//cout<<"tempPosition is : "<<tempPosition<<endl;
			}
		}
	}
	//return rs;
	return rs/counter;
}

double LocalRankingScoreNotCollect(SimpleNetwork &testSet = probSet, int degreeLimited = 5)
{
	double rs = 0.0;
	double tempPosition=0;
	int tempItem=0,size=0,counter = 0,notCollectSize = 0;
	for (int i = 0; i< userNum; i++)
	{
		notCollectSize = itemNum-oldNet.user_item_relation[i].size();
		size = testSet.user_item_relation[i].size();
		int userSize = trainingSet.user_item_relation[i].size();
		if (size>0)
		{
			if (userSize>degreeLimited)
			{
				continue;
			}
			for (int j = 0; j < size; j++)
			{
				counter++;
				tempItem = testSet.user_item_relation[i][j];
				tempPosition = (itemRankForUser[i][tempItem]+1.0)/(notCollectSize);
				//tempPosition = (itemRankForUser[i][tempItem]+1.0)/(itemNum-combainTrainAndLearningSet.user_item_relation[i].size());
				//tempPosition = (itemRankForUser[i][tempItem]+1.0)/(itemNum);
				if (tempPosition>1)
				{
					//cout<<"tempPosition "<<itemRankForUser[i][tempItem]+1.0<<"	"<<(itemNum-combainTrainAndLearningSet.user_item_relation[i].size())<<"	"<<tempPosition<<"	"<<Hybirdscore[i][tempItem]<<endl;
					tempPosition = 1;
				}
				rs += tempPosition;
				//cout<<"tempPosition is : "<<tempPosition<<endl;
			}
		}
	}
	//return rs;
	return rs/counter;
}

double RankingScore(SimpleNetwork &testSet = probSet)
{
	double rs = 0.0;
	double tempPosition=0;
	int tempItem=0,size=0,counter = 0;
	for (int i = 0; i< userNum; i++)
	{
		size = testSet.user_item_relation[i].size();
		if (size>0)
		{
			for (int j = 0; j < size; j++)
			{
				counter++;
				tempItem = testSet.user_item_relation[i][j];
				tempPosition = (itemRankForUser[i][tempItem]+1.0)/(itemNum);
				//tempPosition = (itemRankForUser[i][tempItem]+1.0)/(itemNum-combainTrainAndLearningSet.user_item_relation[i].size());
				//tempPosition = (itemRankForUser[i][tempItem]+1.0)/(itemNum);
				if (tempPosition>1)
				{
					cout<<"colltempPosition "<<itemRankForUser[i][tempItem]+1.0<<"	"<<itemNum<<"	"<<tempPosition<<"	"<<endl;
					//cout<<"tempPosition "<<itemRankForUser[i][tempItem]+1.0<<"	"<<(itemNum-combainTrainAndLearningSet.user_item_relation[i].size())<<"	"<<tempPosition<<"	"<<Hybirdscore[i][tempItem]<<endl;
					tempPosition = 1;
				}

				/*if (trainingSet.itemRankForUser[i][tempItem]==0||trainingSet.itemRankForUser[i][tempItem]>=1681)
				{
					cout<<" user " << i <<" item  " << tempItem <<" rank is "<<trainingSet.itemRankForUser[i][tempItem]<<endl;
					cout<<"tempPosition is : "<<tempPosition<<endl;
				}*/

				//cout<<""<<trainingSet.itemRankForUser[i][tempItem]<<" "<<itemNum-learningSet.network.user_item_relation[i].size()<<endl;
				rs += tempPosition;
				//cout<<"tempPosition is : "<<tempPosition<<endl;
			}
		}
	}
	//return rs;
	return rs/counter;
}

double SingleUserRankingScore(int user, SimpleNetwork &testSet = probSet)
{
	double rs = 0.0;
	double tempPosition=0;
	int tempItem=0,size=0,counter = 0;
	size = testSet.user_item_relation[user].size();
	if (size>0)
	{
		for (int j = 0; j < size; j++)
		{
			counter++;
			tempItem = testSet.user_item_relation[user][j];
			tempPosition = (itemRankForUser[user][tempItem]+1.0)/(itemNum-oldNet.user_item_relation[user].size());
			/*if (trainingSet.itemRankForUser[i][tempItem]==0||trainingSet.itemRankForUser[i][tempItem]>=1681)
			{
				cout<<" user " << i <<" item  " << tempItem <<" rank is "<<trainingSet.itemRankForUser[i][tempItem]<<endl;
				cout<<"tempPosition is : "<<tempPosition<<endl;
			}*/

			//cout<<""<<trainingSet.itemRankForUser[i][tempItem]<<" "<<itemNum-learningSet.network.user_item_relation[i].size()<<endl;
			rs += tempPosition;
			//cout<<"tempPosition is : "<<tempPosition<<endl;
		}
	}
	//return rs;
	return rs/counter;
}

double getRankingScore()
{
	double re = RANKINGSCORE/fullcounter;
	//cout<<RANKINGSCORE<<"	"<<fullcounter<<endl;
	RANKINGSCORE = 0;
	//localcounter = fullcounter;
	fullcounter = 0;
	return re;
}

double getLocalRankingScore()
{
	double re = LOCALRS/localcounter;
	//cout<<LOCALRS<<"	"<<localcounter<<endl;
	LOCALRS = 0;
	localcounter = 0;
	return re;
}

double Precision(SimpleNetwork &testSet = probSet,int topL=TOPL)
{
	double tempprecision=0.0;
	//tempprecision = precision;
	int size = 0, counter = 0, temp=0;
	vector<int> tempvector(topL,0);
	for (int i = 0; i < userNum; i++)
	{
		size = testSet.user_item_relation[i].size();
		if (size>0)
		{
			counter++;
			SimpleHashSet<int> SimpleHashSet((size+topL)*3);
			/*cout<<Hybirdscore[i].size()<<"	"<<Hybirdscore[i][i]<<"	";
			for (int j = 0;j<topL;j++)
			{
				tempvector[j] = Hybirdscore[i][j];
			}*/
			temp = hashInterSection(&SimpleHashSet,testSet.user_item_relation[i],size,Hybirdscore[i],topL);
			SimpleHashSet.SetLength(0);
			//cout<<"size "<<size<<endl;
			tempprecision += temp;
			//cout<<"has same "<<temp<<endl;
		}
	}
	return tempprecision/(counter*topL);
}

double HammingDistance(SimpleNetwork &testSet = combainTrainAndLearningSet, int topL=TOPL)
{
	double hamDs=0, hamTemp=0;
	int commonItems = 0,counter = 0,size1 = 0,size2=0;
	for (int i = 0; i< userNum; i++)
	{
		size1 = testSet.user_item_relation[i].size();
		if (size1 = 0)
		{
			continue;
		}
		for (int j = 0; j < i; j++)
		{
			size2 = testSet.user_item_relation[j].size();
			if (size2 = 0)
			{
				continue;
			}
			counter++;
			SimpleHashSet<int> SimpleHashSet((size1+size2+TOPL)*4);
			commonItems = hashInterSection(&SimpleHashSet,Hybirdscore[i],topL,Hybirdscore[j],topL);
			SimpleHashSet.SetLength(0);
			/*hamDs = getCommonObjects(Hybirdscore[i],topL,Hybirdscore[j],topL);
			if (hamDs != commonItems)
			{
			cout <<"The number of the same elements is: "<<hamDs <<" "<<commonItems<<endl;
			}*/
			hamTemp += (1-commonItems*1.0/topL);
			//cout<<"hamTemp "<<hamTemp <<" "<<counter<< " "<<(1-commonItems*1.0/topL)<<endl;
		}
	}
	return hamTemp/counter;
	//return hamTemp/((userNum-1)*userNum/2);
}

double IntraSimilarity(SimpleNetwork &testSet = probSet, int topL=TOPL)
{
	double is=0.0, isTemp=0.0;
	int commonItems = 0,counter = 0,size = 0;
	for (int i = 0; i< userNum; i++)
	{
		size = testSet.user_item_relation[i].size();
		if (size = 0)
		{
			continue;
		}
		counter++;
		for (int j = 0; j< topL; j++)
		{
			for (int k = 0; k< j; k++)
			{
				isTemp += CosinSimilarity(Hybirdscore[i][j],Hybirdscore[i][k]);
				commonItems++;
			}
		}
		//is += isTemp/((topL-1)*topL);
		is += isTemp/commonItems;
		//cout<<" isTemp is : "<<commonItems<<" "<<(topL-1)*topL<<endl;
		isTemp = 0.0;
		commonItems =0;
		//cout<<" isTemp is : "<<is<<endl;
	}
	return is/counter;
}

double Popularity(SimpleNetwork &testSet = combainTrainAndLearningSet, int topL=TOPL)
{
	int degree = 0,counter = 0,size = 0;
	for (int i = 0; i < userNum; i++)
	{
		size = testSet.user_item_relation[i].size();
		if (size>0)
		{
			counter++;
			for (int j = 0; j<topL; j++)
			{
				//degree+= Hybirdscore[i][j].size();
				degree+= testSet.item_user_relation[Hybirdscore[i][j]].size();
			}
		}
	}
	return degree*1.0/(topL*counter);
}

double novelty(SimpleNetwork &testSet = combainTrainAndLearningSet, int topL=TOPL)
{
	int degree = 0,counter = 0,size = 0;
	for (int i = 0; i < userNum; i++)
	{
		size = testSet.user_item_relation[i].size();
		if (size>0)
		{
			counter++;
			for (int j = 0; j<topL; j++)
			{
				//degree+= Hybirdscore[i][j].size();
				degree+= log(userNum*1.0/(testSet.item_user_relation[Hybirdscore[i][j]].size()))/log(2.0);
			}
		}
	}
	return degree*1.0/(topL*counter);
}

double Recall(SimpleNetwork &testSet = probSet,int topL=TOPL)
{
	double tempRecall=0.0;
	//tempprecision = precision;
	int size = 0, counter = 0, temp=0;
	for (int i = 0; i < userNum; i++)
	{
		size = testSet.user_item_relation[i].size();
		if (size>0)
		{
			counter++;
			SimpleHashSet<int> SimpleHashSet(HASHLENGTH);
			temp = hashInterSection(&SimpleHashSet,testSet.user_item_relation[i],size,Hybirdscore[i],topL);
			SimpleHashSet.SetLength(0);
			if (temp == 0)
			{
				continue;
			}
			tempRecall += temp*1.0/testSet.user_item_relation[i].size();		
			//cout<<"has same "<<temp<<endl;
		}
	}
	//cout<<"tempRecall "<<tempRecall<<"	"<<counter<<"	"<<topL<<endl;
	return tempRecall/(counter);
}

vector<double> ComputeAccuracy(int topL = TOPL,SimpleNetwork &testSet = probSet)
{
	double precision = 0;
	double recall = 0;
	double f1 = 0;
	double hitCount = 0;
	double tempRecall = 0;
	vector<double> accuracyVector(3,0);
	int counter = 0;
	for (int i = 0; i < userNum; i++)
	{
		vector<double> tempAccuracyVector(3,0);
		int size = testSet.user_item_relation[i].size();
		if (size>0)
		{
			counter++;
			SimpleHashSet<int> SimpleHashSet(HASHLENGTH);
			hitCount = hashInterSection(&SimpleHashSet,testSet.user_item_relation[i],size,Hybirdscore[i],topL);
			SimpleHashSet.SetLength(0);
			if (hitCount == 0)
			{
				continue;
			}
			tempRecall += hitCount/testSet.user_item_relation[i].size();		
			precision = hitCount/TOPL;
			//precision = hitCount;
			recall = hitCount/size;
			//recall = hitCount;

			if((precision+recall) > 0)
			{
				f1=2*precision*recall/(precision+recall);
			}
			accuracyVector[0] += precision;
			accuracyVector[1] += recall;
			accuracyVector[2] += f1;
			
		}
	}
	if (counter>0)
	{
		//cout<<tempRecall<<"	"<<counter<<"	"<<accuracyVector[0]<<"	"<<accuracyVector[1]<<"	"<<accuracyVector[2]<<"	"<<endl;
		accuracyVector[0] = accuracyVector[0]/(counter);
		accuracyVector[1] = accuracyVector[1]/(counter);
		accuracyVector[2] = accuracyVector[2]/counter;
	}
	return accuracyVector;
}

double getSingleLamada(SimpleNetwork &testSet = combainTrainAndLearningSet)
{
	double maxRS = 0.0, tempRS = 0.0, bestLamada = 0.0;
	sngleLamadaArray.push_back(0.0);
	for (int i = 0; i<userNum; i++)
	{
		for (double lamada = 0; lamada<2;lamada+=0.1)
		{
			//UHHP_Train(i,lamada,testSet);
			tempRS = SingleUserRankingScore(i);
			if (tempRS>maxRS)
			{
				maxRS = tempRS;
				bestLamada = lamada;
			}
		}
		sngleLamadaArray[i] = bestLamada;
	}
	return bestLamada;
}

void startCalculateRS(int tsize,int i,SimpleNetwork &network)
{
	//cout<<tsize<<"	"<<i<<endl;
	//set collected item score == 0
	for (int n1 = 0; n1<tsize; n1++)
	{
		HybirditemScore[network.user_item_relation[i][n1]] = 0;
	}
	//very important------------------------------------------------
	vector<Ranker> itemPredictValue;
	Ranker p;
	for(long int iii=1; iii<itemNum; iii++)
	{
		if(HybirditemScore[iii] != 0)
		{
			p.id = iii;
			p.value = HybirditemScore[iii];
			itemPredictValue.push_back(p);
		}
	}
	//cout<<"after: "<<itemPredictValue.size()<<"	"<<i<<"	"<<tsize<<endl;
	std::sort(itemPredictValue.begin(), itemPredictValue.end(), Smaller);

	int predictSize = itemPredictValue.size();
	for (int topitem = 0; topitem<predictSize; topitem++)
	{
		if(topitem==TOPL)
		{
			break;
		}
		Hybirdscore[i][topitem] = itemPredictValue[topitem].id;
	}
	if(predictSize < TOPL)
	{
		//这里随机给不在推荐列表里面，而且不在本来已经选过的里面
		int seed;
		double localSeed = (double)time(NULL)+globalSeed;
		srand(localSeed);
		while(1)
		{
			bool hasThisSeed = false;
			seed = 1 + rand() % itemNum;
			//cout<<"seed	"<<seed<<endl;
			for(int index = 0;index<predictSize;index++)
			{
				//cout<<"1 "<<index<<endl;
				if(Hybirdscore[i][index] == seed)
				{
					hasThisSeed = true;
					break;
				}
			}
			if (hasThisSeed)
			{
				continue;
			}
			for(int index = 0;index<network.user_item_relation[i].size();index++)
			{
				if(network.user_item_relation[i][index] == seed)
				{
					hasThisSeed = true;
					break;
				}
			}
			if (hasThisSeed)
			{
				continue;
			}
			Hybirdscore[i][predictSize] = seed;
			//cout<<"predictSize: "<<predictSize<<"	"<<i<<"	"<<seed<<endl;
			predictSize++;
			if(predictSize >= TOPL)
			{
				break;
			}
		}
		globalSeed++;
	}
	vector<int> rankForUser(itemNum,0);
	//cout<<"predictSize  "<<predictSize<<"   "<<rankForUser.size()<<endl;
	for (int itemIndex = 0; itemIndex<itemPredictValue.size(); itemIndex++)
	{
		//cout<<i<<"  "<<itemIndex<<"  "<<itemPredictValue[itemIndex].id<<endl;
		rankForUser[itemPredictValue[itemIndex].id]=itemIndex+1;//让排名从1开始
		//cout<<"after"<<itemIndex<<"  "<<itemPredictValue[itemIndex].id<<endl;
		//rankForUser[index_array[itemIndex]]=itemIndex;
	}
	//caculate RS
	//cout<<"before predictSize  "<<predictSize<<"   "<<itemPredictValue.size()<<"	"<<rankForUser.size()<<endl;
	RankingScoreNotCollect(i,rankForUser,predictSize);
	rankForUser.clear();
	//cout<<"after RankingScoreNotCollect: "<<endl;
}

bool checkRuning(){
	if (isOtherMathodRunning)
	{
		cout<<"other method is runing"<<endl;
		return true;
	}else{
		isOtherMathodRunning = true;
		return false;
	}
}

//IFFSpare
double ProbS(int steps = 1, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is "<<__FUNCTION__<<" steps: "<<steps<<endl;	
	if (checkRuning())
	{
		return -1;
	}
	funcName = __FUNCTION__;
	int tempItem=0,size = 0;
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	////emptyVector(&Hybirdscore);
	//	emptyVector(&itemRankForUser);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//     HybirditemScore.clear();
		//vector<double> HybirditemScore(itemNum,0);
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (long int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = network.user_item_relation[m].size();
						//cout<<m<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);
					}
				}
			}
			//多部扩散
			for(int s = 1; s<steps; s++)
			{
				// 相应的vector要清零
				//cout<<i<<"  "<<s<<" steps "<<steps<<endl;
				for(long int u=0; u<userNum; u++)
				{
					tempUserScore[u] = 0;
				}
				for(long int item=0; item<itemNum; item++)
				{

					if(HybirditemScore[item]>0)
					{
						for (unsigned int k = 0; k < network.item_user_relation[item].size(); k++)
						{
							int linkBackUser = network.item_user_relation[item][k];
							tempUserScore[linkBackUser] += (HybirditemScore[item]/network.item_user_relation[item].size());
						}
					}
				}
				// 相应的vector要清零
				for(long int item=0; item<itemNum; item++)
				{
					HybirditemScore[item] = 0;
				}
				//user to item
				for (long int m = 0; m<userNum; m++)
				{
					if (tempUserScore[m]>0)
					{
						for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
						{
							int linkBackItem = network.user_item_relation[m][l];
							hybirdDegree = network.user_item_relation[m].size();
							//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
							HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);
						}
					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

int ProbS(SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is ProbS "<<endl;
	if (checkRuning())
	{
		return -1;
	}
	funcName = __FUNCTION__;
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	emptyVector(&itemRankForUser);
	////itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = network.user_item_relation[m].size();
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);

					}
				}
			}

			//set collected item score == 0
			if (tsize>0)
			{
				for (int n1 = 0; n1<tsize; n1++)
				{
					HybirditemScore[network.user_item_relation[i][n1]] = 0;
				}
			}

			std::copy(std::begin(HybirditemScore), std::end(HybirditemScore), std::begin(unSortedScore));
			int index_array[itemNum];
			for (int j1 = 0; j1 < itemNum; j1++)
			{
				index_array[j1] = j1;
			}
			qsort(index_array, itemNum, sizeof(index_array[0]), cmp_by_index);

			//sort(HybirditemScore,HybirditemScore+itemNum,greater<double>());
			//sort(begin,end,less<data-type>());
			for (int topitem = 0; topitem<TOPL;topitem++)
			{
				Hybirdscore[i][topitem] = index_array[topitem];
			}
			for (int itemIndex = 0; itemIndex<itemNum; itemIndex++)
			{
				//cout<<itemRankForUser[i].size()<<"	"<<itemIndex<<endl;
				itemRankForUser[i][index_array[itemIndex]]=itemIndex;
				//Hybirdscore[i][itemIndex] = HybirditemScore[index_array[itemIndex]];
			}
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

int ProbSWithSelfAvoiding(SimpleNetwork &network = combainTrainAndLearningSet)
{
	funcName = __FUNCTION__;
	cout<<"this is "<<__FUNCTION__<<endl;
	if (checkRuning())
	{
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	vector<int>::iterator iter;
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					if (linkBackUser == i)//self avoiding
					{
						continue;
					}
					hybirdDegree = network.item_user_relation[tempItemId].size();
					if (hybirdDegree==1)//只有一个user选择这个商品
					{
						continue;
					}else{
						hybirdDegree--;//去的自身
					}
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item,here can not back to the scource
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					SimpleHashSet<int> SimpleHashSet((network.user_item_relation[i].size()+network.user_item_relation[m].size())*5);
					int intersection = SimpleHashSet.getInterSectionSize(&SimpleHashSet,network.user_item_relation[i],network.user_item_relation[i].size(),network.user_item_relation[m],network.user_item_relation[m].size());
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];

						iter = find(network.user_item_relation[i].begin(), network.user_item_relation[i].end(), linkBackItem);
						if(iter!=network.user_item_relation[i].end())
						{
							continue;
						}
						hybirdDegree = network.user_item_relation[m].size()-intersection;
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);

					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

int HeatS(SimpleNetwork &network = combainTrainAndLearningSet)
{
	funcName = __FUNCTION__;
	cout<<"this is HeatS "<<endl;
	if (checkRuning())
	{
		isOtherMathodRunning = false;
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.user_item_relation[linkBackUser].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = network.item_user_relation[linkBackItem].size();
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);

					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;	
	return 0;
}

/************************************************************************/
/*  lamada = 1 probs
lamada = 0 heats
/************************************************************************/

int hybirdHAndPNonLinaer(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	funcName = __FUNCTION__;
	cout<<"this is hybirdHAndPNonLinaer and lamada is :	"<<lamada<<endl;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = pow((double)network.item_user_relation[tempItemId].size(),lamada)*pow((double)network.user_item_relation[linkBackUser].size(),(1-lamada));
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = pow((double)network.user_item_relation[m].size(),lamada)*pow((double)network.item_user_relation[linkBackItem].size(),(1-lamada));
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);
					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	//cout<<oldNet.getUserMaxDegree()<<"	old	"<<oldNet.getItemMaxDegree()<<endl;
	isOtherMathodRunning = false;
	return 0;
}

int Heter_hybrid(double sita,double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	funcName = __FUNCTION__;
	cout<<"this is hybirdHAndPNonLinaer and lamada is :	"<<lamada<<endl;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	int tempItemId = 0;
	double initialScore = 0.0;
	double useritemsize = 0.0, itemusersize = 0.0;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				tempItemId = network.user_item_relation[i][j];
				itemusersize = network.item_user_relation[tempItemId].size();
				for (int k = 0; k < itemusersize; k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = pow(itemusersize,lamada)*pow((double)network.user_item_relation[linkBackUser].size(),(1-lamada));
					initialScore = pow(itemusersize,sita);
					tempUserScore[linkBackUser] += (initialScore/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = pow((double)network.user_item_relation[m].size(),lamada)*pow((double)network.item_user_relation[linkBackItem].size(),(1-lamada));
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);

					}
				}
			}

			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}
//Jian-Guo Liu, Tao Zhou, Qiang Guo, Information filtering via biased heat conduction, Physical Review E 84, 037101 (2011).
int Biased_Heat(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	funcName = __FUNCTION__;
	cout<<"this is Biased_Heat and lamada is :	"<<lamada<<endl;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.user_item_relation[linkBackUser].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = pow((double)network.item_user_relation[linkBackItem].size(),lamada);
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);

					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//initialScore = pow(itemusersize,lamada);
int Heter_NBI(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	funcName = __FUNCTION__;
	cout<<"this is Heter_NBI and lamada is :	"<<lamada<<endl;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	int tempItemId = 0;
	double itemusersize = 0.0;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	double initialScore = 0.0;
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				tempItemId = network.user_item_relation[i][j];
				itemusersize = network.item_user_relation[tempItemId].size();
				for (int k = 0; k < itemusersize; k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					initialScore = pow(itemusersize,lamada);
					tempUserScore[linkBackUser] += (initialScore/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = network.user_item_relation[m].size();
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);

					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//hybirdDegree = pow(itemusersize,lamada)/allItemDegree;
int PD(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	funcName = __FUNCTION__;
	cout<<"this is PD and lamada is :	"<<lamada<<endl;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	int tempItemId = 0;
	double useritemsize = 0.0, itemusersize = 0.0;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					useritemsize = network.user_item_relation[m].size();
					double allItemDegree = 0.0;
					for (int l = 0; l < useritemsize; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						itemusersize = network.item_user_relation[linkBackItem].size();
						allItemDegree += pow(itemusersize,lamada);
					}
					for (int l = 0; l < useritemsize ; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						itemusersize = network.item_user_relation[linkBackItem].size();
						hybirdDegree = pow(itemusersize,lamada)/allItemDegree;
						HybirditemScore[linkBackItem] += (tempUserScore[m]*hybirdDegree);

					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

int RE_PD(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	funcName = __FUNCTION__;
	cout<<"this is "<<__FUNCTION__<<" and lamada is :	"<<lamada<<endl;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double firstTempItemScore[itemNum];
	double secondTempItemScore[itemNum];
	double hybirdDegree;
	int tempItemId = 0,useritemsize = 0;
	double itemusersize = 0.0;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(firstTempItemScore,0,sizeof(firstTempItemScore));
		memset(secondTempItemScore,0,sizeof(secondTempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				tempItemId = network.user_item_relation[i][j];
				itemusersize = network.item_user_relation[tempItemId].size();
				for (int k = 0; k < itemusersize; k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					useritemsize = network.user_item_relation[m].size();
					double allItemDegree = 0.0;
					for (int l = 0; l < useritemsize; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						itemusersize = network.item_user_relation[linkBackItem].size();
						allItemDegree += pow(itemusersize,lamada);
					}
					for (int l = 0; l < useritemsize ; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						itemusersize = network.item_user_relation[linkBackItem].size();
						hybirdDegree = pow(itemusersize,lamada)/allItemDegree;
						firstTempItemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);
					}
				}
			}
			//re nbi, do defussion again, item to user
			memset(tempUserScore,0,sizeof(tempUserScore));
			for (int i2 = 0; i2< itemNum; i2++)
			{
				if (firstTempItemScore[i2]>0)
				{
					tempItemId = i2;
					itemusersize = network.item_user_relation[tempItemId].size();
					for (int k = 0; k < itemusersize; k++)
					{
						int linkBackUser = network.item_user_relation[tempItemId][k];
						hybirdDegree = network.item_user_relation[tempItemId].size();
						tempUserScore[linkBackUser] += (firstTempItemScore[tempItemId]/hybirdDegree);
					}
				}
			}
			//user to item
			for (int m1 = 0; m1<userNum; m1++)
			{
				if (tempUserScore[m1]>0)
				{
					useritemsize = network.user_item_relation[m1].size();
					double allItemDegree = 0.0;
					for (int l = 0; l < useritemsize; l++)
					{
						int linkBackItem = network.user_item_relation[m1][l];
						itemusersize = network.item_user_relation[linkBackItem].size();
						allItemDegree += pow(itemusersize,lamada);
					}
					for (int l = 0; l < useritemsize ; l++)
					{
						int linkBackItem = network.user_item_relation[m1][l];
						itemusersize = network.item_user_relation[linkBackItem].size();
						hybirdDegree = pow(itemusersize,lamada)/allItemDegree;
						HybirditemScore[linkBackItem] += (tempUserScore[m1]*hybirdDegree);
					}
				}
			}

			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}
//lamada=-0.74
//re nbi, do defussion again, item to user
int RE_NBI(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is RE_NBI and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double firstTempItemScore[itemNum];
	double secondTempItemScore[itemNum];
	double hybirdDegree;
	int tempItemId = 0;
	double itemusersize = 0.0;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(firstTempItemScore,0,sizeof(firstTempItemScore));
		memset(secondTempItemScore,0,sizeof(secondTempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				tempItemId = network.user_item_relation[i][j];
				itemusersize = network.item_user_relation[tempItemId].size();
				for (int k = 0; k < itemusersize; k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = network.user_item_relation[m].size();
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						firstTempItemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);

					}
				}
			}
			//re nbi, do defussion again, item to user
			memset(tempUserScore,0,sizeof(tempUserScore));
			for (int i2 = 0; i2< itemNum; i2++)
			{
				if (firstTempItemScore[i2]>0)
				{
					tempItemId = i2;
					itemusersize = network.item_user_relation[tempItemId].size();
					for (int k = 0; k < itemusersize; k++)
					{
						int linkBackUser = network.item_user_relation[tempItemId][k];
						hybirdDegree = network.item_user_relation[tempItemId].size();
						tempUserScore[linkBackUser] += (firstTempItemScore[tempItemId]/hybirdDegree);
					}
				}
			}

			//user to item
			for (int m1 = 0; m1<userNum; m1++)
			{
				if (tempUserScore[m1]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m1].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m1][l];
						hybirdDegree = network.user_item_relation[m1].size();
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						secondTempItemScore[linkBackItem] += (tempUserScore[m1]/hybirdDegree);

					}
				}
			}

			//combian two times defussion

			for (int i1 = 0; i1< itemNum; i1++)
			{
				HybirditemScore[i1] = firstTempItemScore[i1]+lamada*secondTempItemScore[i1];
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//give some initial score
int Heter_PD(double lamada, double sita, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is Heter_PD and lamada is: "<<lamada<<" and sita is:	"<<sita<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double powerItemDegree[itemNum];
	double hybirdDegree;
	int tempItemId = 0;
	double initialScore = 0.0;
	double useritemsize = 0.0, itemusersize = 0.0;
	//emptyVector(&Hybirdscore);
	memset(powerItemDegree,0,sizeof(powerItemDegree));

	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				tempItemId = network.user_item_relation[i][j];
				itemusersize = network.item_user_relation[tempItemId].size();
				if (powerItemDegree[tempItemId] == 0)
				{
					powerItemDegree[tempItemId] = pow(itemusersize,lamada);
				}
				initialScore = powerItemDegree[tempItemId];
				
				for (int k = 0; k < itemusersize; k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					//hybirdDegree = network.item_user_relation[tempItemId].size();
					tempUserScore[linkBackUser] += (initialScore/itemusersize);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					useritemsize = network.user_item_relation[m].size();
					double allItemDegree = 0.0;
					for (int l = 0; l < useritemsize; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];

						if (powerItemDegree[linkBackItem] == 0)
						{
							itemusersize = network.item_user_relation[linkBackItem].size();
							powerItemDegree[linkBackItem] = pow(itemusersize,sita);
						}
						allItemDegree += powerItemDegree[linkBackItem];
					}

					for (int l = 0; l < useritemsize ; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];

						if (powerItemDegree[linkBackItem] == 0)
						{
							itemusersize = network.item_user_relation[linkBackItem].size();
							powerItemDegree[linkBackItem] = pow(itemusersize,sita);
						}						

						hybirdDegree = powerItemDegree[linkBackItem]/allItemDegree;
						HybirditemScore[linkBackItem] += (tempUserScore[m]*hybirdDegree);
					}
				}
			}

			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}
/************************************************************************/
/* An Zeng, Alexandre Vidmer, Matus Medo, Yi-Cheng Zhang, Information filtering via hybridization of similarity preferential diffusion processes, arXiv:1309.0129 (2013)
hybird +pd*/
/************************************************************************/
int SPD(double lamada, double sita,SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is SPD and lamada is :	"<<lamada<<" sita is: "<<sita<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double powerUserDegree[userNum];
	double powerItemDegree[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	memset(powerUserDegree,0,sizeof(powerUserDegree));
	memset(powerItemDegree,0,sizeof(powerItemDegree));

	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					if (powerUserDegree[linkBackUser] == 0)
					{
						powerUserDegree[linkBackUser] = pow((double)network.user_item_relation[linkBackUser].size(),(1-lamada));
					}
					if (powerItemDegree[tempItemId] == 0)
					{
						powerItemDegree[tempItemId] = pow((double)network.item_user_relation[tempItemId].size(),lamada);
					}					
					hybirdDegree = powerItemDegree[tempItemId]*powerUserDegree[linkBackUser];
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			memset(powerUserDegree,0,sizeof(powerUserDegree));
			memset(powerItemDegree,0,sizeof(powerItemDegree));
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];

						if (powerUserDegree[m] == 0)
						{
							powerUserDegree[m] = pow((double)network.user_item_relation[m].size(),lamada);
						}
						if (powerItemDegree[linkBackItem] == 0)
						{
							powerItemDegree[linkBackItem] = pow((double)network.item_user_relation[linkBackItem].size(),(1-lamada));
						}	

						hybirdDegree = powerUserDegree[m]*powerItemDegree[linkBackItem];
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (pow(tempUserScore[m],(double)sita)/hybirdDegree);
					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

/************************************************************************/
/* Yuan Guan, Dandan Zhao, An Zeng, Ming-Sheng Shang, Preference of online users and personalized recommendations, Physica A 392, 3417 (2013).                                                                     */
/************************************************************************/
int UHHP_Train(int i, double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is UHHP_Train and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();

	memset(tempUserScore,0,sizeof(tempUserScore));
	memset(tempItemScore,0,sizeof(tempItemScore));
	memset(unSortedScore,0,sizeof(unSortedScore));
	memset(HybirditemScore,0,sizeof(HybirditemScore));
	//item to user
	int tsize = network.user_item_relation[i].size();
	if (tsize>0)
	{
		for (int j = 0; j < tsize; j++)
		{
			int tempItemId = network.user_item_relation[i][j];
			for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
			{
				int linkBackUser = network.item_user_relation[tempItemId][k];
				hybirdDegree = pow((double)network.item_user_relation[tempItemId].size(),lamada)*pow((double)network.user_item_relation[linkBackUser].size(),(1-lamada));
				tempUserScore[linkBackUser] += (1.0/hybirdDegree);
			}
		}
		//user to item
		for (int m = 0; m<userNum; m++)
		{
			if (tempUserScore[m]>0)
			{
				for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
				{
					int linkBackItem = network.user_item_relation[m][l];
					hybirdDegree = pow((double)network.user_item_relation[m].size(),lamada)*pow((double)network.item_user_relation[linkBackItem].size(),(1-lamada));
					//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
					HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);

				}
			}
		}

		startCalculateRS(tsize,i,network);
	}
	isOtherMathodRunning = false;
	return 0;
}

int UHHP_Test(SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is UHHP_Test and lamada is :	"<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	double lamada;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		lamada = sngleLamadaArray[i];
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = pow((double)network.item_user_relation[tempItemId].size(),lamada)*pow((double)network.user_item_relation[linkBackUser].size(),(1-lamada));
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = pow((double)network.user_item_relation[m].size(),lamada)*pow((double)network.item_user_relation[linkBackItem].size(),(1-lamada));
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);

					}
				}
			}

			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

int addGrandNode(SimpleNetwork &network = combainTrainAndLearningSet)
{
	int counter = 0;
	for (int allitem = 0; allitem<itemNum;allitem++)
	{
		counter++;
		network.addEdge(userNum-1,allitem);
		/*network.user_item_relation[userNum-1].push_back(allitem);
		network.item_user_relation[allitem].push_back(userNum-1);*/
	}
	return counter;
}

int GrandNodeHybird(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"please Run addGrandNode()first \n this is GrandNodeHybird and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum-1; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = pow((double)network.item_user_relation[tempItemId].size(),lamada)*pow((double)network.user_item_relation[linkBackUser].size(),(1-lamada));
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = pow((double)network.user_item_relation[m].size(),lamada)*pow((double)network.item_user_relation[linkBackItem].size(),(1-lamada));
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);

					}
				}
			}

			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

int GrandNodeHybirdMatrix(double lamada, SimpleNetwork &network = combainTrainAndLearningSet){
	cout<<"this is MatrixHybird and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	int allcommonUsers, sizei,sizej;

	//transformationMatrix = Matrix<double,Dynamic,Dynamic>::Zero();
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			if (allcommonUsers == 0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					temp += 1.0/(network.user_item_relation[itemsCommonNeighbor[commonuser]].size());
					/*cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;*/
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			//transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0));
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0))));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/(pow((sizej*1.0),(1-lamada*1.0))*pow((sizei*1.0),lamada*1.0))));
			//transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0));//lamada = 1-lamada
			/*cout<<itemAlpha<<"	"<<itemBeta<<"	"<<transformationMatrix(itemAlpha,itemBeta)<<"	"<<temp<<"	"<<pow((network.item_user_relation[itemBeta].size()*1.0),lamada)<<" "<<allcommonUsers<<endl;*/

		}
		//cout<<itemAlpha<<" transformationMatrix  "<<transformationMatrix(itemAlpha,itemAlpha)<<endl;
	}
	//set collected item score == 0
	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	////emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);;
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);;
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}		
	}
	isOtherMathodRunning = false;
	return 0;
}

//-Yanbo Zhou,Linyuan Lu,Weiping Liu,Jianlin Zhang, The Power of Ground User in Recommender Systems, PLoS One8(8), e70094 (2013). Ground User
int WeightiedGrandNodeHC(double weight, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"please Run addGrandNode()first \n this is WeightiedGrandNodeHC and lamada is :	"<<weight<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	int allcommonUsers, sizei,sizej;

	//transformationMatrix = Matrix<double,Dynamic,Dynamic>::Zero();
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			if (allcommonUsers==0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				double initialScore = 1.0;
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					if (itemsCommonNeighbor[commonuser] == userNum-1)
					{
						initialScore = weight;
					}
					temp += initialScore/network.user_item_relation[itemsCommonNeighbor[commonuser]].size();
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			//transformationMatrix(itemAlpha,itemBeta) = temp/sizei;
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/sizei));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/sizej));
		}
	}
	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);

	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);;
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);;
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}		
	}
	isOtherMathodRunning = false;
	return 0;
}

int WeightiedGrandNodeHybird(double lamada,double weight, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"please Run addGrandNode()first \n this is WeightiedGrandNodeHybird and lamada is :	"<<weight<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	int allcommonUsers, sizei,sizej;

	//transformationMatrix = Matrix<double,Dynamic,Dynamic>::Zero();
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			if (allcommonUsers==0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;

			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				double initialScore = 1.0;
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					if (itemsCommonNeighbor[commonuser] == userNum-1)
					{
						initialScore = weight;
					}
					temp += initialScore/(network.user_item_relation[itemsCommonNeighbor[commonuser]].size());
					/*cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;*/
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0))));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/(pow((sizej*1.0),(1-lamada*1.0))*pow((sizei*1.0),lamada*1.0))));
			//transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0));
		}
	}
	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);;
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);;
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}		
	}
	isOtherMathodRunning = false;
	return 0;
}
//-Qian-Ming Zhang,An Zeng,Ming-Sheng Shang, Extracting the Information Backbone in Online System, PLoS One8(5), e62624 (2013).

int ExtractingBackbone(double lamada,double percent, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is ExtractingBackbone and lamada is: "<<lamada<<"  percent is:	"<<percent<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	
	//itemRankForUser.clear();
	int degrees = network.countDegree();
	int tempdegre = 0,tempu=0,tempi=0,tempItem,tempmulitidegree,tpi;
	tempdegre = 0;
	while (network.countDegree()>degrees*percent)
	{
		for (int u = 0; u<userNum; u++)
		{
			for (unsigned int itemindex = 0; itemindex < network.user_item_relation[u].size(); itemindex++)
			{
				tempItem = network.user_item_relation[u][itemindex];
				tempmulitidegree = network.user_item_relation[u].size()*network.item_user_relation[tempItem].size();
				if (tempdegre < tempmulitidegree)
				{
					tempdegre = tempmulitidegree;
					tempu=u;
					tempi=tempItem;
				}
				tpi = itemindex;
			}
		}
		//cout<<"tempdegre: "<<tempdegre<<endl;
		//cout<<tempu<<" "<<tpi<<" "<<tempItem<<"  "<<tempi<<" "<<tempmulitidegree<<" "<<network.user_item_relation[tempu][tpi]<<endl;
		network.removeEdge(tempu,tempi);
	}
	cout<<degrees<<" now we have drgree: "<<network.countDegree()<<"tempdegre: "<<tempdegre<<endl;
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = pow((double)network.item_user_relation[tempItemId].size(),lamada)*pow((double)network.user_item_relation[linkBackUser].size(),(1-lamada));
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = pow((double)network.user_item_relation[m].size(),lamada)*pow((double)network.item_user_relation[linkBackItem].size(),(1-lamada));
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);
					}
				}
			}

			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//-Wei Zeng, An Zeng, Ming-Sheng Shang, Yi-Cheng Zhang, Information filtering in sparse online systems: recommendation via semi-local diffusion, arXiv:1308.3060 (2013).
//多次扩散
int IFFSparseMatrix(int times, SimpleNetwork &network = combainTrainAndLearningSet){
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	if (times == 0)
	{
		times++;
	}
	cout<<"this is IFonSparse and lamada is :	"<<times<<endl;
	int allcommonUsers, sizei,sizej;

	//transformationMatrix = Matrix<double,Dynamic,Dynamic>::Zero();
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			/*if (itemAlpha==itemBeta)
			{
				continue;
			}*/
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			if (allcommonUsers==0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					temp += 1.0/(network.user_item_relation[itemsCommonNeighbor[commonuser]].size());
					/*cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;*/
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			//transformationMatrix(itemAlpha,itemBeta) = temp/sizej;
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/sizej));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/sizei));
			/*cout<<itemAlpha<<"	"<<itemBeta<<"	"<<transformationMatrix(itemAlpha,itemBeta)<<"	"<<temp<<"	"<<pow((network.item_user_relation[itemBeta].size()*1.0),lamada)<<" "<<allcommonUsers<<endl;*/

		}
	}
	//set collected item score == 0
	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));

		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);;
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);;
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int i1 = 0; i1< times-1; i1++)
		{
			answerV = mat*answerV;
		}

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

/************************************************************************/
/*  lamada = 1 probs
lamada = 0 heats
/************************************************************************/
int MatrixHybirdReal(double lamada, SimpleNetwork &network = combainTrainAndLearningSet){
	cout<<"this is MatrixHybirdReal and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	int allcommonUsers, sizei,sizej;

	//transformationMatrix = Matrix<double,Dynamic,Dynamic>::Zero();
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			if (allcommonUsers==0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					temp += 1.0/(network.user_item_relation[itemsCommonNeighbor[commonuser]].size());
					/*cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;*/
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			//transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizej*1.0),(1-lamada*1.0))*pow((sizei*1.0),lamada*1.0));
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/(pow((sizej*1.0),(1-lamada*1.0))*pow((sizei*1.0),lamada*1.0))));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0))));
			//transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0));//lamada = 1-lamada
			/*cout<<itemAlpha<<"	"<<itemBeta<<"	"<<transformationMatrix(itemAlpha,itemBeta)<<"	"<<temp<<"	"<<pow((network.item_user_relation[itemBeta].size()*1.0),lamada)<<" "<<allcommonUsers<<endl;*/

		}
		//cout<<itemAlpha<<" transformationMatrix  "<<transformationMatrix(itemAlpha,itemAlpha)<<endl;
	}
	//set collected item score == 0
	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,1,Dynamic> tempv;
		tempv= Matrix<double,1,Dynamic>::Zero(Dynamic);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(0,network.user_item_relation[i][j]) =1.0;
		}

		Matrix<double,1,Dynamic> answerV;
		answerV= Matrix<double,1,Dynamic>::Zero(Dynamic);
		answerV = tempv*mat;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(0)[k];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}		
	}
	isOtherMathodRunning = false;
	return 0;
}

int MatrixHybird(double lamada, SimpleNetwork &network = combainTrainAndLearningSet){
	cout<<"this is MatrixHybird and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	int allcommonUsers, sizei,sizej;

	//transformationMatrix = Matrix<double,Dynamic,Dynamic>::Zero();
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			if (allcommonUsers==0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					temp += 1.0/(network.user_item_relation[itemsCommonNeighbor[commonuser]].size());
					/*cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;*/
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			//transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0));
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0))));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/(pow((sizej*1.0),(1-lamada*1.0))*pow((sizei*1.0),lamada*1.0))));
			//transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0));//lamada = 1-lamada
			/*cout<<itemAlpha<<"	"<<itemBeta<<"	"<<transformationMatrix(itemAlpha,itemBeta)<<"	"<<temp<<"	"<<pow((network.item_user_relation[itemBeta].size()*1.0),lamada)<<" "<<allcommonUsers<<endl;*/

		}
		//cout<<itemAlpha<<" transformationMatrix  "<<transformationMatrix(itemAlpha,itemAlpha)<<endl;
	}
	//set collected item score == 0

	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);;
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);;
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//Marcel Blattner, B-Rank: A top N Recommendation Algorithm, arXiv:0908.2741 (2009).
int B_Rank(SimpleNetwork &network = combainTrainAndLearningSet){
	cout<<"this is B_Rank "<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	int allcommonUsers, sizei,sizej;
	//transformationMatrix = Matrix< double , Dynamic , Dynamic >::Zero(itemNum,itemNum);
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			if (itemAlpha==itemBeta)
			{
				continue;
			}
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			if (allcommonUsers==0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					temp += 1.0/(network.user_item_relation[itemsCommonNeighbor[commonuser]].size());
					/*cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;*/
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			//transformationMatrix(itemAlpha,itemBeta) = temp/sizej;
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/sizej));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/sizei));
		}
		//cout<<itemAlpha<<" transformationMatrix  "<<transformationMatrix(itemAlpha,itemAlpha)<<endl;
	}
	//set collected item score == 0

	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);;
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);;
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//Tian Qiu, Guang Chen, Zi-Ke Zhang and Tao Zhou, An item-oriented recommendation algorithm on cold-start problem, EPL 95, 58003 (2011).
int Cold_StartMatrix(double gama, SimpleNetwork &network = combainTrainAndLearningSet){
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	double lamada=0.0;
	int allcommonUsers, sizei,sizej;
	int itemMaxDegree = network.getItemMaxDegree();	
	double powItemDegree[itemNum];
	memset(powItemDegree,0,sizeof(powItemDegree));
	for (int i = 0; i< itemNum; i++)
	{		
		powItemDegree[i] = pow(network.item_user_relation[i].size()*1.0/itemMaxDegree,lamada*1.0);
	}

	//transformationMatrix = Matrix< double , Dynamic , Dynamic >::Zero(itemNum,itemNum);
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			if (itemAlpha==itemBeta)
			{
				continue;
			}
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			if (allcommonUsers==0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					temp += 1.0/(network.user_item_relation[itemsCommonNeighbor[commonuser]].size());
					/*cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;*/
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			//lamada = pow(sizej*1.0/itemMaxDegree,gama*1.0);
			//cout<<"this is Cold_Start and gama is :	"<<gama<<" lamada "<<lamada<<endl;
			//transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizei*1.0),(1-powItemDegree[itemBeta]))*pow((sizej*1.0),powItemDegree[itemBeta]));
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/(pow((sizei*1.0),(1-powItemDegree[itemBeta]))*pow((sizej*1.0),powItemDegree[itemBeta]))));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/(pow((sizej*1.0),(1-powItemDegree[itemAlpha]))*pow((sizei*1.0),powItemDegree[itemAlpha]))));
		}
	}
	//set collected item score == 0
	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);;
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);;
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//Tian Qiu, Guang Chen, Zi-Ke Zhang and Tao Zhou, An item-oriented recommendation algorithm on cold-start problem, EPL 95, 58003 (2011).
int Cold_Start_notwork(double lamada, SimpleNetwork &network = combainTrainAndLearningSet){
	funcName = __FUNCTION__;
	cout<<"this is Cold_Start and lamada is :	"<<lamada<<endl;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	int itemMaxDegree = network.getItemMaxDegree();
	double powItemDegree[itemNum];
	memset(powItemDegree,0,sizeof(powItemDegree));
	for (int i = 0; i< itemNum; i++)
	{		
		powItemDegree[i] = pow(network.item_user_relation[i].size()*1.0/itemMaxDegree,lamada*1.0);
	}

	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));

		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = pow((double)network.item_user_relation[tempItemId].size(),powItemDegree[tempItemId])*pow((double)network.user_item_relation[linkBackUser].size(),(1-powItemDegree[tempItemId]));
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = pow((double)network.user_item_relation[m].size(),powItemDegree[linkBackItem])*pow((double)network.item_user_relation[linkBackItem].size(),(1-powItemDegree[linkBackItem]));
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/hybirdDegree);
					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	//cout<<oldNet.getUserMaxDegree()<<"	old	"<<oldNet.getItemMaxDegree()<<endl;
	isOtherMathodRunning = false;
	return 0;
}

//Run-Ran Liu,Jian-Guo Liu,  Chun-Xiao Jia, Bing-Hong Wang, Personal recommendation via unequal resource allocation on bipartite networks, Physica A 389, 3282 (2010).
int URA_NBI(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is URA_NBI and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	int tempItemId = 0;
	double useritemsize = 0.0, itemusersize = 0.0;
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				tempItemId =network.user_item_relation[i][j];
				itemusersize = network.item_user_relation[tempItemId].size();
				double allUsrDegree = 0.0;
				for (int l = 0; l < itemusersize; l++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][l];
					useritemsize = network.item_user_relation[linkBackUser].size();
					allUsrDegree += pow(useritemsize,lamada);
				}
				for (int l = 0; l < itemusersize ; l++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][l];
					useritemsize = network.item_user_relation[linkBackUser].size();
					hybirdDegree = pow(itemusersize,lamada)/allUsrDegree;
					tempUserScore[linkBackUser] += (1.0*hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					useritemsize = network.user_item_relation[m].size();
					double allItemDegree = 0.0;
					for (int l = 0; l < useritemsize; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						itemusersize = network.item_user_relation[linkBackItem].size();
						allItemDegree += pow(itemusersize,lamada);
					}
					for (int l = 0; l < useritemsize ; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						itemusersize = network.item_user_relation[linkBackItem].size();
						hybirdDegree = pow(itemusersize,lamada)/allItemDegree;
						HybirditemScore[linkBackItem] += (tempUserScore[m]*hybirdDegree);

					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//Jian-Guo Liu, Qiang Guo, Yi-Cheng Zhang, Information filtering via weighted heat conduction algorithm, Physica A 390, 2414 (2011).
int WHCMatrix(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is WHCMatrix and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	int allcommonUsers, sizei,sizej;

	//transformationMatrix = Matrix<double,Dynamic,Dynamic>::Zero();
	//transformationMatrix = Matrix< double , Dynamic , Dynamic >::Zero(itemNum,itemNum);
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			//cout<<sizei<<"	"<<sizej<<endl;
			SimpleHashSet<int> simpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&simpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			simpleHashSet.SetLength(0);
			if (allcommonUsers==0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				double initialScore = 0.0;
				initialScore = sizei*sizej;
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					temp += pow(initialScore,lamada)*pow(network.user_item_relation[itemsCommonNeighbor[commonuser]].size(),2.0*lamada-1);
					//temp += 1.0/(network.user_item_relation[itemsCommonNeighbor[commonuser]].size());
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			//transformationMatrix(itemAlpha,itemBeta) = temp/sizei;
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/sizei));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/sizej));
		}
	}
	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);;
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);;
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//Jian-Guo Liu, Qiang Guo, Yi-Cheng Zhang, Information filtering via weighted heat conduction algorithm, Physica A 390, 2414 (2011).
////////////////////initialScore = sizei*sizej/////////////////////////////
int WHC(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	
	return 0;
}

//Jian-Guo Liu, Kerui Shi, and Qiang Guo, Solving the accuracy-diversity dilemma via directed random walks, Physical Review E 85, 016118 (2012).
int NCF(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is NCF and lamada is: "<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	//network.initUserItemMatrix();
	double oneToAllUserSimilarityVectorSum[userNum];
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	vector<vector <double> > MCFUserUserMatrix;//(userNum,vector<double>(userNum,0.0));
	emptyVector(&MCFUserUserMatrix);
	memset(oneToAllUserSimilarityVectorSum,0,sizeof(oneToAllUserSimilarityVectorSum));
	for (int i = 0; i<userNum; i++)
	{
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					MCFUserUserMatrix[i][linkBackUser]+= (1.0/hybirdDegree);

					/*int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);*/
				}
			}
		}
	}

	for (int i = 0; i<userNum; i++)
	{
		MCFUserUserMatrix[i][i]=0;
		for (int j = 0; j< userNum; j++)
		{
			oneToAllUserSimilarityVectorSum[i] +=  pow(MCFUserUserMatrix[i][j],lamada*1.0);
		}
	}

	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			memset(tempUserScore,0,sizeof(tempUserScore));
			memset(tempItemScore,0,sizeof(tempItemScore));
			memset(unSortedScore,0,sizeof(unSortedScore));
			memset(HybirditemScore,0,sizeof(HybirditemScore));
			double allSimilarity = oneToAllUserSimilarityVectorSum[i];
			for (int j = 0; j<itemNum; j++)
			{
				//allSimilarity += pow(MCFUserUserMatrix[i][j],lamada*1.0);
				/*for (int n1 = 0; n1<tsize; n1++)
				{
				HybirditemScore[network.user_item_relation[i][n1]] = 0;
				}*/
				/*if (network.oriTopoNetwork[i][j]==1)
				{
					HybirditemScore[j]=0;
				}
				else
				{*/
				int linkedBackUserSize = network.item_user_relation[j].size();
				if (linkedBackUserSize>0)
				{
					double revolvledUserSimilarity = 0.0;
					for (int k = 0; k<linkedBackUserSize; k++)
					{
						int linkBackUser = network.item_user_relation[j][k];
						revolvledUserSimilarity+=pow(MCFUserUserMatrix[i][linkBackUser],lamada*1.0);
					}
					HybirditemScore[j] = revolvledUserSimilarity/allSimilarity;
				}
				//}
			}
			startCalculateRS(tsize,i,network);
		}		
	}
	isOtherMathodRunning = false;
	return 0;
}

//Jian-Guo Liu, Kerui Shi, and Qiang Guo, Solving the accuracy-diversity dilemma via directed random walks, Physical Review E 85, 016118 (2012).
//for user itemslistcore[item]=(sum(item selected user similary))/(user collected item link back all user similary to sum)
int NCFNew(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is NCF and lamada is: "<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{
		return -1;
	}
	//vector<vector <double> > MCFUserUserMatrix(userNum,vector<double>(userNum,0.0));
	//network.initUserItemMatrix();
	double oneToAllUserSimilarityVectorSum;
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	double MCFUserUserMatrix[userNum];
	//emptyVector(&MCFUserUserMatrix);
	//memset(oneToAllUserSimilarityVectorSum,0,sizeof(oneToAllUserSimilarityVectorSum));
	
	for (int i = 0; i<userNum; i++)
	{
		memset(MCFUserUserMatrix,0,sizeof(MCFUserUserMatrix));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					MCFUserUserMatrix[linkBackUser] += (1.0/hybirdDegree);
				}
			}
		}

		MCFUserUserMatrix[i]=0;
		for (int j = 0; j< userNum; j++)
		{
			oneToAllUserSimilarityVectorSum +=  pow(MCFUserUserMatrix[j],lamada*1.0);
		}

		//emptyVector(&Hybirdscore);
		//itemRankForUser.clear();

		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		double allSimilarity = oneToAllUserSimilarityVectorSum;
		for (int j = 0; j<itemNum; j++)
		{				
			int linkedBackUserSize = network.item_user_relation[j].size();
			if (linkedBackUserSize>0)
			{
				double revolvledUserSimilarity = 0.0;
				for (int k = 0; k<linkedBackUserSize; k++)
				{
					int linkBackUser = network.item_user_relation[j][k];
					revolvledUserSimilarity+=pow(MCFUserUserMatrix[linkBackUser],lamada*1.0);
				}
				HybirditemScore[j] = revolvledUserSimilarity/allSimilarity;
			}

			//if (MCFUserUserMatrix[i]>0)
			//{
			//	for (unsigned int l = 0; l < network.user_item_relation[i].size(); l++)
			//	{
			//		int linkBackItem = network.user_item_relation[i][l];
			//		hybirdDegree = network.user_item_relation[i].size();
			//		//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
			//		HybirditemScore[linkBackItem] += (tempUserScore[i]/hybirdDegree);
			//	}
			//}
		}
		startCalculateRS(tsize,i,network);
	}	
	isOtherMathodRunning = false;
	return 0;
}

int MCF(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is NCF and lamada is: "<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	//network.initUserItemMatrix();
	double oneToAllUserSimilarityVectorSum[userNum];
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	vector<vector <double> > MCFUserUserMatrix;//(userNum,vector<double>(userNum,0.0));
	emptyVector(&MCFUserUserMatrix);
	memset(oneToAllUserSimilarityVectorSum,0,sizeof(oneToAllUserSimilarityVectorSum));
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					//tempUserScore[linkBackUser] += (1.0/hybirdDegree);
					MCFUserUserMatrix[i][linkBackUser]+= (1.0/hybirdDegree);
				}
			}
		}
	}

	for (int i = 0; i<userNum; i++)
	{
		MCFUserUserMatrix[i][i]=0;
		for (int j = 0; j< userNum; j++)
		{
			oneToAllUserSimilarityVectorSum[i] +=  pow(MCFUserUserMatrix[i][j],lamada*1.0);
		}
		for (int j = 0; j<i; j++)
		{
			if (MCFUserUserMatrix[i][j]<MCFUserUserMatrix[j][i])
			{
				MCFUserUserMatrix[i][j] = MCFUserUserMatrix[j][i];
			}else{
				MCFUserUserMatrix[j][i] = MCFUserUserMatrix[i][j];
			}
		}
	}
	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			memset(tempUserScore,0,sizeof(tempUserScore));
			memset(tempItemScore,0,sizeof(tempItemScore));
			memset(unSortedScore,0,sizeof(unSortedScore));
			memset(HybirditemScore,0,sizeof(HybirditemScore));
			double allSimilarity = oneToAllUserSimilarityVectorSum[i];
			for (int j = 0; j<itemNum; j++)
			{
				/*if (network.oriTopoNetwork[i][j]==1)
				{
					HybirditemScore[j]=0;
				}else
				{*/
				int linkedBackUserSize = network.item_user_relation[j].size();
				if (linkedBackUserSize>0)
				{
					double revolvledUserSimilarity = 0.0;
					for (int k = 0; k<linkedBackUserSize; k++)
					{
						int linkBackUser = network.item_user_relation[j][k];
						revolvledUserSimilarity+=pow(MCFUserUserMatrix[i][linkBackUser],lamada*1.0);
					}
					HybirditemScore[j] = revolvledUserSimilarity/allSimilarity;
				}
				//}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//-Q. Guo, R. Leng, K. Shi, J.G. Liu, Heat conduction information filtering via local information of bipartite networks, EPJB 85, 286 (2012)
int IHC_donnotknow(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is IHC and lamada is: "<<lamada<<endl;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;

	double powerItemDegree[itemNum];
	memset(powerItemDegree,0,sizeof(powerItemDegree));

	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.user_item_relation[linkBackUser].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = network.item_user_relation[linkBackItem].size();
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						if (powerItemDegree[linkBackItem] == 0)
						{
							powerItemDegree[linkBackItem] = pow(hybirdDegree*1.0,lamada);
						}
						HybirditemScore[linkBackItem] += (tempUserScore[m]/powerItemDegree[linkBackItem]);
						//HybirditemScore[linkBackItem] += (tempUserScore[m]/pow(hybirdDegree*1.0,lamada));
					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

int IMD(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is IMD and lamada is: "<<lamada<<endl;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;

	double powerUserDegree[userNum];
	memset(powerUserDegree,0,sizeof(powerUserDegree));

	//emptyVector(&Hybirdscore);
	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					hybirdDegree = network.item_user_relation[tempItemId].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (long int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						hybirdDegree = network.user_item_relation[m].size();

						if (powerUserDegree[m] == 0)
						{
							powerUserDegree[m] = pow(hybirdDegree*1.0,lamada);
						}
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/powerUserDegree[m]);
					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

int IHCMatrix(double lamada, SimpleNetwork &network = combainTrainAndLearningSet){
	cout<<"this is IHCMatrix and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	int allcommonUsers, sizei,sizej;
	//transformationMatrix = Matrix< double , Dynamic , Dynamic >::Zero(itemNum,itemNum);

	double powerUserDegree[userNum];
	memset(powerUserDegree,0,sizeof(powerUserDegree));

	//transformationMatrix = Matrix<double,Dynamic,Dynamic>::Zero();
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			if (allcommonUsers==0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					if (powerUserDegree[itemsCommonNeighbor[commonuser]] == 0)
					{
						powerUserDegree[itemsCommonNeighbor[commonuser]] = pow(network.user_item_relation[itemsCommonNeighbor[commonuser]].size()*1.0,lamada);
					}
					temp += 1.0/powerUserDegree[itemsCommonNeighbor[commonuser]];										
					//temp += 1.0/pow(network.user_item_relation[itemsCommonNeighbor[commonuser]].size()*1.0,lamada);
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			//transformationMatrix(itemAlpha,itemBeta) = temp/sizei;
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/sizei));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/sizej));

			/////transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0));//lamada = 1-lamada
			/*cout<<itemAlpha<<"	"<<itemBeta<<"	"<<transformationMatrix(itemAlpha,itemBeta)<<"	"<<temp<<"	"<<pow((network.item_user_relation[itemBeta].size()*1.0),lamada)<<" "<<allcommonUsers<<endl;*/

		}
		//cout<<itemAlpha<<" transformationMatrix  "<<transformationMatrix(itemAlpha,itemAlpha)<<endl;
	}
	//set collected item score == 0
	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//Tian Qiu, Tian-Tian Wang, Zi-Ke Zhang, Li-Xin Zhong, Guang Chen, Heterogeneity Involved Network-based Algorithm Leads toAccurate and Personalized Recommendations, arXiv:1305.7438 (2013).
int HHCMatrix(double lamada, SimpleNetwork &network = combainTrainAndLearningSet){
	cout<<"this is HHCMatrix and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{
		return -1;
	}

	int allcommonUsers, sizei,sizej;
	double powerItemDegree[itemNum];
	memset(powerItemDegree,0,sizeof(powerItemDegree));

	//transformationMatrix = Matrix<double,Dynamic,Dynamic>::Zero();
	//transformationMatrix = Matrix< double , Dynamic , Dynamic >::Zero(itemNum,itemNum);
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			if (allcommonUsers==0)
			{
				continue;
			}
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			for (int commonuser = 0;commonuser<allcommonUsers;commonuser++)
			{
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					temp += 1.0/network.user_item_relation[itemsCommonNeighbor[commonuser]].size();
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}

			if (powerItemDegree[itemAlpha] == 0)
			{
				powerItemDegree[itemAlpha] = pow(sizei*1.0,lamada);
			}
			//transformationMatrix(itemAlpha,itemBeta) = temp/powerItemDegree[itemAlpha];
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/powerItemDegree[itemAlpha]));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/powerItemDegree[itemBeta]));
			//transformationMatrix(itemAlpha,itemBeta) = temp/pow(sizei*1.0,lamada);
			//transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizei*1.0),(1-lamada*1.0))*pow((sizej*1.0),lamada*1.0));//lamada = 1-lamada
			/*cout<<itemAlpha<<"	"<<itemBeta<<"	"<<transformationMatrix(itemAlpha,itemBeta)<<"	"<<temp<<"	"<<pow((network.item_user_relation[itemBeta].size()*1.0),lamada)<<" "<<allcommonUsers<<endl;*/

		}
		//cout<<itemAlpha<<" transformationMatrix  "<<transformationMatrix(itemAlpha,itemAlpha)<<endl;
	}
	//set collected item score == 0

	SpMatrix mat(itemNum,itemNum);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	tripletList.clear();
	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);;
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);;
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//approved
int HHC(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	funcName = __FUNCTION__;
	cout<<"this is HHC and lamada is :	"<<lamada<<endl;
	if (checkRuning())
	{
		isOtherMathodRunning = false;
		return -1;
	}
	double tempUserScore[userNum];
	double tempItemScore[itemNum];
	double hybirdDegree;
	//emptyVector(&Hybirdscore);
	double powerItemDegree[itemNum];
	memset(powerItemDegree,0,sizeof(powerItemDegree));

	double powerUserDegree[userNum];
	memset(powerUserDegree,0,sizeof(powerUserDegree));

	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(tempItemScore,0,sizeof(tempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = network.user_item_relation[i][j];
				for (unsigned int k = 0; k < network.item_user_relation[tempItemId].size(); k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					if (powerUserDegree[linkBackUser] == 0)
					{
						powerUserDegree[linkBackUser] = pow(network.user_item_relation[linkBackUser].size()*1.0,lamada);
					}
					//hybirdDegree = network.user_item_relation[linkBackUser].size();
					tempUserScore[linkBackUser] += (1.0/powerUserDegree[linkBackUser]);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					for (unsigned int l = 0; l < network.user_item_relation[m].size(); l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						if (powerItemDegree[linkBackItem] == 0)
						{
							powerItemDegree[linkBackItem] = pow(network.item_user_relation[linkBackItem].size()*1.0,lamada);
						}
						//hybirdDegree = network.item_user_relation[linkBackItem].size();
						//cout<<linkBackUser<<" "<<l<<" "<<linkBackItem<<endl;
						HybirditemScore[linkBackItem] += (tempUserScore[m]/powerItemDegree[linkBackItem]);

					}
				}
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;	
	return 0;
}

//this is cold start(OHHP)
int caclSparseNetwork(double lamada, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"this is caclSparseNetwork and lamada is :	"<<lamada<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	//double lamada=0.0;
	int allcommonUsers, sizei,sizej;
	int itemMaxDegree = network.getItemMaxDegree();	
	double powItemDegree[itemNum];
	memset(powItemDegree,0,sizeof(powItemDegree));
	for (int i = 0; i< itemNum; i++)
	{		
		powItemDegree[i] = pow(network.item_user_relation[i].size()*1.0/itemMaxDegree,lamada*1.0);
	}

	//transformationMatrix = Matrix< double , Dynamic , Dynamic >::Zero(itemNum,itemNum);
	for (int itemAlpha=0; itemAlpha< itemNum; itemAlpha++)
	{
		sizei = network.item_user_relation[itemAlpha].size();
		if (sizei<=0)
		{
			continue;
		}
		for (int itemBeta=0; itemBeta< itemAlpha; itemBeta++)
		{
			if (itemAlpha==itemBeta)
			{
				continue;
			}
			double temp = 0.0;
			sizej = network.item_user_relation[itemBeta].size();
			if (sizej<=0)
			{
				continue;
			}
			SimpleHashSet<int> SimpleHashSet((sizei+sizej)*4);
			allcommonUsers = hashInterSection(&SimpleHashSet,network.item_user_relation[itemAlpha],sizei,network.item_user_relation[itemBeta],sizej);
			SimpleHashSet.SetLength(0);
			//cout<<allcommonUsers<<" "<<itemsCommonNeighbor.size()<<endl;
			if (allcommonUsers == 0)
			{
				continue;
			}			
			for (unsigned int commonuser = 0;commonuser<itemsCommonNeighbor.size();commonuser++)
			{
				if (network.user_item_relation[itemsCommonNeighbor[commonuser]].size() != 0)
				{
					temp += 1.0/(network.user_item_relation[itemsCommonNeighbor[commonuser]].size());
					/*cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;*/
				}else{
					cout<<itemsCommonNeighbor[commonuser]<<" network.user_item_relation[commonuser] size is :"<< network.user_item_relation[itemsCommonNeighbor[commonuser]].size()<<endl;
				}
			}
			//lamada = pow(sizej*1.0/itemMaxDegree,gama*1.0);
			//cout<<"this is Cold_Start and gama is :	"<<gama<<" lamada "<<lamada<<endl;
			//transformationMatrix(itemAlpha,itemBeta) = temp/(pow((sizei*1.0),(1-powItemDegree[itemBeta]))*pow((sizej*1.0),powItemDegree[itemBeta]));
			double alphaDegree = pow((sizei*1.0),(1-powItemDegree[itemBeta]))*pow((sizej*1.0),powItemDegree[itemBeta]);
			double betaDegree = pow((sizej*1.0),(1-powItemDegree[itemAlpha]))*pow((sizei*1.0),powItemDegree[itemAlpha]);
			tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/alphaDegree));
			tripletList.push_back(TRIPLET(itemBeta,itemAlpha,temp/betaDegree));
			//tripletList.push_back(TRIPLET(itemAlpha,itemBeta,temp/(pow((sizei*1.0),(1-powItemDegree[itemBeta]))*pow((sizej*1.0),powItemDegree[itemBeta]))));
		}
		//cout<<"tripletList size is : "<<itemAlpha<<"\t"<<tripletList.size()<<endl;
	}
	SpMatrix mat(itemNum,itemNum);
	//cout<<"tripletList size is : "<<tripletList.size()<<endl;
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	//cout<<"setFromTriplets size is : "<<mat.size()<<endl;
	tripletList.clear();
	//set collected item score == 0

	for (int i = 0; i<userNum; i++)
	{
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//every use has a transformaionmatrix
		Matrix<double,Dynamic,1> tempv(itemNum);;
		tempv= Matrix<double,Dynamic,1>::Zero(itemNum);

		for (unsigned int j = 0; j<network.user_item_relation[i].size();j++)
		{
			tempv(network.user_item_relation[i][j],0) =1.0;
		}

		Matrix<double,Dynamic,1> answerV(itemNum);;
		answerV= Matrix<double,Dynamic,1>::Zero(itemNum);
		answerV = mat*tempv;

		for (int k = 0;k<itemNum;k++)
		{
			HybirditemScore[k] = answerV.row(k)[0];
			//userRecommondatedItemScore[i][k]=HybirditemScore[k];
		}
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}


/************************************************************************/
/* sita = spd gram = pd  sfsilong=re-nbi
bestsita: 1.2   bestgama: -0.6  bestefsilong: -0.8 bestrs:      0.0781399*
bestsita: 1     bestgama: -0.48 bestefsilong: -0.7 bestrs:      0.0775405/
/************************************************************************/
int Basied_PD_RE_MD(double lamada,double sita, double gama, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"Basied_PD_RE_MD "<<" lamada "<<lamada<<" sita "<<sita<<" gama "<<gama<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double firstTempItemScore[itemNum];
	double secondTempItemScore[itemNum];
	double hybirdDegree;
	int tempItemId = 0;
	double itemusersize = 0.0, useritemsize = 0.0;
	//emptyVector(&Hybirdscore);

	double powerItemDegree[itemNum];
	memset(powerItemDegree,0,sizeof(powerItemDegree));

	//itemRankForUser.clear();
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(firstTempItemScore,0,sizeof(firstTempItemScore));
		memset(secondTempItemScore,0,sizeof(secondTempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				tempItemId = network.user_item_relation[i][j];
				itemusersize = network.item_user_relation[tempItemId].size();
				for (int k = 0; k < itemusersize; k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					//hybirdDegree = pow((double)network.item_user_relation[tempItemId].size(),lamada)*pow((double)network.user_item_relation[linkBackUser].size(),(1-lamada));
					hybirdDegree = network.item_user_relation[tempItemId].size();
					tempUserScore[linkBackUser] += (1.0/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					tempUserScore[m] = pow(tempUserScore[m],lamada);
					useritemsize = network.user_item_relation[m].size();
					double allItemDegree = 0.0;
					for (int l = 0; l < useritemsize; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						itemusersize = network.item_user_relation[linkBackItem].size();

						if (powerItemDegree[linkBackItem] == 0)
						{
							powerItemDegree[linkBackItem] = pow(itemusersize,sita);
						}
						allItemDegree += powerItemDegree[linkBackItem];
					}

					for (int l = 0; l < useritemsize ; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];

						if (powerItemDegree[linkBackItem] == 0)
						{
							itemusersize = network.item_user_relation[linkBackItem].size();
							powerItemDegree[linkBackItem] = pow(itemusersize,sita);
						}						

						hybirdDegree = powerItemDegree[linkBackItem]/allItemDegree;
						firstTempItemScore[linkBackItem] += (tempUserScore[m]*hybirdDegree);
					}
				}
			}
			//re nbi, do defussion again, item to user
			memset(tempUserScore,0,sizeof(tempUserScore));
			for (int i2 = 0; i2< itemNum; i2++)
			{
				if (firstTempItemScore[i2]>0)
				{
					tempItemId = i2;
					itemusersize = network.item_user_relation[tempItemId].size();
					for (int k = 0; k < itemusersize; k++)
					{
						int linkBackUser = network.item_user_relation[tempItemId][k];
						hybirdDegree = network.item_user_relation[tempItemId].size();
						tempUserScore[linkBackUser] += (firstTempItemScore[tempItemId]/hybirdDegree);
					}
				}
			}

			//user to item
			for (int m1 = 0; m1<userNum; m1++)
			{
				if (tempUserScore[m1]>0)
				{
					tempUserScore[m1] = pow(tempUserScore[m1],lamada);

					useritemsize = network.user_item_relation[m1].size();
					double allItemDegree = 0.0;
					for (int l = 0; l < useritemsize; l++)
					{
						int linkBackItem = network.user_item_relation[m1][l];
						itemusersize = network.item_user_relation[linkBackItem].size();

						if (powerItemDegree[linkBackItem] == 0)
						{
							powerItemDegree[linkBackItem] = pow(itemusersize,sita);
						}
						allItemDegree += powerItemDegree[linkBackItem];
					}
					for (int l = 0; l < useritemsize ; l++)
					{
						int linkBackItem = network.user_item_relation[m1][l];

						if (powerItemDegree[linkBackItem] == 0)
						{
							itemusersize = network.item_user_relation[linkBackItem].size();
							powerItemDegree[linkBackItem] = pow(itemusersize,sita);
						}	

						hybirdDegree = powerItemDegree[linkBackItem]/allItemDegree;
						secondTempItemScore[linkBackItem] += (tempUserScore[m1]*hybirdDegree);
					}
				}
			}

			//combian two times defussion
			for (int i1 = 0; i1< itemNum; i1++)
			{
				HybirditemScore[i1] = firstTempItemScore[i1]+gama*secondTempItemScore[i1];
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

//bestsita: 5.52  bestgama: 1     bestefsilong: 0 bestrs: 0.107266
int Heter_PD_RE_MD(double lamada,double sita, double gama, SimpleNetwork &network = combainTrainAndLearningSet)
{
	cout<<"Heter_PD_RE_MD "<<" lamada "<<lamada<<" sita "<<sita<<" gama "<<gama<<endl;
	funcName = __FUNCTION__;
	if (checkRuning())
	{		
		return -1;
	}
	double tempUserScore[userNum];
	double firstTempItemScore[itemNum];
	double secondTempItemScore[itemNum];
	double hybirdDegree;
	int tempItemId = 0;
	double initialScore = 0.0;
	double itemusersize = 0.0, useritemsize = 0.0;
	//emptyVector(&Hybirdscore);
	//emptyVector(&itemRankForUser);

	double powerItemDegree[itemNum];
	memset(powerItemDegree,0,sizeof(powerItemDegree));

	//itemRankForUser.clear();
	//emptyVector(&userRecommondatedItemScore);
	for (int i = 0; i<userNum; i++)
	{
		memset(tempUserScore,0,sizeof(tempUserScore));
		memset(firstTempItemScore,0,sizeof(firstTempItemScore));
		memset(secondTempItemScore,0,sizeof(secondTempItemScore));
		memset(unSortedScore,0,sizeof(unSortedScore));
		memset(HybirditemScore,0,sizeof(HybirditemScore));
		//item to user
		int tsize = network.user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				tempItemId = network.user_item_relation[i][j];
				itemusersize = network.item_user_relation[tempItemId].size();
				if (powerItemDegree[tempItemId] == 0)
				{
					initialScore = pow(itemusersize,lamada);
					powerItemDegree[tempItemId] = pow(itemusersize,lamada);
				}
				initialScore = powerItemDegree[tempItemId];

				for (int k = 0; k < itemusersize; k++)
				{
					int linkBackUser = network.item_user_relation[tempItemId][k];
					//hybirdDegree = pow((double)network.item_user_relation[tempItemId].size(),lamada)*pow((double)network.user_item_relation[linkBackUser].size(),(1-lamada));
					hybirdDegree = network.item_user_relation[tempItemId].size();
					tempUserScore[linkBackUser] += (initialScore/hybirdDegree);
				}
			}
			//user to item
			for (int m = 0; m<userNum; m++)
			{
				if (tempUserScore[m]>0)
				{
					useritemsize = network.user_item_relation[m].size();
					double allItemDegree = 0.0;
					for (int l = 0; l < useritemsize; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						if (powerItemDegree[linkBackItem] == 0)
						{
							itemusersize = network.item_user_relation[linkBackItem].size();
							powerItemDegree[linkBackItem] = pow(itemusersize,sita);
						}
						allItemDegree += powerItemDegree[linkBackItem];
					}
					for (int l = 0; l < useritemsize ; l++)
					{
						int linkBackItem = network.user_item_relation[m][l];
						if (powerItemDegree[linkBackItem] == 0)
						{
							itemusersize = network.item_user_relation[linkBackItem].size();
							powerItemDegree[linkBackItem] = pow(itemusersize,sita);
						}
						hybirdDegree = powerItemDegree[linkBackItem]/allItemDegree;
						firstTempItemScore[linkBackItem] += (tempUserScore[m]*hybirdDegree);
					}
				}
			}
			//re nbi, do defussion again, item to user
			memset(tempUserScore,0,sizeof(tempUserScore));
			for (int i2 = 0; i2< itemNum; i2++)
			{
				if (firstTempItemScore[i2]>0)
				{
					tempItemId = i2;
					itemusersize = network.item_user_relation[tempItemId].size();
					for (int k = 0; k < itemusersize; k++)
					{
						int linkBackUser = network.item_user_relation[tempItemId][k];
						hybirdDegree = network.item_user_relation[tempItemId].size();
						//initialScore = pow(itemusersize,lamada);
						tempUserScore[linkBackUser] += (firstTempItemScore[tempItemId]/hybirdDegree);
					}
				}
			}

			//user to item
			for (int m1 = 0; m1<userNum; m1++)
			{
				if (tempUserScore[m1]>0)
				{
					double allItemDegree = 0.0;
					useritemsize = network.user_item_relation[m1].size();
					for (int l = 0; l < useritemsize; l++)
					{
						int linkBackItem = network.user_item_relation[m1][l];
						itemusersize = network.item_user_relation[linkBackItem].size();
						
						if (powerItemDegree[linkBackItem] == 0)
						{
							powerItemDegree[linkBackItem] = pow(itemusersize,sita);
						}
						allItemDegree += powerItemDegree[linkBackItem];
					}
					for (int l = 0; l < useritemsize ; l++)
					{
						int linkBackItem = network.user_item_relation[m1][l];

						if (powerItemDegree[linkBackItem] == 0)
						{
							itemusersize = network.item_user_relation[linkBackItem].size();
							powerItemDegree[linkBackItem] = pow(itemusersize,sita);
						}	

						hybirdDegree = powerItemDegree[linkBackItem]/allItemDegree;
						secondTempItemScore[linkBackItem] += (tempUserScore[m1]*hybirdDegree);
					}
				}
			}

			//combian two times defussion
			for (int i1 = 0; i1< itemNum; i1++)
			{
				HybirditemScore[i1] = firstTempItemScore[i1]+gama*secondTempItemScore[i1];
			}
			startCalculateRS(tsize,i,network);
		}
	}
	isOtherMathodRunning = false;
	return 0;
}

void writefile(string filename,string content)
{
//    if( remove( filename ) != 0 ){
//		cout<<filename<<endl;
//		perror( "Error deleting file" );
//	}

	ofstream resultfile(filename,ios::app);           //打开文件用于写,若文件不存在就创建它
	if(!resultfile)
		cout << "open file：" << filename << " failure！";;                 //打开文件失败则结束运行
	resultfile<<content<<endl;     //使用插入运算符写文件内容
	resultfile.close();         //关闭文件

//	FILE *F = fopen(filename.c_str(), "wb+");
//    cout<<filename<<endl;
//    const char *p =  content.c_str();
//    cout<<content<<endl;
//    fprintf(F,"%s\n", p);
//    fclose(F);

}

void removeAndWritefile(string filename,string content)
{
	if( remove( filename.c_str()) != 0 ){
		cout<<"Error deleting file "<<filename<<endl;
		perror( "Error deleting file" );
	}
	ofstream resultfile(filename,ios::app);           //打开文件用于写,若文件不存在就创建它
	if(!resultfile)
		cout << "open file：" << filename << " failure！";;                 //打开文件失败则结束运行
	resultfile<<content<<endl;     //使用插入运算符写文件内容
	resultfile.close();         //关闭文件
}

void testifright()
{
	init();
	for (double lamada = 0.0; lamada<=1.0; lamada+=0.02)
	{
		WeightiedGrandNodeHC(lamada);
		ProbS(1);
		cout<<"lamada is: "<<lamada;
		cout<<"	RankingScore	"<<RankingScoreNotCollect()<<endl;
		//cout<<"	RankingScore	"<<RankingScoreNotCollect()<<"	localRScore	"<<LocalRankingScoreNotCollect()<<endl;
		/*combainTrainAndLearningSet.empty();
		init();*/
	}
	//combainTrainAndLearningSet.hybirdHAndPNonLinaer(0.18);

	/*combainTrainAndLearningSet.MatrixHybirdReal(0.8);
	cout<<"RankingScore	"<<RankingScoreNotCollect()<<endl;
	cout<<"localRankScore	"<<LocalRankingScoreNotCollect()<<endl;*/
}

void getProbs()
{
	init();
	double bestsita = 0.0, bestlamada = 0.0,bestgama = 0.0,bestRankingScore = 1.0, temprs = 0.0;
	stringstream tempcontentstream,resultFilePathSS;
	for (int lamada = 1; lamada<=1; lamada++)
	{

		long start=clock(),end(0);
		ProbS(lamada,trainingSet);
		if(resultFilePathSS.str().size()==0)
		{
			resultfile = generateFiledate();
			resultFilePathSS<<resultfilepath<<resultfile<<"_"<<funcName<<"_step"<<totalstepts<<".txt";
		}
		vector<double> accuracyVector(3,0);
		cout<<RANKINGSCORE<<"	"<<fullcounter<<" lamada: "<<lamada<<"	RS: ";
		temprs = getRankingScore();
		cout<<temprs<<endl;
		tempcontentstream<<funcName<<"\tstep"<<totalstepts<<"\t"<<lamada<<"\t"<<temprs;//<<endl;
		//tempcontentstream<<lamada<<"\t"<<temprs;
		string tempcontents = tempcontentstream.str();
		//cout<<"RANKINGSCORE: "<<tempcontents<<endl;
		accuracyVector = ComputeAccuracy();
		cout<<"Precision	"<<Precision()<<"   "<<accuracyVector[0]<<endl;
		cout<<"recall          "<<Recall()<<"   "<<accuracyVector[1]<<endl;
		cout<<"Popularity	"<<Popularity()<<"   "<<accuracyVector[2]<<endl;
		writefile(resultFilePathSS.str(),tempcontentstream.str());
		//                cout<<"sita: "<<sita<<" gama: "<<gama<<" lamada: "<<lamada<<"	RS: "<<temprs<<endl;
		tempcontentstream.str("");
		if ((temprs - bestRankingScore) <= wucha)
		{
			bestRankingScore = temprs;
			bestlamada = lamada;
			//                    bestsita = sita;
			//                    bestgama = gama;
		}
		end=clock();
		long result=(end-start);
		cout<<"time is : "<<result<<endl;
		//            }
		//        }
	}
	//    tempcontentstream<<"bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	bestlamada: "<<bestlamada<<"	bestrs:	"<<bestRankingScore<<endl;
	tempcontentstream<<"bestlamada: "<<bestlamada<<"	bestrs:	"<<bestRankingScore<<endl;
	writefile(resultfile,tempcontentstream.str());
	cout<<"bestlamada: "<<bestlamada<<"	bestrs:	"<<bestRankingScore<<endl;
	tempcontentstream.str("");
}

//bestgama:    2.9 bestsita:   3.2 bestefsilong:   0.1 bestrs:     0.084889
void train()
{
	//---------------------------------------------
	runingMode = 0;//set testdate to train. 80%---10%
	//---------------------------------------------

	init();
	double bestsita = 0.0, bestlamada = 0.0,bestgama = 0.0,bestRankingScore = 1.0, temprs = 0.0;
	for (double sita = 0.5; sita<=1.5; sita+=0.1)
	{
		for (double gama = -0.20; gama<=0.75; gama+=0.1)
		{
			for (double efsilong = 0.13; efsilong<=0.23; efsilong+=0.1)
			{
				cout<<"sita:	"<<sita<<" gama:	"<<gama<<" efsilong:	"<<efsilong;
				//SPD(sita,gama,trainingSet);
				//Basied_PD_RE_MD(sita,gama,efsilong,trainingSet);
				WHC(efsilong,trainingSet);
				//Heter_PD_RE_MD(sita,gama,efsilong,trainingSet);
				temprs = getRankingScore();
				cout<<"	RS:	"<<temprs<<endl;
				if ((temprs - bestRankingScore) <= wucha)
				{
					bestRankingScore = temprs;
					bestsita = sita;
					bestlamada = efsilong;
					bestgama = gama;
				}
			}
		}
		cout<<"bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	bestefsilong: "<<bestlamada<<" bestrs:	"<<bestRankingScore<<endl;
	}

	//trainingSet.addGrandNode();

	//for (double lamada = 1.2; lamada<=1.4; lamada+=0.01)
	//{
	//	//hybirdHAndPNonLinaer(lamada,trainingSet);
	//	//ProbS(1,trainingSet);
	//	//HHCMatrix(lamada,trainingSet);
	//	//HeatS(trainingSet);
	//	//Biased_Heat(lamada,trainingSet);
	//	//WHC(lamada,trainingSet);
	//	//IHCMatrix(lamada,trainingSet);
	//	//RE_NBI(lamada,trainingSet);
	//	//Heter_NBI(lamada,trainingSet);
	//	//PD(lamada,trainingSet);
	//	//Heter_PD()
	//	//URA_NBI(lamada,trainingSet);
	//	//Cold_Start(lamada,trainingSet);

	//	//cout<<"lamada is :	"<<lamada;
	//	cout<<"lamada is :	"<<lamada<<"	RankingScore is:	"<<getRankingScore()<<endl;
	//}
	//combainTrainAndLearningSet.hybirdHAndPNonLinaer(0.18);
	//oldNet.hybirdHAndPNonLinaer(0.2);
	//cout<<"RankingScore	"<<RankingScore()*1000<<endl;
	//cout<<"Precision	"<<Precision()<<endl;
	//cout<<"IntraSimilarity	"<<IntraSimilarity()<<endl;
	//cout<<"HammingDistance	"<<HammingDistance()<<endl;
	//cout<<"Popularity	"<<Popularity()<<endl;
}

void test()
{
	//0.31 1.95
	//combainTrainAndLearningSet.SPD(0.84);

	long start=clock(),end(0);
	init();
	//hybirdHAndPNonLinaer(0.22,combainTrainAndLearningSet);
	//combainTrainAndLearningSet.Basied_PD_RE_MD(1.1,-0.5,-0.78);
	//IHCMatrix(1.14);//0,-0.48,-0.7
	//Heter_PD_RE_MD(-0.1,-0.44,-0.72);
	//Heter_PD_RE_MD(-0.16,-0.48,-0.7);
	//Heter_PD_RE_MD(0,-0.5,-0.7);
	//ProbS(1);
	//B_Rank();
	//HeatS();
	WHC(0.16);
	//Biased_Heat(0.80);
	//IHC(0.8);
	//HHCMatrix(0.8);
	//RE_NBI(-0.76);
	//Heter_NBI(-0.64);
	//PD(-0.8);
	//URA_NBI(-0.28);
	//Cold_Start(1.36);
	//Heter_PD(-0.09,-0.78);
	//SPD(1.77,0.34);
	end=clock();
	long result=(end-start);
	cout<<" time is : "<<result<<endl;
	/*cout<<"RSNotCollect	"<<RankingScoreNotCollect()<<endl;
	cout<<"localRankScore	"<<LocalRankingScoreNotCollect()<<endl;*/
	cout<<"RSNotCollect	"<<getRankingScore()<<endl;
	cout<<"localRankScore	"<<getLocalRankingScore()<<endl;
	cout<<"Precision	"<<Precision()<<endl;
	cout<<"Recall		"<<Recall()<<endl;
	cout<<"IntraSimilarity	"<<IntraSimilarity()<<endl;
	cout<<"HammingDistance	"<<HammingDistance()<<endl;
	cout<<"Popularity	"<<Popularity()<<endl;
}

vector<paraGroup> getLastRuningParmeter()
{
	char buffer[256];
	ifstream infile("oldparam.txt");	
	if(!infile)
	{
		cout << "can not open this file oldparam.txt"<< endl;
		return paraGroupVector;
	}

	if (infile.is_open())
	{
		int times = 0;
		int bestTimes = 0;
		double lamada = 0,sita= 0,gama = 0;
		double oldRS = 0;
		double rankscore = 0;
		double lrs = 0;
		double pricis = 0;
		double recall = 0;
		double intrSim = 0;
		double hamdis = 0;
		double Popul = 0;
		cout << "times" << "\t" << "lamada" << "\t" << "sita" << "\t" << "gama" << "\t" << "Bests" << "\t" << "oldRS" << "\t" << "rs" << "\t" << "lrs" << "\t" << "pricis" << "\t" << "recall" << "\t" << "intrSim" << "\t" << "hamdis" << "\t" << "Popul" << endl;
		int counter=0;
		while (!infile.eof() )
		{	
			infile.getline (buffer,256);
			stringstream ss(buffer);
			if (ss.str().length() < 1){
				continue;
			}
			ss>>times>>lamada>>sita>>gama>>bestTimes>>oldRS>>rankscore>>lrs>>pricis>>recall>>intrSim>>hamdis>>Popul;
			//cout << "line " <<ss.str() << endl;
			ss.str("");
			//cout<<times<<"\t"<<lamada<<"\t"<<sita<<"\t"<<gama<<"\t"<<bestTimes<<"\t"<<oldRS<<"\t"<<rankscore<<"\t"<<lrs<<"\t"<<pricis<<"\t"<<recall<<"\t"<<intrSim<<"\t"<<hamdis<<"\t"<<Popul<<endl;;
			//sscanf(buffer,"%d%lf%lf%lf%d%lf%lf%lf%lf%lf%lf%lf%lf",&times,&lamada,&sita,&gama,&bestTimes,&oldRS,&rankscore,&lrs,&pricis,&recall,&intrSim,&hamdis,&Popul);
			//sscanf(buffer,"%d%lf%lf%lf%d%f",&times,&lamada,&sita,&gama,&bestTimes,&oldRS);
			//cout<<"befor "<<times<<" "<<lamada<<"	"<<sita<<" "<<gama<<endl;
			if (counter >= 10 && times == 0)
			{
				cout<<"times more over 10! "<<times<<endl;
				paraGroupVector.clear();
				//return paraGroupVector;
			}
			paraGroup temPg;
			temPg.times = times;
			temPg.lamada = lamada;
			temPg.sita = sita;
			temPg.gama = gama;
			temPg.rs = rankscore;
			temPg.isTheBestTimes = bestTimes;
			temPg.oldRS = oldRS;
			temPg.lrs = lrs;
			temPg.rs = rankscore;
			temPg.pricis = pricis;
			temPg.recall = recall;
			temPg.intrSim = intrSim;
			temPg.hamdis = hamdis;
			temPg.Popul = Popul;
			//cout<<"after "<<temPg.times<<" "<<temPg.lamada<<"	"<<temPg.sita<<" "<<temPg.gama<<endl;
			paraGroupVector.push_back(temPg);
			counter++;
		}
		infile.close();
		for (int i = 0; i <= times; i++){
			cout << paraGroupVector[i].times << "\t" << paraGroupVector[i].lamada << "\t" << paraGroupVector[i].sita << "\t" << paraGroupVector[i].gama << "\t" << paraGroupVector[i].isTheBestTimes << "\t" << paraGroupVector[i].oldRS << "\t" << paraGroupVector[i].rs << "\t" << paraGroupVector[i].lrs << "\t" << paraGroupVector[i].pricis << "\t" << paraGroupVector[i].recall << "\t" << paraGroupVector[i].intrSim << "\t" << paraGroupVector[i].hamdis << "\t" << paraGroupVector[i].Popul << endl;
		}
	}
	else
	{
		cout<<"Error,can't open...\n";
	}
	return paraGroupVector;
}

//这里读取之前的参数来计算local rs
double calculataLocalRsAgain(){
	const int cishu = 10;
	bool hasOldPara = false;double avg = 0.0, sum = 0.0, stadardivation = 0.0;
	oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem

	ifstream infile("oldparam.txt");
	if (!infile)
	{
		cout << "can not open this file oldparam.txt" << endl;
		hasOldPara = false;
	}
	else{
		getLastRuningParmeter();
		hasOldPara = true;
	}

	for (int times = 0; times < cishu; times++)
	{
		if (hasOldPara)
		{
			init(times);
			//---------------------------------------------
			runingMode = 1;//set testdate to testSet. 90%---10%
			//---------------------------------------------
			//WHCMatrix(paraGroupVector[times].lamada);
			//IHC(paraGroupVector[times].lamada);
			//IMD(paraGroupVector[times].lamada);
			//IHCMatrix(paraGroupVector[times].lamada);
			//HHC(paraGroupVector[times].lamada);
			//hybirdHAndPNonLinaer(paraGroupVector[times].lamada);
			//RE_NBI(paraGroupVector[times].lamada);
			//Heter_NBI(paraGroupVector[times].lamada);
			//PD(paraGroupVector[times].lamada);
			//URA_NBI(paraGroupVector[times].lamada);
			//Biased_Heat(paraGroupVector[times].lamada);
			//Cold_Start(paraGroupVector[times].lamada);
			//Cold_StartMatrix(paraGroupVector[times].lamada);
			//caclSparseNetwork(paraGroupVector[times].lamada);
			//NCF(paraGroupVector[times].lamada);
			//NCFNew(paraGroupVector[times].lamada);
			//MCF();
			//HeatS();
			//B_Rank();
			//ProbS(1);
			//ProbS(paraGroupVector[times].lamada);
			//Heter_PD(paraGroupVector[times].lamada, paraGroupVector[times].sita);
			//SPD(paraGroupVector[times].lamada, paraGroupVector[times].sita);	
			Basied_PD_RE_MD(paraGroupVector[times].lamada, paraGroupVector[times].sita,paraGroupVector[times].gama);
			//Heter_PD_RE_MD(paraGroupVector[times].lamada, paraGroupVector[times].sita, paraGroupVector[times].gama);

			double templs = getLocalRankingScore();
			cout << "localrs	" << templs << "	and lamada is: " << paraGroupVector[times].lamada << endl;
			sum += templs;
		}
	}
	sum = sum / cishu;
	cout << "------------------------------ "<<endl;
	cout << "avg localrs is: "<< sum << endl;
	return sum;
}

double calculata9010RsOne() {
	const int cishu = 10;
	bool hasOldPara = false;
	double oldParaTimesRange = 3;
	int counterX = 0, isBestTimes = 0;
	double rsAndOthersArray[7][cishu];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double evaluationIndex[7];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS

	double avg = 0.0, sum = 0.0, stadardivation = 0.0;
	double bestlamada = 0.0, bestRankingScore = 10.0, temprs = 0.0, avgLamada = 0.0, avgOldRS = 0.0, lastRankingScore = 10.0;
	double prametesArray[cishu][5];
	memset(prametesArray, 0, sizeof(prametesArray));
	//cout<<" First prametesArray[9][4] is : "<<prametesArray[cishu-1][4]<<endl;
	memset(rsAndOthersArray, 0, sizeof(rsAndOthersArray));
	memset(evaluationIndex, 0, sizeof(evaluationIndex));
	oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem
	string resultfile = generateFilename();

	ifstream infile("oldparam.txt");
	if (!infile)
	{
		cout << "can not open this file oldparam.txt" << endl;
		hasOldPara = false;
	}
	else{
		getLastRuningParmeter();
		hasOldPara = true;
	}

	for (int times = 0; times < cishu; times++)
	{
		bool isWrongRange = false;
		lastRankingScore = 10.0;
		bestlamada = 0.0, bestRankingScore = 1.0, temprs = 0.0;

		if (hasOldPara)
		{
			if (paraGroupVector[times].isTheBestTimes > 0.3 && paraGroupVector[times].isTheBestTimes < 100)//如果对这个数据集来说参数已经是最优的，那么就直接跳过这个数据集，还是用以前的参数。<1000
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][2] = paraGroupVector[times].gama;;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes + 1;
				//rsAndOthersArray[7][times] = bestRankingScore;
				continue;
			}
			else
			{
				cout << "isTheBestTimes is wired" << paraGroupVector[times].isTheBestTimes << endl;
			}
			cout << "old lamada: " << paraGroupVector[times].lamada << endl;
			if (paraGroupVector[times].lamada != 0)
			{
				minlamada = paraGroupVector[times].lamada - oldParaTimesRange*lamadaIncremental;
				maxlamada = minlamada + 2 * oldParaTimesRange*lamadaIncremental;
				//lamadaIncremental = 0.01;
			}
		}

		//cout<<"Second prametesArray[times][4] is : "<<prametesArray[times][4]<<endl;
		cout << "minlamada: " << minlamada << " maxlamada: " << maxlamada << " lamadaIncremental: " << lamadaIncremental << endl;

		init(times);
		for (double lamada = minlamada; lamada <= maxlamada; lamada += lamadaIncremental)
		{
			//---------------------------------------------
			runingMode = 1;//set testdate to train. 80%---10%
			//---------------------------------------------

			//WHCMatrix(lamada);
			//IHC(lamada);
			//IMD(lamada);
			//IHCMatrix(lamada);
			//HHC(lamada);
			//hybirdHAndPNonLinaer(lamada);
			//RE_NBI(lamada);
			//Heter_NBI(lamada);
			//PD(lamada);
			//URA_NBI(lamada);
			//Biased_Heat(lamada);
			//Cold_Start(lamada);
			//Cold_StartMatrix(lamada);
			//caclSparseNetwork(lamada);
			//NCF(lamada);
			//NCFNew(lamada);
			//MCF();
			//HeatS();
			//B_Rank();
			//ProbS(1);
			ProbS(lamada);
			temprs = getRankingScore();
			cout << "lamada: " << lamada << "	sita: " << 0 << " gama: " << 0 << "	nowRS: " << temprs << "	lastRankingScore " << lastRankingScore << endl;
			if ((temprs - bestRankingScore) <= wucha)
			{
				bestRankingScore = temprs;
				bestlamada = lamada;
			}

			if (temprs > lastRankingScore)
			{
				//isWrongRange = true;
				//cout<<"lamada: "<<lamada<<"	sita: "<<0<<" gama: "<<0<<"	RS: "<<temprs<<endl;
				cout << "last is the best bestlamada: " << bestlamada << "	temprs:	" << temprs << endl;
				cout << "---------------------------------------" << times << "---------------------------------------------" << endl;

				lastRankingScore = 10.0;
				break;
			}

			lastRankingScore = temprs;
		}
		prametesArray[times][3] = bestRankingScore;


		stringstream errorMessagess;
		if (bestlamada >= maxlamada)
		{
			isWrongRange = true;
			errorMessagess << "need bigger lamada: bestlamada " << bestlamada << "	maxlamada: " << maxlamada << "	bestrs:	" << bestRankingScore << endl;
		}
		else if (bestlamada <= minlamada)
		{
			isWrongRange = true;
			errorMessagess << "need smaller lamada: bestlamada " << bestlamada << "	minlamada: " << minlamada << "	bestrs:	" << bestRankingScore << endl;
		}
		if (isWrongRange)
		{
			//cout<<"---------------------------------------isWrongRange-----------------------------------------------"<<endl;
			string tempcontents = errorMessagess.str();
			cout << tempcontents << endl;
			cout << "--------------------------------------isWrongRange finish this" << times << "------------------------" << endl;
		}

		//here to add some code to record the old rs
		if (hasOldPara && (paraGroupVector[times].oldRS<1 && paraGroupVector[times].oldRS>0))
		{
			cout << "old one: " << paraGroupVector[times].oldRS << "	bestRankingScore:	" << bestRankingScore << (paraGroupVector[times].oldRS - bestRankingScore) << endl;
			if ((paraGroupVector[times].oldRS - bestRankingScore) <= wucha)
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes + 1;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				cout << "bestlamada is old one: " << prametesArray[times][0] << "	bestrs:	" << prametesArray[times][3] << endl;	
			}
			else if (bestlamada == paraGroupVector[times].lamada){
				prametesArray[times][0] = bestlamada;
				prametesArray[times][3] = bestRankingScore;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes;
				cout << "bestlamada: " << prametesArray[times][0] << "	bestrs:	" << prametesArray[times][3] << endl;
			}
			else{
				prametesArray[times][0] = bestlamada;
				prametesArray[times][3] = bestRankingScore;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes;
				cout << "bestlamada: " << prametesArray[times][0] << "	bestrs:	" << prametesArray[times][3] << endl;;
			}
		}
		else{
			prametesArray[times][0] = bestlamada;
			prametesArray[times][3] = bestRankingScore;
			prametesArray[times][4] = 0;
			cout << "bestlamada: " << prametesArray[times][0] << "	bestrs:	" << prametesArray[times][3] << endl;
		}
		sum += prametesArray[times][3];
	}
	avg = sum / cishu;

	stringstream tempcontentstream;
	tempcontentstream << "t	lamada	sita	gama	oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul" << endl;
	writefile(resultfile, tempcontentstream.str());
	tempcontentstream.str("");
	//cout<<"Thrid  prametesArray[times][4] is : "<<prametesArray[cishu-1][4]<<endl;
	stringstream oldParaFile;
	stringstream oldParaLogFile;

	for (int times = 0; times < cishu; times++)
	{
		avgLamada += prametesArray[times][0];
		avgOldRS += prametesArray[times][3];

		//		cout<<"net "<<times<<" lamada "<<prametesArray[times][0]<<" rs is: "<<prametesArray[times][3]<<endl;
		tempcontentstream << times << "\t" << prametesArray[times][0] << "\t" << prametesArray[times][1] << "\t" << prametesArray[times][2];
		//oldParaFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\t"<<isBestTimes<<"\t"<<prametesArray[times][3]<<"\r\n";
		//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;;
		tempcontentstream << "\t" << prametesArray[times][3] << "\t" << rsAndOthersArray[0][times] << "\t" << rsAndOthersArray[1][times] << "\t" << rsAndOthersArray[2][times] << "\t" << rsAndOthersArray[3][times] << "\t" << rsAndOthersArray[4][times] << "\t" << rsAndOthersArray[5][times] << "\t" << rsAndOthersArray[6][times];
		cout << tempcontentstream.str() << endl;
		writefile(resultfile, tempcontentstream.str());
		tempcontentstream.str("");

		oldParaLogFile << times << "\t" << prametesArray[times][0] << "\t" << prametesArray[times][1] << "\t" << prametesArray[times][2] << "\t" << prametesArray[times][3] << "\t" << prametesArray[times][4] << "\r\n";
		
		oldParaFile << times << "\t" << prametesArray[times][0] << "\t" << prametesArray[times][1] << "\t" << prametesArray[times][2] << "\t" << prametesArray[times][4];
		oldParaFile << "\t" << prametesArray[times][3] << "\t" << 100 << "\t" << 100 << "\t" << rsAndOthersArray[2][times] << "\t" << rsAndOthersArray[3][times] << "\t" << rsAndOthersArray[4][times] << "\t" << rsAndOthersArray[5][times] << "\t" << rsAndOthersArray[6][times];
		
		writefile("oldparam.txt", oldParaFile.str());
		oldParaFile.str("");
		//cout<<"Fourth  prametesArray[times][4] is : "<<prametesArray[times][4]<<endl;
	}
	avgOldRS = avgOldRS / cishu;
	avgLamada = avgLamada / cishu;
	oldParaLogFile << "\r\n avg 90_10 rs is: " << avgOldRS << "	avgLamada is:	" << avgLamada << endl;
	cout << oldParaLogFile.str() << endl;
	removeAndWritefile("oldparam_log.txt", oldParaLogFile.str());
	oldParaLogFile.str("");
}

//两个参数的
double calculata9010RsTwo()
{
	const int cishu = 10;
	bool hasOldPara = false;
	int counterX = 0, isBestTimes = 0;
	double rsAndOthersArray[7][cishu];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double evaluationIndex[7];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double oldParaTimesRange = 3;
	double avg = 0.0, sum = 0.0, stadardivation = 0.0;
	double bestsita = 0.0, bestlamada = 0.0, bestRankingScore = 10.0, temprs = 0.0, avgLamada = 0.0, avgSita = 0.0, avgOldRS = 0.0, lastRankingScore = 10.0, lastLamada = 0;
	double prametesArray[cishu][5];
	memset(evaluationIndex, 0, sizeof(evaluationIndex));
	memset(prametesArray, 0, sizeof(prametesArray));
	memset(rsAndOthersArray, 0, sizeof(rsAndOthersArray));
	oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem
	string resultfile = generateFilename();

	ifstream infile("oldparam.txt");
	if (!infile)
	{
		cout << "can not open this file oldparam.txt" << endl;
		hasOldPara = false;
	}
	else{
		getLastRuningParmeter();
		hasOldPara = true;
	}

	for (int times = 0; times<cishu; times++)
	{
		bool isWrongRange = false;
		lastRankingScore = 10.0;
		bestlamada = 0.0, bestsita = 0.0, bestRankingScore = 1.0, temprs = 0.0;

		if (hasOldPara)
		{
			if (paraGroupVector[times].isTheBestTimes > 0.3 && paraGroupVector[times].isTheBestTimes < 100)//如果对这个数据集来说参数已经是最优的，那么就直接跳过这个数据集，还是用以前的参数。<1000
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][2] = paraGroupVector[times].gama;;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes + 1;

				rsAndOthersArray[0][times] = paraGroupVector[times].rs;
				rsAndOthersArray[1][times] = paraGroupVector[times].lrs;
				//rsAndOthersArray[7][times] = bestRankingScore;

				//prametesArray[times][3] = rsAndOthersArray[0][times];
				cout << "RS on 9---1 dividing datab	" << rsAndOthersArray[0][times] << endl;
				sum += rsAndOthersArray[0][times];

				continue;
			}
			else
			{
				cout << "isTheBestTimes is wired" << paraGroupVector[times].isTheBestTimes << endl;
			}
			/*for (int i = 0; i< 10;i++)
			{
			cout<<paraGroupVector[i].lamada<<endl;
			}*/
			cout << "old lamada: " << paraGroupVector[times].lamada << "old sita: " << paraGroupVector[times].sita << endl;

			if (paraGroupVector[times].lamada != 0)
			{
				minlamada = paraGroupVector[times].lamada - oldParaTimesRange*lamadaIncremental;
				maxlamada = minlamada + 2 * oldParaTimesRange*lamadaIncremental;
				//lamadaIncremental = 0.01;
			}

			if (paraGroupVector[times].sita != 0)
			{
				minsita = paraGroupVector[times].sita - oldParaTimesRange*sitaIncremental;
				maxsita = minsita + 2 * oldParaTimesRange*sitaIncremental;
				//sitaIncremental = 0.01;
			}
		}

		cout << "minlamada: " << minlamada << " maxlamada: " << maxlamada << " lamadaIncremental: " << lamadaIncremental << endl;
		cout << "minsita: " << minsita << " maxsita: " << maxsita << " sitaIncremental: " << sitaIncremental << endl;

		init(times);
		for (double lamada = minlamada; lamada <= maxlamada; lamada += lamadaIncremental)
		{
			if (isWrongRange && (minsita == maxsita) && (maxgama == mingama))
			{
				cout << "minsita==maxsita &&(maxgama==mingama)" << endl;
				break;
			}
			//reset isWrongRange
			isWrongRange = false;

			for (double sita = minsita; sita <= maxsita; sita += sitaIncremental)
			{

				//---------------------------------------------
				runingMode = 1;//set testdate to train. 80%---10%
				//---------------------------------------------

				//counterX++;

				//Heter_PD(lamada,sita);
				SPD(lamada,sita);
				
				temprs = getRankingScore();

				if ((temprs - bestRankingScore) <= wucha)
				{
					bestRankingScore = temprs;
					bestsita = sita;
					bestlamada = lamada;
				}
				cout << "lamada: " << lamada << "	sita: " << sita << "	nowRS: " << temprs << "	lastRankingScore " << lastRankingScore << endl;

				if ((temprs>lastRankingScore) && (lastLamada == lamada))
				{
					isWrongRange = true;
					//cout<<"lamada: "<<lamada<<"	sita: "<<sita<<"	RS: "<<temprs<<endl;
					cout << "last is the best bestlamada: " << bestlamada << "	bestsita: " << bestsita << "	temprs:	" << temprs << endl;
					cout << "---------------------------------------" << times << "---------------------------------------------" << endl;

					lastRankingScore = 10.0;
					break;
				}

				lastRankingScore = temprs;
				lastLamada = lamada;

				/*if ((temprs - bestRankingScore) <= wucha)
				{
				bestRankingScore = temprs;
				bestsita = sita;
				bestlamada = lamada;
				}*/
			}
		}

		stringstream errorMessagess;
		if (bestlamada >= maxlamada)
		{
			isWrongRange = true;
			errorMessagess << "need bigger lamada: bestlamada " << bestlamada << "	maxlamada: " << maxlamada << "	bestsita: " << bestsita << "	bestrs:	" << bestRankingScore << endl;
		}
		else if (bestlamada <= minlamada)
		{
			isWrongRange = true;
			errorMessagess << "need smaller lamada: bestlamada " << bestlamada << "	minlamada: " << minlamada << "	bestsita: " << bestsita << "	bestrs:	" << bestRankingScore << endl;
		}
		if (bestsita >= maxsita)
		{
			isWrongRange = true;
			errorMessagess << "need bigger sita: bestlamada " << bestlamada << "	bestsita: " << bestsita << "	maxsita: " << maxsita << "	bestrs:	" << bestRankingScore << endl;
		}
		else if (bestsita <= minsita)
		{
			isWrongRange = true;
			errorMessagess << "need smaller sita: bestlamada " << bestlamada << "	minsita: " << minsita << "	maxsita: " << maxsita << "	bestrs:	" << bestRankingScore << endl;
		}
		if (isWrongRange)
		{
			//cout<<"---------------------------------------isWrongRange-----------------------------------------------"<<endl;
			string tempcontents = errorMessagess.str();
			cout << tempcontents << endl;
			cout << "--------------------------------------isWrongRange finish this" << times << "------------------------" << endl;
		}

		//here to add some code to record the old rs
		if (hasOldPara && (paraGroupVector[times].oldRS<1 && paraGroupVector[times].oldRS>0))
		{
			if ((paraGroupVector[times].oldRS - bestRankingScore) <= wucha)
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes + 1;
				cout << "bestlamada is old one: " << prametesArray[times][0] << "	bestsita: " << prametesArray[times][1] << "	bestrs:	" << prametesArray[times][3] << endl;
			}
			else if ((bestlamada == paraGroupVector[times].lamada) && (bestsita == paraGroupVector[times].sita)){
				prametesArray[times][0] = bestlamada;
				prametesArray[times][1] = bestsita;
				prametesArray[times][3] = bestRankingScore;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes+1;
				cout << "bestlamada: " << bestlamada << "	bestsita: " << bestsita << "	bestrs:	" << bestRankingScore << endl;
			}
			else{
				prametesArray[times][0] = bestlamada;
				prametesArray[times][1] = bestsita;
				prametesArray[times][3] = bestRankingScore;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes;
				cout << "bestlamada: " << bestlamada << "	bestsita: " << bestsita << "	bestrs:	" << bestRankingScore << endl;
			}
			
		}else{
			prametesArray[times][0] = bestlamada;
			prametesArray[times][1] = bestsita;
			prametesArray[times][3] = bestRankingScore;
			prametesArray[times][4] = 0;
			cout << "bestlamada: " << bestlamada << "	bestsita: " << bestsita << "	bestrs:	" << bestRankingScore << endl;
		}

		//---------------------------------------------

		rsAndOthersArray[0][times] = prametesArray[times][3];
		//rsAndOthersArray[7][times] = bestRankingScore;

		//prametesArray[times][3] = rsAndOthersArray[0][times];
		cout << "RS on 9---1 dividing datab	" << rsAndOthersArray[0][times] << endl;
		sum += rsAndOthersArray[0][times];
	}
	avg = sum / cishu;

	stringstream tempcontentstream;
	tempcontentstream << "t	lamada	sita	gama	oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul" << endl;
	writefile(resultfile, tempcontentstream.str());
	tempcontentstream.str("");

	stringstream oldParaFile;
	stringstream oldParaLogFile;

	for (int times = 0; times<cishu; times++)
	{
		avgLamada += prametesArray[times][0];
		avgSita += prametesArray[times][1];
		avgOldRS += prametesArray[times][3];

		evaluationIndex[0] += rsAndOthersArray[0][times];
		evaluationIndex[1] += rsAndOthersArray[1][times];

		stadardivation += ((rsAndOthersArray[0][times] - avg)*(rsAndOthersArray[0][times] - avg));
		//		cout<<"net "<<times<<" lamada "<<prametesArray[times][0]<<" rs is: "<<prametesArray[times][3]<<endl;
		tempcontentstream << times << "\t" << prametesArray[times][0] << "\t" << prametesArray[times][1] << "\t" << prametesArray[times][2];
		//oldParaFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\t"<<"\t"<<isBestTimes<<prametesArray[times][3]<<"\r\n";
		//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;;
		tempcontentstream << "\t" << prametesArray[times][3] << "\t" << rsAndOthersArray[0][times] << "\t" << rsAndOthersArray[1][times] << "\t" << rsAndOthersArray[2][times] << "\t" << rsAndOthersArray[3][times] << "\t" << rsAndOthersArray[4][times] << "\t" << rsAndOthersArray[5][times] << "\t" << rsAndOthersArray[6][times];
		cout << tempcontentstream.str() << endl;
		writefile(resultfile, tempcontentstream.str());
		tempcontentstream.str("");

		oldParaLogFile << times << "\t" << prametesArray[times][0] << "\t" << prametesArray[times][1] << "\t" << prametesArray[times][2] << "\t" << prametesArray[times][3] << "\t" << prametesArray[times][4] << "\r\n";
		oldParaFile << times << "\t" << prametesArray[times][0] << "\t" << prametesArray[times][1] << "\t" << prametesArray[times][2] << "\t" << prametesArray[times][4];
		oldParaFile << "\t" << prametesArray[times][3] << "\t" << rsAndOthersArray[0][times] << "\t" << rsAndOthersArray[1][times] << "\t" << rsAndOthersArray[2][times] << "\t" << rsAndOthersArray[3][times] << "\t" << rsAndOthersArray[4][times] << "\t" << rsAndOthersArray[5][times] << "\t" << rsAndOthersArray[6][times];
		writefile("oldparam.txt", oldParaFile.str());
		oldParaFile.str("");

	}
	//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;
	removeAndWritefile("oldparam_log.txt", oldParaLogFile.str());
	oldParaLogFile.str("");

	avgLamada = avgLamada / cishu;
	avgSita = avgSita / cishu;
	avgOldRS = avgOldRS / cishu;

	evaluationIndex[0] = evaluationIndex[0] / cishu;
	evaluationIndex[1] = evaluationIndex[1] / cishu;

	stadardivation = sqrt(stadardivation / cishu);

	stringstream avgParamStream; 

	avgParamStream << funcName << endl;
	avgParamStream << "minlamada: " << minlamada << " maxlamada: " << maxlamada << " lamadaIncremental: " << lamadaIncremental << endl;
	avgParamStream << "minsita: " << minsita << " maxsita: " << maxsita << " sitaIncremental: " << sitaIncremental << endl;
	avgParamStream << "oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul	lamada	sita	gama	standdivation" << endl;
	avgParamStream << "avg: " << avgOldRS << "\t" << avg << "\t" << evaluationIndex[1] << "\t" << evaluationIndex[2] << "\t" << evaluationIndex[3] << "\t" << evaluationIndex[4] << "\t" << evaluationIndex[5] << "\t" << evaluationIndex[6] << "\t" << avgLamada << "\t" << avgSita << "\t" << "0" << "\t" << stadardivation << endl;

	cout << "stadardivation:	" << stadardivation << " avg: " << avg << endl;
	cout << "avgParam:" << avgLamada << "\t" << avgSita << "\t" << "0" << "\t" << " stadardivation:	" << stadardivation << endl << "avg: " << avgOldRS << "\t" << avg << "\t" << evaluationIndex[1] << "\t" << evaluationIndex[2] << "\t" << evaluationIndex[3] << "\t" << evaluationIndex[4] << "\t" << evaluationIndex[5] << "\t" << evaluationIndex[6] << endl;

	writefile(resultfile, avgParamStream.str());
	return stadardivation;
}

//三个参数的
double calculata9010RsThree()
{
	const int cishu = 10;
	bool hasOldPara = false;
	int counterX = 0, isBestTimes;
	double rsAndOthersArray[7][cishu];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double evaluationIndex[7];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double oldParaTimesRange = 3;

	double avg = 0.0, sum = 0.0, stadardivation = 0.0;
	double bestsita = 0.0, bestlamada = 0.0, bestgama = 0.0, bestRankingScore = 10.0, temprs = 0.0, avgLamada = 0.0, avgSita = 0.0, avgGama = 0.0, avgOldRS = 0.0, lastRankingScore = 10.0, lastLamada = 0, Lastsita = 0;
	double prametesArray[cishu][5];
	memset(prametesArray, 0, sizeof(prametesArray));
	memset(rsAndOthersArray, 0, sizeof(rsAndOthersArray));
	memset(evaluationIndex, 0, sizeof(evaluationIndex));
	oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem
	string resultfile = generateFilename();

	ifstream infile("oldparam.txt");
	if (!infile)
	{
		cout << "can not open this file oldparam.txt" << endl;
		hasOldPara = false;
	}
	else{
		getLastRuningParmeter();
		hasOldPara = true;
	}

	for (int times = 0; times<cishu; times++)
	{
		bool isWrongRange = false;
		lastRankingScore = 10.0;
		bestlamada = 0.0, bestsita = 0.0, bestgama = 0.0, bestRankingScore = 1.0, temprs = 0.0;

		if (hasOldPara)
		{
			/*for (int i = 0; i< 10;i++)
			{
			cout<<paraGroupVector[i].lamada<<endl;
			}*/

			if (paraGroupVector[times].isTheBestTimes > 0.3 && paraGroupVector[times].isTheBestTimes < 100)//如果对这个数据集来说参数已经是最优的，那么就直接跳过这个数据集，还是用以前的参数。<1000
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][2] = paraGroupVector[times].gama;;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes + 1;

				rsAndOthersArray[0][times] = paraGroupVector[times].rs;
				rsAndOthersArray[1][times] = paraGroupVector[times].lrs;
				//rsAndOthersArray[7][times] = bestRankingScore;

				//prametesArray[times][3] = rsAndOthersArray[0][times];
				cout << "RS on 9---1 dividing datab	" << rsAndOthersArray[0][times] << endl;
				sum += rsAndOthersArray[0][times];

				continue;
			}
			else
			{
				cout << "isTheBestTimes is wired" << paraGroupVector[times].isTheBestTimes << endl;
			}

			cout << "old lamada: " << paraGroupVector[times].lamada << "old sita: " << paraGroupVector[times].sita << "old gama: " << paraGroupVector[times].gama << endl;

			if (paraGroupVector[times].lamada != 0)
			{
				minlamada = paraGroupVector[times].lamada - oldParaTimesRange*lamadaIncremental;
				maxlamada = minlamada + 2 * oldParaTimesRange*lamadaIncremental;
				///lamadaIncremental = 0.01;
			}

			if (paraGroupVector[times].sita != 0)
			{
				minsita = paraGroupVector[times].sita - oldParaTimesRange*sitaIncremental;
				maxsita = minsita + 2 * oldParaTimesRange*sitaIncremental;
				//sitaIncremental = 0.01;
			}

			if (paraGroupVector[times].gama != 0)
			{
				mingama = paraGroupVector[times].gama - oldParaTimesRange*gamaIncremental;
				maxgama = mingama + 2 * oldParaTimesRange*gamaIncremental;
				//gamaIncremental = 0.01;
			}
		}

		cout << "minlamada: " << minlamada << " maxlamada: " << maxlamada << " lamadaIncremental: " << lamadaIncremental << endl;
		cout << "minsita: " << minsita << " maxsita: " << maxsita << " sitaIncremental: " << sitaIncremental << endl;
		cout << "mingama: " << mingama << " maxgama: " << maxgama << " gamaIncremental: " << gamaIncremental << endl;

		init(times);
		for (double lamada = minlamada; lamada <= maxlamada; lamada += lamadaIncremental)
		{
			if (isWrongRange && (minsita == maxsita) && (maxgama == mingama))
			{
				cout << "minsita==maxsita &&(maxgama==mingama)" << endl;
				break;
			}

			//reset isWrongRange
			isWrongRange = false;

			for (double sita = minsita; sita <= maxsita; sita += sitaIncremental)
			{
				if (isWrongRange && (maxgama == mingama))
				{
					cout << "mingama==maxgama" << endl;
					break;
				}
				for (double gama = mingama; gama <= maxgama; gama += gamaIncremental)
				{
					//---------------------------------------------
					runingMode = 1;//set testdate to train. 80%---10%
					//---------------------------------------------


					Basied_PD_RE_MD(lamada,sita,gama);
					//Heter_PD_RE_MD(lamada, sita, gama);

					temprs = getRankingScore();
					if ((temprs - bestRankingScore) <= wucha)
					{
						bestRankingScore = temprs;
						bestsita = sita;
						bestlamada = lamada;
						bestgama = gama;
					}

					cout << "lamada: " << lamada << "	sita: " << sita << " gama: " << gama << "	nowRS: " << temprs << "	lastRankingScore " << lastRankingScore << endl;

					if ((temprs>lastRankingScore) && (lastLamada == lamada) && (Lastsita == sita))
					{
						isWrongRange = true;
						//cout<<"lamada: "<<lamada<<"	sita: "<<sita<<" gama: "<<gama<<"	RS: "<<temprs<<endl;
						cout << "last is the best bestlamada: " << bestlamada << "	bestsita: " << bestsita << "	bestgama: " << bestgama << "	last:	" << lastRankingScore << "	temprs:	" << temprs << endl;
						cout << "---------------------------------------" << times << "---------------------------------------------" << endl;

						lastRankingScore = 10.0;
						break;
					}

					lastRankingScore = temprs;
					lastLamada = lamada;
					Lastsita = sita;

					/*if ((temprs - bestRankingScore) <= wucha)
					{
					bestRankingScore = temprs;
					bestsita = sita;
					bestlamada = lamada;
					bestgama = gama;
					}*/
				}
			}
		}

		stringstream errorMessagess;
		if (bestlamada >= maxlamada)
		{
			isWrongRange = true;
			errorMessagess << "need bigger lamada: bestlamada " << bestlamada << "	maxlamada: " << maxlamada << "	bestsita: " << bestsita << "	bestgama: " << bestgama << "	bestrs:	" << bestRankingScore << endl;
		}
		else if (bestlamada <= minlamada)
		{
			isWrongRange = true;
			errorMessagess << "need smaller lamada: bestlamada " << bestlamada << "	minlamada: " << minlamada << "	bestsita: " << bestsita << "	bestgama: " << bestgama << "	bestrs:	" << bestRankingScore << endl;
		}
		if (bestsita >= maxsita)
		{
			isWrongRange = true;
			errorMessagess << "need bigger sita: bestlamada " << bestlamada << "	bestsita: " << bestsita << "	maxsita: " << maxsita << "	bestgama: " << bestgama << "	bestrs:	" << bestRankingScore << endl;
		}
		else if (bestsita <= minsita)
		{
			isWrongRange = true;
			errorMessagess << "need smaller sita: bestlamada " << bestlamada << "	minsita: " << minsita << "	maxsita: " << maxsita << "	bestgama: " << bestgama << "	bestrs:	" << bestRankingScore << endl;
		}
		if (bestgama >= maxgama)
		{
			isWrongRange = true;
			errorMessagess << "need bigger gama: bestlamada " << bestlamada << "	bestsita: " << bestsita << "	bestgama: " << bestgama << "	maxgama: " << maxgama << "	bestrs:	" << bestRankingScore << endl;
		}
		else if (bestgama <= mingama)
		{
			isWrongRange = true;
			errorMessagess << "need smaller gama: bestlamada " << bestlamada << "	bestsita: " << bestsita << "	bestgama: " << bestgama << "	mingama: " << mingama << "	bestrs:	" << bestRankingScore << endl;
		}
		if (isWrongRange)
		{
			//cout<<"---------------------------------------isWrongRange-----------------------------------------------"<<endl;
			string tempcontents = errorMessagess.str();
			cout << tempcontents << endl;
			cout << "--------------------------------------isWrongRange finish this" << times << "------------------------" << endl;
		}
		errorMessagess.str("");
		//here to add some code to record the old rs
		if (hasOldPara && (paraGroupVector[times].oldRS>0 && paraGroupVector[times].oldRS < 1))
		{
			if ((paraGroupVector[times].oldRS - bestRankingScore) <= wucha) 
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][2] = paraGroupVector[times].gama;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes + 1;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				cout << "bestlamada is old one: " << prametesArray[times][0] << "	bestsita: " << prametesArray[times][1] << "\tbestgama: " << prametesArray[times][2] << "	bestrs:	" << prametesArray[times][3] << endl;
			}
			else if ((bestlamada = paraGroupVector[times].lamada) && (bestsita = paraGroupVector[times].sita) && (bestgama = paraGroupVector[times].gama)){
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][2] = paraGroupVector[times].gama;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes + 1;
				prametesArray[times][3] = bestRankingScore;
			}
			else{
				prametesArray[times][0] = bestlamada;
				prametesArray[times][1] = bestsita;
				prametesArray[times][2] = bestgama;
				prametesArray[times][3] = bestRankingScore;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes;
				cout << "bestlamada: " << bestlamada << "\tbestsita: " << bestsita << "\tbestgama: " << bestgama << "\tbestrs:	" << bestRankingScore << endl;
			}
		}else{
			prametesArray[times][0] = bestlamada;
			prametesArray[times][1] = bestsita;
			prametesArray[times][2] = bestgama;
			prametesArray[times][3] = bestRankingScore;
			prametesArray[times][4] = 0;
			cout << "bestlamada: " << bestlamada << "\tbestsita: " << bestsita << "\tbestgama: " << bestgama << "\tbestrs:	" << bestRankingScore << endl;
		}

		//---------------------------------------------

		rsAndOthersArray[0][times] = prametesArray[times][3];
		cout << "RS on 9---1 dividing datab	" << rsAndOthersArray[0][times] << endl;
		sum += rsAndOthersArray[0][times];
	}


	avg = sum / cishu;

	stringstream tempcontentstream;
	tempcontentstream << "t	lamada	sita	gama	oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul\tbesttimes" << endl;
	writefile(resultfile, tempcontentstream.str());
	tempcontentstream.str("");

	stringstream oldParaFile;
	stringstream oldParaLogFile;

	for (int times = 0; times<cishu; times++)
	{
		avgLamada += prametesArray[times][0];
		avgSita += prametesArray[times][1];
		avgGama += prametesArray[times][2];
		avgOldRS += prametesArray[times][3];

		evaluationIndex[0] += rsAndOthersArray[0][times];

		stadardivation += ((rsAndOthersArray[0][times] - avg)*(rsAndOthersArray[0][times] - avg));
		//		cout<<"net "<<times<<" lamada "<<prametesArray[times][0]<<" rs is: "<<prametesArray[times][3]<<endl;
		tempcontentstream << times << "\t" << prametesArray[times][0] << "\t" << prametesArray[times][1] << "\t" << prametesArray[times][2];
		//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;;
		tempcontentstream << "\t" << prametesArray[times][3] << "\t" << rsAndOthersArray[0][times] << "\t" << rsAndOthersArray[1][times] << "\t" << rsAndOthersArray[2][times] << "\t" << rsAndOthersArray[3][times] << "\t" << rsAndOthersArray[4][times] << "\t" << rsAndOthersArray[5][times] << "\t" << rsAndOthersArray[6][times] << "\t" << prametesArray[times][4];
		cout << tempcontentstream.str() << endl;
		writefile(resultfile, tempcontentstream.str());
		tempcontentstream.str("");

		oldParaLogFile << times << "\t" << prametesArray[times][0] << "\t" << prametesArray[times][1] << "\t" << prametesArray[times][2] << "\t" << prametesArray[times][3] << "\t" << prametesArray[times][4] << "\r\n";

		oldParaFile << times << "\t" << prametesArray[times][0] << "\t" << prametesArray[times][1] << "\t" << prametesArray[times][2] << "\t" << prametesArray[times][4];
		oldParaFile << "\t" << prametesArray[times][3] << "\t" << rsAndOthersArray[0][times] << "\t" << rsAndOthersArray[1][times] << "\t" << rsAndOthersArray[2][times] << "\t" << rsAndOthersArray[3][times] << "\t" << rsAndOthersArray[4][times] << "\t" << rsAndOthersArray[5][times] << "\t" << rsAndOthersArray[6][times];
		writefile("oldparam.txt", oldParaFile.str());
		oldParaFile.str("");
	}
	//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;
	removeAndWritefile("oldparam_log.txt", oldParaLogFile.str());
	oldParaLogFile.str("");

	avgLamada = avgLamada / cishu;
	avgGama = avgGama / cishu;
	avgSita = avgSita / cishu;
	avgOldRS = avgOldRS / cishu;

	evaluationIndex[0] = evaluationIndex[0] / cishu;
	evaluationIndex[1] = evaluationIndex[1] / cishu;

	stadardivation = sqrt(stadardivation / cishu);

	stringstream avgParamStream;

	avgParamStream << funcName << endl;
	avgParamStream << "minlamada: " << minlamada << " maxlamada: " << maxlamada << " lamadaIncremental: " << lamadaIncremental << endl;
	avgParamStream << "minsita: " << minsita << " maxsita: " << maxsita << " sitaIncremental: " << sitaIncremental << endl;
	avgParamStream << "mingama: " << mingama << " maxgama: " << maxgama << " gamaIncremental: " << gamaIncremental << endl;
	avgParamStream << "oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul	lamada	sita	gama	standdivation" << endl;
	avgParamStream << "avg: " << avgOldRS << "\t" << avg << "\t" << evaluationIndex[1] << "\t" << evaluationIndex[2] << "\t" << evaluationIndex[3] << "\t" << evaluationIndex[4] << "\t" << evaluationIndex[5] << "\t" << evaluationIndex[6] << "\t" << avgLamada << "\t" << avgSita << "\t" << avgGama << "\t" << stadardivation << endl;

	cout << "stadardivation:	" << stadardivation << " avg: " << avg << endl;
	cout << "avgParam:" << avgLamada << "\t" << avgSita << "\t" << avgGama << "\t" << " stadardivation:	" << stadardivation << endl << "avg: " << avgOldRS << "\t" << avg << "\t" << evaluationIndex[1] << "\t" << evaluationIndex[2] << "\t" << evaluationIndex[3] << "\t" << evaluationIndex[4] << "\t" << evaluationIndex[5] << "\t" << evaluationIndex[6] << endl;

	writefile(resultfile, avgParamStream.str());
	return stadardivation;
}


//一个参数的
double getStandardDevationOne()
{
	const int cishu = 10;
	bool hasOldPara = false;
	double oldParaTimesRange = 3;
	int counterX = 0,isBestTimes = 0;
	double rsAndOthersArray[7][cishu];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double evaluationIndex[7];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS

	double avg = 0.0,sum = 0.0, stadardivation=0.0;
	double bestlamada = 0.0 ,bestRankingScore = 10.0, temprs = 0.0,avgLamada = 0.0,avgOldRS = 0.0, lastRankingScore = 10.0;
	double prametesArray[cishu][5];
	memset(prametesArray,0,sizeof(prametesArray));
	//cout<<" First prametesArray[9][4] is : "<<prametesArray[cishu-1][4]<<endl;
	memset(rsAndOthersArray,0,sizeof(rsAndOthersArray));
	memset(evaluationIndex,0,sizeof(evaluationIndex));
	oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem
	string resultfile = generateFilename();
	
	ifstream infile("oldparam.txt");
	if(!infile)
	{
		cout << "can not open this file oldparam.txt"<< endl;
		hasOldPara = false;
	}else{
		getLastRuningParmeter();
		hasOldPara = true;
	}

	for (int times = 0; times<cishu; times++)
	{
		bool isWrongRange = false;
		lastRankingScore = 10.0;
		bestlamada = 0.0,bestRankingScore = 1.0, temprs = 0.0;

		if (hasOldPara)
		{
			/*for (int i = 0; i< 10;i++)
			{
			cout<<paraGroupVector[i].lamada<<endl;
			}*/
			if (paraGroupVector[times].isTheBestTimes > 0.3 && paraGroupVector[times].isTheBestTimes < 100)//如果对这个数据集来说参数已经是最优的，那么就直接跳过这个数据集，还是用以前的参数。<1000
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][2] = paraGroupVector[times].gama;;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes+1;

				rsAndOthersArray[0][times] = paraGroupVector[times].rs;
				rsAndOthersArray[1][times] = paraGroupVector[times].lrs;
				rsAndOthersArray[2][times] = paraGroupVector[times].pricis;
				rsAndOthersArray[3][times] = paraGroupVector[times].recall;
				rsAndOthersArray[4][times] = paraGroupVector[times].intrSim;
				rsAndOthersArray[5][times] = paraGroupVector[times].hamdis;
				rsAndOthersArray[6][times] = paraGroupVector[times].Popul;
				//rsAndOthersArray[7][times] = bestRankingScore;

				//prametesArray[times][3] = rsAndOthersArray[0][times];
				cout<<"RS on 9---1 dividing datab	"<<rsAndOthersArray[0][times]<<endl;
				sum+=rsAndOthersArray[0][times];

				continue;
			}else
			{
				cout<<"isTheBestTimes is wired"<<paraGroupVector[times].isTheBestTimes<<endl;
			}
			cout<<"old lamada: "<<paraGroupVector[times].lamada<<endl;
			if (paraGroupVector[times].lamada!=0)
			{
				minlamada = paraGroupVector[times].lamada-oldParaTimesRange*lamadaIncremental;
				maxlamada = minlamada+2*oldParaTimesRange*lamadaIncremental;
				//lamadaIncremental = 0.01;
			}
		}

		//cout<<"Second prametesArray[times][4] is : "<<prametesArray[times][4]<<endl;
		cout<<"minlamada: "<<minlamada<<" maxlamada: "<<maxlamada<<" lamadaIncremental: "<<lamadaIncremental<<endl;

		init(times);
		for (double lamada = minlamada; lamada<=maxlamada; lamada+=lamadaIncremental)
		{			
			//---------------------------------------------
			runingMode = 0;//set testdate to train. 80%---10%
			//---------------------------------------------

			//counterX++;
			//WHCMatrix(lamada,trainingSet);
			//IHC(lamada,trainingSet);
			//IMD(lamada,trainingSet);
			//IHCMatrix(lamada,trainingSet);
			//HHCMatrix(lamada,trainingSet);
			//HHC(lamada,trainingSet);
			//hybirdHAndPNonLinaer(lamada,trainingSet);
			//RE_NBI(lamada,trainingSet);
			//Heter_NBI(lamada,trainingSet);
			PD(lamada,trainingSet);
			//URA_NBI(lamada,trainingSet);
			//Biased_Heat(lamada,trainingSet);
			//Cold_Start(lamada,trainingSet);
			//Cold_StartMatrix(lamada,trainingSet);
			//caclSparseNetwork(lamada,trainingSet);
			//NCF(lamada,trainingSet);
			//NCFNew(lamada,trainingSet);
			//MCF(lamada,trainingSet);
			//ProbS(lamada,trainingSet);
			//HeatS(trainingSet);
			//B_Rank(trainingSet);
			//ProbS(1,trainingSet);
			//temprs = RankingScoreNotCollect(learningSet);
			//Heter_PD(lamada,sita,trainingSet);
			//SPD(lamada,sita,trainingSet);
			//Basied_PD_RE_MD(lamada,sita,gama,trainingSet);
			/*Heter_PD_RE_MD(lamada,sita,gama);*/
			temprs = getRankingScore();
			cout<<"lamada: "<<lamada<<"	sita: "<<0<<" gama: "<<0<<"	nowRS: "<<temprs<<"	lastRankingScore "<<lastRankingScore<<endl;
			cout<<"local is : "<<getLocalRankingScore()<<endl;
			if ((temprs - bestRankingScore) <= wucha)
			{
				bestRankingScore = temprs;
				bestlamada = lamada;		
			}

			if (temprs>lastRankingScore)
			{
				isWrongRange = true;
				//cout<<"lamada: "<<lamada<<"	sita: "<<0<<" gama: "<<0<<"	RS: "<<temprs<<endl;
				cout<<"last is the best bestlamada: "<<bestlamada<<"	temprs:	"<<temprs<<endl;
				cout<<"---------------------------------------"<<times<<"---------------------------------------------"<<endl;
						
				lastRankingScore = 10.0;
				break;
			}

			lastRankingScore = temprs;			
		}

		stringstream errorMessagess;
		if (bestlamada >= maxlamada)
		{
			isWrongRange = true;
			errorMessagess<<"need bigger lamada: bestlamada "<<bestlamada<<"	maxlamada: "<<maxlamada<<"	bestrs:	"<<bestRankingScore<<endl;
		}else if (bestlamada <= minlamada)
		{
			isWrongRange = true;
			errorMessagess<<"need smaller lamada: bestlamada "<<bestlamada<<"	minlamada: "<<minlamada<<"	bestrs:	"<<bestRankingScore<<endl;
		}		
		if (isWrongRange)
		{
			//cout<<"---------------------------------------isWrongRange-----------------------------------------------"<<endl;
			string tempcontents = errorMessagess.str();
			cout<<tempcontents<<endl;
			cout<<"--------------------------------------isWrongRange finish this"<<times<<"------------------------"<<endl;
		}

		//here to add some code to record the old rs
		if (hasOldPara)
		{
			if ((bestlamada == paraGroupVector[times].lamada) || ((paraGroupVector[times].oldRS - bestRankingScore) <= wucha && paraGroupVector[times].oldRS>0))
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes+1;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				cout<<"bestlamada is old one: "<<prametesArray[times][0]<<"	bestrs:	"<<prametesArray[times][3]<<endl;
			}else{
				prametesArray[times][0] = bestlamada;
				prametesArray[times][3] = bestRankingScore;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes;
				cout<<"bestlamada: "<<prametesArray[times][0]<<"	bestrs:	"<<prametesArray[times][3]<<endl;
			}
		}else{
			prametesArray[times][0] = bestlamada;
			prametesArray[times][3] = bestRankingScore;
			prametesArray[times][4] = 0;
			cout<<"bestlamada: "<<prametesArray[times][0]<<"	bestrs:	"<<prametesArray[times][3]<<endl;
		}
		//---------------------------------------------
		runingMode = 1;//set testdate to testSet. 90%---10%
		//---------------------------------------------
		//WHCMatrix(prametesArray[times][0]);
		//IHC(prametesArray[times][0]);
		//IMD(prametesArray[times][0]);
		//IHCMatrix(prametesArray[times][0]);
		//HHCMatrix(prametesArray[times][0]);
		//HHC(prametesArray[times][0]);
		//hybirdHAndPNonLinaer(prametesArray[times][0]);
		//RE_NBI(prametesArray[times][0]);
		//Heter_NBI(prametesArray[times][0]);
		PD(prametesArray[times][0]);
		//URA_NBI(prametesArray[times][0]);
		//Biased_Heat(prametesArray[times][0]);
		//Cold_Start(prametesArray[times][0]);
		//Cold_StartMatrix(prametesArray[times][0]);
		//caclSparseNetwork(prametesArray[times][0]);
		//NCF(prametesArray[times][0]);
		//NCFNew(prametesArray[times][0]);
		//MCF();
		//ProbS(prametesArray[times][0]);
		//HeatS();
		//B_Rank();
		//ProbS(1);
		//Heter_PD(prametesArray[times][0],prametesArray[times][1]);
		//SPD(prametesArray[times][0],prametesArray[times][1]);
		//Basied_PD_RE_MD(prametesArray[times][0],prametesArray[times][1],prametesArray[times][2],trainingSet);
		/*Heter_PD_RE_MD(prametesArray[times][0],prametesArray[times][1],prametesArray[times][2],trainingSet);*/
		//rsArray[times] = RankingScoreNotCollect();

		rsAndOthersArray[0][times] = getRankingScore();
		rsAndOthersArray[1][times] = getLocalRankingScore();
		rsAndOthersArray[2][times] = Precision();
		rsAndOthersArray[3][times] = Recall();
		rsAndOthersArray[4][times] = IntraSimilarity();
		rsAndOthersArray[5][times] = HammingDistance();
		rsAndOthersArray[6][times] = Popularity();
		//rsAndOthersArray[7][times] = bestRankingScore;

		//prametesArray[times][3] = rsAndOthersArray[0][times];
		cout<<"RS on 9---1 dividing datab	"<<rsAndOthersArray[0][times]<<endl;
		sum+=rsAndOthersArray[0][times];
	}
	avg = sum/cishu;

	stringstream tempcontentstream;
	tempcontentstream<<"t	lamada	sita	gama	oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul"<<endl;
	writefile(resultfile,tempcontentstream.str());
	tempcontentstream.str("");
	//cout<<"Thrid  prametesArray[times][4] is : "<<prametesArray[cishu-1][4]<<endl;
	stringstream oldParaFile;
	stringstream oldParaLogFile;

	for (int times = 0; times<cishu; times++)
	{
		avgLamada += prametesArray[times][0];
		avgOldRS+= prametesArray[times][3];

		evaluationIndex[0] += rsAndOthersArray[0][times];
		evaluationIndex[1] += rsAndOthersArray[1][times];
		evaluationIndex[2] += rsAndOthersArray[2][times];
		evaluationIndex[3] += rsAndOthersArray[3][times];
		evaluationIndex[4] += rsAndOthersArray[4][times];
		evaluationIndex[5] += rsAndOthersArray[5][times];
		evaluationIndex[6] += rsAndOthersArray[6][times];

		stadardivation+=((rsAndOthersArray[0][times]-avg)*(rsAndOthersArray[0][times]-avg));
//		cout<<"net "<<times<<" lamada "<<prametesArray[times][0]<<" rs is: "<<prametesArray[times][3]<<endl;
		tempcontentstream<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2];
		//oldParaFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\t"<<isBestTimes<<"\t"<<prametesArray[times][3]<<"\r\n";
		//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;;
		tempcontentstream<<"\t"<<prametesArray[times][3]<<"\t"<<rsAndOthersArray[0][times]<<"\t"<<rsAndOthersArray[1][times]<<"\t"<<rsAndOthersArray[2][times]<<"\t"<<rsAndOthersArray[3][times]<<"\t"<<rsAndOthersArray[4][times]<<"\t"<<rsAndOthersArray[5][times]<<"\t"<<rsAndOthersArray[6][times];
		cout<<tempcontentstream.str()<<endl;
		writefile(resultfile,tempcontentstream.str());
		tempcontentstream.str("");

		oldParaLogFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\t"<<prametesArray[times][3]<<"\t"<<prametesArray[times][4]<<"\r\n";
		oldParaFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\t"<<prametesArray[times][4];
		oldParaFile<<"\t"<<prametesArray[times][3]<<"\t"<<rsAndOthersArray[0][times]<<"\t"<<rsAndOthersArray[1][times]<<"\t"<<rsAndOthersArray[2][times]<<"\t"<<rsAndOthersArray[3][times]<<"\t"<<rsAndOthersArray[4][times]<<"\t"<<rsAndOthersArray[5][times]<<"\t"<<rsAndOthersArray[6][times];
		writefile("oldparam.txt",oldParaFile.str());
		oldParaFile.str("");
		//cout<<"Fourth  prametesArray[times][4] is : "<<prametesArray[times][4]<<endl;

	}
	//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;
	removeAndWritefile("oldparam_log.txt",oldParaLogFile.str());
	oldParaLogFile.str("");

	avgLamada = avgLamada/cishu;
	avgOldRS = avgOldRS/cishu;

	evaluationIndex[0] = evaluationIndex[0]/cishu;
	evaluationIndex[1] = evaluationIndex[1]/cishu;
	evaluationIndex[2] = evaluationIndex[2]/cishu;
	evaluationIndex[3] = evaluationIndex[3]/cishu;
	evaluationIndex[4] = evaluationIndex[4]/cishu;
	evaluationIndex[5] = evaluationIndex[5]/cishu;
	evaluationIndex[6] = evaluationIndex[6]/cishu;

	stadardivation = sqrt(stadardivation/cishu);

	stringstream avgParamStream;

	avgParamStream<<funcName<<endl;
	avgParamStream<<"minlamada: "<<minlamada<<" maxlamada: "<<maxlamada<<" lamadaIncremental: "<<lamadaIncremental<<endl;
	avgParamStream<<"oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul	lamada	sita	gama	standdivation"<<endl;
	avgParamStream<<"avg: "<<avgOldRS<<"\t"<<avg<<"\t"<<evaluationIndex[1]<<"\t"<<evaluationIndex[2]<<"\t"<<evaluationIndex[3]<<"\t"<<evaluationIndex[4]<<"\t"<<evaluationIndex[5]<<"\t"<<evaluationIndex[6]<<"\t"<<avgLamada<<"\t"<<"0"<<"\t"<<"0"<<"\t"<<stadardivation <<endl;
	
	cout<<"stadardivation:	"<<stadardivation <<" avg: "<<avg<<endl;
	cout<<"avgParam:"<<avgLamada<<"\t"<<"0"<<"\t"<<"0"<<"\t"<<" stadardivation:	"<<stadardivation <<endl<<"avg: "<<avgOldRS<<"\t"<<avg<<"\t"<<evaluationIndex[1]<<"\t"<<evaluationIndex[2]<<"\t"<<evaluationIndex[3]<<"\t"<<evaluationIndex[4]<<"\t"<<evaluationIndex[5]<<"\t"<<evaluationIndex[6]<<endl;

	writefile(resultfile,avgParamStream.str());
	return stadardivation;
}

//两个参数的
double getStandardDevationTwo()
{
	const int cishu = 10;
	bool hasOldPara = false;
	int counterX = 0,isBestTimes = 0;
	double rsAndOthersArray[7][cishu];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double evaluationIndex[7];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double oldParaTimesRange = 5;
	double avg = 0.0,sum = 0.0, stadardivation=0.0;
	double bestsita = 0.0, bestlamada = 0.0,bestRankingScore = 10.0, temprs = 0.0,avgLamada = 0.0,avgSita = 0.0,avgOldRS = 0.0, lastRankingScore = 10.0,lastLamada = 0;
	double prametesArray[cishu][5];
	memset(evaluationIndex,0,sizeof(evaluationIndex));
	memset(prametesArray,0,sizeof(prametesArray));
	memset(rsAndOthersArray,0,sizeof(rsAndOthersArray));
	oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem
	string resultfile = generateFilename();	

	ifstream infile("oldparam.txt");	
	if(!infile)
	{
		cout << "can not open this file oldparam.txt"<< endl;
		hasOldPara = false;
	}else{
		getLastRuningParmeter();
		hasOldPara = true;
	}

	for (int times = 0; times<cishu; times++)
	{
		bool isWrongRange = false;
		lastRankingScore = 10.0;
		bestlamada = 0.0, bestsita = 0.0, bestRankingScore = 1.0, temprs = 0.0;

		if (hasOldPara)
		{
			if (paraGroupVector[times].isTheBestTimes > 0.3 && paraGroupVector[times].isTheBestTimes < 100)//如果对这个数据集来说参数已经是最优的，那么就直接跳过这个数据集，还是用以前的参数。<1000
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][2] = paraGroupVector[times].gama;;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes+1;

				rsAndOthersArray[0][times] = paraGroupVector[times].rs;
				rsAndOthersArray[1][times] = paraGroupVector[times].lrs;
				rsAndOthersArray[2][times] = paraGroupVector[times].pricis;
				rsAndOthersArray[3][times] = paraGroupVector[times].recall;
				rsAndOthersArray[4][times] = paraGroupVector[times].intrSim;
				rsAndOthersArray[5][times] = paraGroupVector[times].hamdis;
				rsAndOthersArray[6][times] = paraGroupVector[times].Popul;
				//rsAndOthersArray[7][times] = bestRankingScore;

				//prametesArray[times][3] = rsAndOthersArray[0][times];
				cout<<"RS on 9---1 dividing datab	"<<rsAndOthersArray[0][times]<<endl;
				sum+=rsAndOthersArray[0][times];

				continue;
			}else
			{
				cout<<"isTheBestTimes is wired"<<paraGroupVector[times].isTheBestTimes<<endl;
			}
			/*for (int i = 0; i< 10;i++)
			{
			cout<<paraGroupVector[i].lamada<<endl;
			}*/
			cout<<"old lamada: "<<paraGroupVector[times].lamada<<"old sita: "<<paraGroupVector[times].sita<<endl;

			if (paraGroupVector[times].lamada!=0)
			{
				minlamada = paraGroupVector[times].lamada-oldParaTimesRange*lamadaIncremental;
				maxlamada = minlamada+2*oldParaTimesRange*lamadaIncremental;
				//lamadaIncremental = 0.01;
			}

			if (paraGroupVector[times].sita!=0)
			{
				minsita = paraGroupVector[times].sita-oldParaTimesRange*sitaIncremental;
				maxsita = minsita+2*oldParaTimesRange*sitaIncremental;
				//sitaIncremental = 0.01;
			}
		}

		cout<<"minlamada: "<<minlamada<<" maxlamada: "<<maxlamada<<" lamadaIncremental: "<<lamadaIncremental<<endl;
		cout<<"minsita: "<<minsita<<" maxsita: "<<maxsita<<" sitaIncremental: "<<sitaIncremental<<endl;

		init(times);
		for (double lamada = minlamada; lamada<=maxlamada; lamada+=lamadaIncremental)
		{
			if (isWrongRange&&(minsita==maxsita)&&(maxgama==mingama))
			{
				cout<<"minsita==maxsita &&(maxgama==mingama)"<<endl;
				break;
			}
			//reset isWrongRange
			isWrongRange = false;

			for (double sita = minsita; sita<=maxsita; sita+=sitaIncremental)
			{
				
				//---------------------------------------------
				runingMode = 0;//set testdate to train. 80%---10%
				//---------------------------------------------

				//counterX++;

				//Heter_PD(lamada,sita,trainingSet);
				SPD(lamada,sita,trainingSet);

				temprs = getRankingScore();

				if ((temprs - bestRankingScore) <= wucha)
				{
					bestRankingScore = temprs;
					bestsita = sita;
					bestlamada = lamada;			
				}
				cout<<"lamada: "<<lamada<<"	sita: "<<sita<<"	nowRS: "<<temprs<<"	lastRankingScore "<<lastRankingScore<<endl;

				if ((temprs>lastRankingScore) && (lastLamada ==lamada))
				{
					isWrongRange = true;
					//cout<<"lamada: "<<lamada<<"	sita: "<<sita<<"	RS: "<<temprs<<endl;
					cout<<"last is the best bestlamada: "<<bestlamada<<"	bestsita: "<<bestsita<<"	temprs:	"<<temprs<<endl;
					cout<<"---------------------------------------"<<times<<"---------------------------------------------"<<endl;

					lastRankingScore = 10.0;
					break;
				}

				lastRankingScore = temprs;	
				lastLamada = lamada;

				/*if ((temprs - bestRankingScore) <= wucha)
				{
				bestRankingScore = temprs;
				bestsita = sita;
				bestlamada = lamada;			
				}*/
			}
		}

		stringstream errorMessagess;
		if (bestlamada >= maxlamada)
		{
			isWrongRange = true;
			errorMessagess<<"need bigger lamada: bestlamada "<<bestlamada<<"	maxlamada: "<<maxlamada<<"	bestsita: "<<bestsita<<"	bestrs:	"<<bestRankingScore<<endl;
		}else if (bestlamada <= minlamada)
		{
			isWrongRange = true;
			errorMessagess<<"need smaller lamada: bestlamada "<<bestlamada<<"	minlamada: "<<minlamada<<"	bestsita: "<<bestsita<<"	bestrs:	"<<bestRankingScore<<endl;
		}
		if (bestsita >= maxsita)
		{
			isWrongRange = true;
			errorMessagess<<"need bigger sita: bestlamada "<<bestlamada<<"	bestsita: "<<bestsita<<"	maxsita: "<<maxsita<<"	bestrs:	"<<bestRankingScore<<endl;
		}else if (bestsita <= minsita)
		{
			isWrongRange = true;
			errorMessagess<<"need smaller sita: bestlamada "<<bestlamada<<"	minsita: "<<minsita<<"	maxsita: "<<maxsita<<"	bestrs:	"<<bestRankingScore<<endl;
		}		
		if (isWrongRange)
		{
			//cout<<"---------------------------------------isWrongRange-----------------------------------------------"<<endl;
			string tempcontents = errorMessagess.str();
			cout<<tempcontents<<endl;
			cout<<"--------------------------------------isWrongRange finish this"<<times<<"------------------------"<<endl;
		}

		//here to add some code to record the old rs
		if (hasOldPara)
		{
			if (((bestlamada == paraGroupVector[times].lamada) && (bestsita == paraGroupVector[times].sita)) || ((paraGroupVector[times].oldRS - bestRankingScore) <= wucha && paraGroupVector[times].oldRS>0))
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][3] = paraGroupVector[times].oldRS;	
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes+1;					
				cout<<"bestlamada is old one: "<<prametesArray[times][0]<<"	bestsita: "<<prametesArray[times][1]<<"	bestrs:	"<<prametesArray[times][3]<<endl;
			}else{
				prametesArray[times][0] = bestlamada;
				prametesArray[times][1] = bestsita;
				prametesArray[times][3] = bestRankingScore;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes;
				cout<<"bestlamada: "<<bestlamada<<"	bestsita: "<<bestsita<<"	bestrs:	"<<bestRankingScore<<endl;
			}
		}else{
			prametesArray[times][0] = bestlamada;
			prametesArray[times][1] = bestsita;
			prametesArray[times][3] = bestRankingScore;
			prametesArray[times][4] = 0;
			cout<<"bestlamada: "<<bestlamada<<"	bestsita: "<<bestsita<<"	bestrs:	"<<bestRankingScore<<endl;
		}	

		//---------------------------------------------
		runingMode = 1;//set testdate to testSet. 90%---10%
		//---------------------------------------------

		//Heter_PD(prametesArray[times][0],prametesArray[times][1]);
		SPD(prametesArray[times][0],prametesArray[times][1]);
		//rsArray[times] = RankingScoreNotCollect();

		rsAndOthersArray[0][times] = getRankingScore();
		rsAndOthersArray[1][times] = getLocalRankingScore();
		rsAndOthersArray[2][times] = Precision();
		rsAndOthersArray[3][times] = Recall();
		rsAndOthersArray[4][times] = IntraSimilarity();
		rsAndOthersArray[5][times] = HammingDistance();
		rsAndOthersArray[6][times] = Popularity();
		//rsAndOthersArray[7][times] = bestRankingScore;

		//prametesArray[times][3] = rsAndOthersArray[0][times];
		cout<<"RS on 9---1 dividing datab	"<<rsAndOthersArray[0][times]<<endl;
		sum+=rsAndOthersArray[0][times];
	}
	avg = sum/cishu;

	stringstream tempcontentstream;
	tempcontentstream<<"t	lamada	sita	gama	oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul"<<endl;
	writefile(resultfile,tempcontentstream.str());
	tempcontentstream.str("");

	stringstream oldParaFile;
	stringstream oldParaLogFile;

	for (int times = 0; times<cishu; times++)
	{
		avgLamada += prametesArray[times][0];
		avgSita += prametesArray[times][1];
		avgOldRS+= prametesArray[times][3];

		evaluationIndex[0] += rsAndOthersArray[0][times];
		evaluationIndex[1] += rsAndOthersArray[1][times];
		evaluationIndex[2] += rsAndOthersArray[2][times];
		evaluationIndex[3] += rsAndOthersArray[3][times];
		evaluationIndex[4] += rsAndOthersArray[4][times];
		evaluationIndex[5] += rsAndOthersArray[5][times];
		evaluationIndex[6] += rsAndOthersArray[6][times];

		stadardivation+=((rsAndOthersArray[0][times]-avg)*(rsAndOthersArray[0][times]-avg));
		//		cout<<"net "<<times<<" lamada "<<prametesArray[times][0]<<" rs is: "<<prametesArray[times][3]<<endl;
		tempcontentstream<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2];
		//oldParaFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\t"<<"\t"<<isBestTimes<<prametesArray[times][3]<<"\r\n";
		//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;;
		tempcontentstream<<"\t"<<prametesArray[times][3]<<"\t"<<rsAndOthersArray[0][times]<<"\t"<<rsAndOthersArray[1][times]<<"\t"<<rsAndOthersArray[2][times]<<"\t"<<rsAndOthersArray[3][times]<<"\t"<<rsAndOthersArray[4][times]<<"\t"<<rsAndOthersArray[5][times]<<"\t"<<rsAndOthersArray[6][times];
		cout<<tempcontentstream.str()<<endl;
		writefile(resultfile,tempcontentstream.str());
		tempcontentstream.str("");

		oldParaLogFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\t"<<prametesArray[times][3]<<"\t"<<prametesArray[times][4]<<"\r\n";
		oldParaFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\t"<<prametesArray[times][4];
		oldParaFile<<"\t"<<prametesArray[times][3]<<"\t"<<rsAndOthersArray[0][times]<<"\t"<<rsAndOthersArray[1][times]<<"\t"<<rsAndOthersArray[2][times]<<"\t"<<rsAndOthersArray[3][times]<<"\t"<<rsAndOthersArray[4][times]<<"\t"<<rsAndOthersArray[5][times]<<"\t"<<rsAndOthersArray[6][times];
		writefile("oldparam.txt",oldParaFile.str());
		oldParaFile.str("");

	}
	//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;
	removeAndWritefile("oldparam_log.txt",oldParaLogFile.str());
	oldParaLogFile.str("");

	avgLamada = avgLamada/cishu;
	avgSita = avgSita/cishu;
	avgOldRS = avgOldRS/cishu;

	evaluationIndex[0] = evaluationIndex[0]/cishu;
	evaluationIndex[1] = evaluationIndex[1]/cishu;
	evaluationIndex[2] = evaluationIndex[2]/cishu;
	evaluationIndex[3] = evaluationIndex[3]/cishu;
	evaluationIndex[4] = evaluationIndex[4]/cishu;
	evaluationIndex[5] = evaluationIndex[5]/cishu;
	evaluationIndex[6] = evaluationIndex[6]/cishu;

	stadardivation = sqrt(stadardivation/cishu);

	stringstream avgParamStream;

	avgParamStream<<funcName<<endl;
	avgParamStream<<"minlamada: "<<minlamada<<" maxlamada: "<<maxlamada<<" lamadaIncremental: "<<lamadaIncremental<<endl;
	avgParamStream<<"minsita: "<<minsita<<" maxsita: "<<maxsita<<" sitaIncremental: "<<sitaIncremental<<endl;
	avgParamStream<<"oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul	lamada	sita	gama	standdivation"<<endl;
	avgParamStream<<"avg: "<<avgOldRS<<"\t"<<avg<<"\t"<<evaluationIndex[1]<<"\t"<<evaluationIndex[2]<<"\t"<<evaluationIndex[3]<<"\t"<<evaluationIndex[4]<<"\t"<<evaluationIndex[5]<<"\t"<<evaluationIndex[6]<<"\t"<<avgLamada<<"\t"<<avgSita<<"\t"<<"0"<<"\t"<<stadardivation <<endl;

	cout<<"stadardivation:	"<<stadardivation <<" avg: "<<avg<<endl;
	cout<<"avgParam:"<<avgLamada<<"\t"<<avgSita<<"\t"<<"0"<<"\t"<<" stadardivation:	"<<stadardivation <<endl<<"avg: "<<avgOldRS<<"\t"<<avg<<"\t"<<evaluationIndex[1]<<"\t"<<evaluationIndex[2]<<"\t"<<evaluationIndex[3]<<"\t"<<evaluationIndex[4]<<"\t"<<evaluationIndex[5]<<"\t"<<evaluationIndex[6]<<endl;

	writefile(resultfile,avgParamStream.str());
	return stadardivation;
}

//三个参数的
double getStandardDevationThree()
{
	const int cishu = 10;
	bool hasOldPara = false;
	int counterX = 0,isBestTimes;
	double rsAndOthersArray[7][cishu];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double evaluationIndex[7];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double oldParaTimesRange = 3;

	double avg = 0.0,sum = 0.0, stadardivation=0.0;
	double bestsita = 0.0, bestlamada = 0.0,bestgama = 0.0, bestRankingScore = 10.0, temprs = 0.0,avgLamada = 0.0,avgSita = 0.0,avgGama = 0.0,avgOldRS = 0.0, lastRankingScore = 10.0,lastLamada = 0,Lastsita = 0;
	double prametesArray[cishu][5];////记录其他一些观测量的。0是lamada，1:sita	2: gama	3: oldrs 4: besttimes
	memset(prametesArray,0,sizeof(prametesArray));
	memset(rsAndOthersArray,0,sizeof(rsAndOthersArray));
	memset(evaluationIndex,0,sizeof(evaluationIndex));
	oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem
	string resultfile = generateFilename();	

	ifstream infile("oldparam.txt");	
	if(!infile)
	{
		cout << "can not open this file oldparam.txt"<< endl;
		hasOldPara = false;
	}else{
		getLastRuningParmeter();
		hasOldPara = true;
	}

	for (int times = 0; times<cishu; times++)
	{
		bool isWrongRange = false;
		lastRankingScore = 10.0;
		bestlamada = 0.0, bestsita = 0.0, bestgama = 0.0,bestRankingScore = 1.0, temprs = 0.0;

		if (hasOldPara)
		{
			/*for (int i = 0; i< 10;i++)
			{
			cout<<paraGroupVector[i].lamada<<endl;
			}*/

			if (paraGroupVector[times].isTheBestTimes > 0.3 && paraGroupVector[times].isTheBestTimes < 100)//如果对这个数据集来说参数已经是最优的，那么就直接跳过这个数据集，还是用以前的参数。<1000
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][2] = paraGroupVector[times].gama;;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes+1;

				rsAndOthersArray[0][times] = paraGroupVector[times].rs;
				rsAndOthersArray[1][times] = paraGroupVector[times].lrs;
				rsAndOthersArray[2][times] = paraGroupVector[times].pricis;
				rsAndOthersArray[3][times] = paraGroupVector[times].recall;
				rsAndOthersArray[4][times] = paraGroupVector[times].intrSim;
				rsAndOthersArray[5][times] = paraGroupVector[times].hamdis;
				rsAndOthersArray[6][times] = paraGroupVector[times].Popul;
				//rsAndOthersArray[7][times] = bestRankingScore;

				//prametesArray[times][3] = rsAndOthersArray[0][times];
				cout<<"RS on 9---1 dividing datab	"<<rsAndOthersArray[0][times]<<endl;
				sum+=rsAndOthersArray[0][times];

				continue;
			}else
			{
				cout<<"isTheBestTimes is wired"<<paraGroupVector[times].isTheBestTimes<<endl;
			}

			cout<<"old lamada: "<<paraGroupVector[times].lamada<<"old sita: "<<paraGroupVector[times].sita<<"old gama: "<<paraGroupVector[times].gama<<endl;

			if (paraGroupVector[times].lamada!=0)
			{
				minlamada = paraGroupVector[times].lamada-oldParaTimesRange*lamadaIncremental;
				maxlamada = minlamada+2*oldParaTimesRange*lamadaIncremental;
				///lamadaIncremental = 0.01;
			}

			if (paraGroupVector[times].sita!=0)
			{
				minsita = paraGroupVector[times].sita-oldParaTimesRange*sitaIncremental;
				maxsita = minsita+2*oldParaTimesRange*sitaIncremental;
				//sitaIncremental = 0.01;
			}

			if (paraGroupVector[times].gama!=0)
			{
				mingama = paraGroupVector[times].gama-oldParaTimesRange*gamaIncremental;
				maxgama = mingama+2*oldParaTimesRange*gamaIncremental;
				//gamaIncremental = 0.01;
			}
		}

		cout<<"minlamada: "<<minlamada<<" maxlamada: "<<maxlamada<<" lamadaIncremental: "<<lamadaIncremental<<endl;
		cout<<"minsita: "<<minsita<<" maxsita: "<<maxsita<<" sitaIncremental: "<<sitaIncremental<<endl;
		cout<<"mingama: "<<mingama<<" maxgama: "<<maxgama<<" gamaIncremental: "<<gamaIncremental<<endl;

		init(times);
		for (double lamada = minlamada; lamada<=maxlamada; lamada+=lamadaIncremental)
		{
			if (isWrongRange&&(minsita==maxsita)&&(maxgama==mingama))
			{
				cout<<"minsita==maxsita &&(maxgama==mingama)"<<endl;
				break;
			}

			//reset isWrongRange
			isWrongRange = false;

			for (double sita = minsita; sita<=maxsita; sita+=sitaIncremental)
			{
				if (isWrongRange&&(maxgama==mingama))
				{
					cout<<"mingama==maxgama"<<endl;
					break;
				}
				for (double gama = mingama; gama<=maxgama; gama+=gamaIncremental)
				{
					//---------------------------------------------
					runingMode = 0;//set testdate to train. 80%---10%
					//---------------------------------------------

					
					//Basied_PD_RE_MD(lamada,sita,gama,trainingSet);


					Heter_PD_RE_MD(lamada,sita,gama,trainingSet);
					
					temprs = getRankingScore();
					if ((temprs - bestRankingScore) <= wucha)
					{
						bestRankingScore = temprs;
						bestsita = sita;
						bestlamada = lamada;
						bestgama = gama;
					}

					cout<<"lamada: "<<lamada<<"	sita: "<<sita<<" gama: "<<gama<<"	nowRS: "<<temprs<<"	lastRankingScore "<<lastRankingScore<<endl;

					if ((temprs>lastRankingScore) && (lastLamada == lamada) && (Lastsita == sita))
					{
						isWrongRange = true;
						//cout<<"lamada: "<<lamada<<"	sita: "<<sita<<" gama: "<<gama<<"	RS: "<<temprs<<endl;
						cout<<"last is the best bestlamada: "<<bestlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	last:	"<<lastRankingScore<<"	temprs:	"<<temprs<<endl;
						cout<<"---------------------------------------"<<times<<"---------------------------------------------"<<endl;

						lastRankingScore = 10.0;
						break;
					}

					lastRankingScore = temprs;
					lastLamada = lamada;
					Lastsita = sita;

					/*if ((temprs - bestRankingScore) <= wucha)
					{
						bestRankingScore = temprs;
						bestsita = sita;
						bestlamada = lamada;
						bestgama = gama;					
					}*/
				}
			}
		}

		stringstream errorMessagess;
		if (bestlamada >= maxlamada)
		{
			isWrongRange = true;
			errorMessagess<<"need bigger lamada: bestlamada "<<bestlamada<<"	maxlamada: "<<maxlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	bestrs:	"<<bestRankingScore<<endl;
		}else if (bestlamada <= minlamada)
		{
			isWrongRange = true;
			errorMessagess<<"need smaller lamada: bestlamada "<<bestlamada<<"	minlamada: "<<minlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	bestrs:	"<<bestRankingScore<<endl;
		}
		if (bestsita >= maxsita)
		{
			isWrongRange = true;
			errorMessagess<<"need bigger sita: bestlamada "<<bestlamada<<"	bestsita: "<<bestsita<<"	maxsita: "<<maxsita<<"	bestgama: "<<bestgama<<"	bestrs:	"<<bestRankingScore<<endl;
		}else if (bestsita <= minsita)
		{
			isWrongRange = true;
			errorMessagess<<"need smaller sita: bestlamada "<<bestlamada<<"	minsita: "<<minsita<<"	maxsita: "<<maxsita<<"	bestgama: "<<bestgama<<"	bestrs:	"<<bestRankingScore<<endl;
		}
		if (bestgama >= maxgama)
		{
			isWrongRange = true;
			errorMessagess<<"need bigger gama: bestlamada "<<bestlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	maxgama: "<<maxgama<<"	bestrs:	"<<bestRankingScore<<endl;
		}else if (bestgama <= mingama)
		{
			isWrongRange = true;
			errorMessagess<<"need smaller gama: bestlamada "<<bestlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	mingama: "<<mingama<<"	bestrs:	"<<bestRankingScore<<endl;
		}		
		if (isWrongRange)
		{
			//cout<<"---------------------------------------isWrongRange-----------------------------------------------"<<endl;
			string tempcontents = errorMessagess.str();
			cout<<tempcontents<<endl;
			cout<<"--------------------------------------isWrongRange finish this"<<times<<"------------------------"<<endl;
		}

		//here to add some code to record the old rs
		if (hasOldPara)
		{
			if (((paraGroupVector[times].oldRS - bestRankingScore) <= wucha  && paraGroupVector[times].oldRS>0) || ((bestlamada = paraGroupVector[times].lamada) && (bestsita = paraGroupVector[times].sita) && (bestgama = paraGroupVector[times].gama)))
			{
				prametesArray[times][0] = paraGroupVector[times].lamada;
				prametesArray[times][1] = paraGroupVector[times].sita;
				prametesArray[times][2] = paraGroupVector[times].gama;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes+1;
				prametesArray[times][3] = paraGroupVector[times].oldRS;
				cout<<"bestlamada is old one: "<<prametesArray[times][0]<<"	bestsita: "<<prametesArray[times][1]<<"\tbestgama: "<<prametesArray[times][2]<<"	bestrs:	"<<prametesArray[times][3]<<endl;
			}else{
				prametesArray[times][0] = bestlamada;
				prametesArray[times][1] = bestsita;
				prametesArray[times][2] = bestgama;
				prametesArray[times][3] = bestRankingScore;
				prametesArray[times][4] = paraGroupVector[times].isTheBestTimes;
				cout<<"bestlamada: "<<bestlamada<<"\tbestsita: "<<bestsita<<"\tbestgama: "<<bestgama<<"\tbestrs:	"<<bestRankingScore<<endl;
			}
		}else{
			prametesArray[times][0] = bestlamada;
			prametesArray[times][1] = bestsita;
			prametesArray[times][2] = bestgama;
			prametesArray[times][3] = bestRankingScore;
			prametesArray[times][4] = 0;
			cout<<"bestlamada: "<<bestlamada<<"\tbestsita: "<<bestsita<<"\tbestgama: "<<bestgama<<"\tbestrs:	"<<bestRankingScore<<endl;
		}

		//---------------------------------------------
		runingMode = 1;//set testdate to testSet. 90%---10%
		//---------------------------------------------
		
		//Basied_PD_RE_MD(prametesArray[times][0],prametesArray[times][1],prametesArray[times][2]);
		Heter_PD_RE_MD(prametesArray[times][0],prametesArray[times][1],prametesArray[times][2]);
		//rsArray[times] = RankingScoreNotCollect();

		rsAndOthersArray[0][times] = getRankingScore();
		rsAndOthersArray[1][times] = getLocalRankingScore();
		rsAndOthersArray[2][times] = Precision();
		rsAndOthersArray[3][times] = Recall();
		rsAndOthersArray[4][times] = IntraSimilarity();
		rsAndOthersArray[5][times] = HammingDistance();
		rsAndOthersArray[6][times] = Popularity();
		//rsAndOthersArray[7][times] = bestRankingScore;

		//prametesArray[times][3] = rsAndOthersArray[0][times];
		cout<<"RS on 9---1 dividing datab	"<<rsAndOthersArray[0][times]<<endl;
		sum+=rsAndOthersArray[0][times];
	}


	avg = sum/cishu;

	stringstream tempcontentstream;
	tempcontentstream<<"t	lamada	sita	gama	oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul\tbesttimes"<<endl;
	writefile(resultfile,tempcontentstream.str());
	tempcontentstream.str("");

	stringstream oldParaFile;
	stringstream oldParaLogFile;

	for (int times = 0; times<cishu; times++)
	{
		avgLamada += prametesArray[times][0];
		avgSita += prametesArray[times][1];
		avgGama += prametesArray[times][2];
		avgOldRS+= prametesArray[times][3];

		evaluationIndex[0] += rsAndOthersArray[0][times];
		evaluationIndex[1] += rsAndOthersArray[1][times];
		evaluationIndex[2] += rsAndOthersArray[2][times];
		evaluationIndex[3] += rsAndOthersArray[3][times];
		evaluationIndex[4] += rsAndOthersArray[4][times];
		evaluationIndex[5] += rsAndOthersArray[5][times];
		evaluationIndex[6] += rsAndOthersArray[6][times];

		stadardivation+=((rsAndOthersArray[0][times]-avg)*(rsAndOthersArray[0][times]-avg));
		//		cout<<"net "<<times<<" lamada "<<prametesArray[times][0]<<" rs is: "<<prametesArray[times][3]<<endl;
		tempcontentstream<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2];
		//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;;
		tempcontentstream<<"\t"<<prametesArray[times][3]<<"\t"<<rsAndOthersArray[0][times]<<"\t"<<rsAndOthersArray[1][times]<<"\t"<<rsAndOthersArray[2][times]<<"\t"<<rsAndOthersArray[3][times]<<"\t"<<rsAndOthersArray[4][times]<<"\t"<<rsAndOthersArray[5][times]<<"\t"<<rsAndOthersArray[6][times]<<"\t"<<prametesArray[times][4];
		cout<<tempcontentstream.str()<<endl;
		writefile(resultfile,tempcontentstream.str());
		tempcontentstream.str("");

		oldParaLogFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\t"<<prametesArray[times][3]<<"\t"<<prametesArray[times][4]<<"\r\n";

		oldParaFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\t"<<prametesArray[times][4];
		oldParaFile<<"\t"<<prametesArray[times][3]<<"\t"<<rsAndOthersArray[0][times]<<"\t"<<rsAndOthersArray[1][times]<<"\t"<<rsAndOthersArray[2][times]<<"\t"<<rsAndOthersArray[3][times]<<"\t"<<rsAndOthersArray[4][times]<<"\t"<<rsAndOthersArray[5][times]<<"\t"<<rsAndOthersArray[6][times];
		writefile("oldparam.txt",oldParaFile.str());
		oldParaFile.str("");
	}
	//cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;
	removeAndWritefile("oldparam_log.txt",oldParaLogFile.str());
	oldParaLogFile.str("");

	avgLamada = avgLamada/cishu;
	avgGama = avgGama/cishu;
	avgSita = avgSita/cishu;
	avgOldRS = avgOldRS/cishu;

	evaluationIndex[0] = evaluationIndex[0]/cishu;
	evaluationIndex[1] = evaluationIndex[1]/cishu;
	evaluationIndex[2] = evaluationIndex[2]/cishu;
	evaluationIndex[3] = evaluationIndex[3]/cishu;
	evaluationIndex[4] = evaluationIndex[4]/cishu;
	evaluationIndex[5] = evaluationIndex[5]/cishu;
	evaluationIndex[6] = evaluationIndex[6]/cishu;

	stadardivation = sqrt(stadardivation/cishu);

	stringstream avgParamStream;

	avgParamStream<<funcName<<endl;
	avgParamStream<<"minlamada: "<<minlamada<<" maxlamada: "<<maxlamada<<" lamadaIncremental: "<<lamadaIncremental<<endl;
	avgParamStream<<"minsita: "<<minsita<<" maxsita: "<<maxsita<<" sitaIncremental: "<<sitaIncremental<<endl;
	avgParamStream<<"mingama: "<<mingama<<" maxgama: "<<maxgama<<" gamaIncremental: "<<gamaIncremental<<endl;
	avgParamStream<<"oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul	lamada	sita	gama	standdivation"<<endl;
	avgParamStream<<"avg: "<<avgOldRS<<"\t"<<avg<<"\t"<<evaluationIndex[1]<<"\t"<<evaluationIndex[2]<<"\t"<<evaluationIndex[3]<<"\t"<<evaluationIndex[4]<<"\t"<<evaluationIndex[5]<<"\t"<<evaluationIndex[6]<<"\t"<<avgLamada<<"\t"<<avgSita<<"\t"<<avgGama<<"\t"<<stadardivation <<endl;

	cout<<"stadardivation:	"<<stadardivation <<" avg: "<<avg<<endl;
	cout<<"avgParam:"<<avgLamada<<"\t"<<avgSita<<"\t"<<avgGama<<"\t"<<" stadardivation:	"<<stadardivation <<endl<<"avg: "<<avgOldRS<<"\t"<<avg<<"\t"<<evaluationIndex[1]<<"\t"<<evaluationIndex[2]<<"\t"<<evaluationIndex[3]<<"\t"<<evaluationIndex[4]<<"\t"<<evaluationIndex[5]<<"\t"<<evaluationIndex[6]<<endl;

	writefile(resultfile,avgParamStream.str());
	return stadardivation;
}

//一个参数的,这里加入计算90——10划分数据集的时候，遍历得到最优rs，用来和80——10——10划分比较，另外这里重新计算了localrs
//这里的rs是直接读取之前的数据结果，找到最优参数直接计算。
double getStandardDevation()
{
	const int cishu = 10;
	bool hasOldPara = false;
	int counterX = 0;
	double rsAndOthersArray[7][cishu];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS
	double evaluationIndex[7];//记录其他一些观测量的。0是RS，1是localRS，2Precision	3Recall	4IntraSimilarity	5HammingDistance	6Popularity 7 80%-10% RS

	double avg = 0.0,sum = 0.0, stadardivation=0.0;
	double bestsita = 0.0, bestlamada = 0.0,bestgama = 0.0,bestRankingScore = 10.0, temprs = 0.0,avgLamada = 0.0,avgSita = 0.0,avgGama = 0.0,avgOldRS = 0.0, lastRankingScore = 10.0;
	double prametesArray[cishu][4];
	memset(prametesArray,0,sizeof(prametesArray));
	memset(prametesArray,0,sizeof(rsAndOthersArray));
	oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem
	string resultfile = generateFilename();	

	ifstream infile("oldparam.txt");	
	if(!infile)
	{
		cout << "can not open this file oldparam.txt"<< endl;
		hasOldPara = false;
	}else{
		getLastRuningParmeter();
		hasOldPara = true;
	}

	for (int times = 0; times<cishu; times++)
	{
		bool isWrongRange = false;
		lastRankingScore = 10.0;
		bestlamada = 0.0, bestsita = 0.0, bestgama = 0.0,bestRankingScore = 1.0, temprs = 0.0;

		if (hasOldPara)
		{
			for (int i = 0; i< 10;i++)
			{
				cout<<paraGroupVector[i].lamada<<endl;
			}
			if (paraGroupVector[times].lamada!=0)
			{
				minlamada = paraGroupVector[times].lamada-lamadaIncremental;
				maxlamada = minlamada+2*lamadaIncremental;
				lamadaIncremental = 0.01;
			}

			if (paraGroupVector[times].sita!=0)
			{
				minsita = paraGroupVector[times].sita-sitaIncremental;
				maxsita = minsita+2*sitaIncremental;
				sitaIncremental = 0.01;
			}

			if (paraGroupVector[times].gama!=0)
			{
				mingama = paraGroupVector[times].gama-gamaIncremental;
				maxgama = mingama+2*gamaIncremental;
				gamaIncremental = 0.01;
			}
		}

		cout<<"minlamada: "<<minlamada<<" maxlamada: "<<maxlamada<<" lamadaIncremental: "<<lamadaIncremental<<endl;
		cout<<"minsita: "<<minsita<<" maxsita: "<<maxsita<<" sitaIncremental: "<<sitaIncremental<<endl;
		cout<<"mingama: "<<mingama<<" maxgama: "<<maxgama<<" gamaIncremental: "<<gamaIncremental<<endl;

		init(times);
		for (double lamada = minlamada; lamada<=maxlamada; lamada+=lamadaIncremental)
		{
			if (isWrongRange&&(minsita==maxsita)&&(maxgama==mingama))
			{
				cout<<"minsita==maxsita &&(maxgama==mingama)"<<endl;
				break;
			}

			//reset isWrongRange
			isWrongRange = false;

			for (double sita = minsita; sita<=maxsita; sita+=sitaIncremental)
			{
				if (isWrongRange&&(maxgama==mingama))
				{
					cout<<"mingama==maxgama"<<endl;
					break;
				}
				for (double gama = mingama; gama<=maxgama; gama+=gamaIncremental)
				{
					//---------------------------------------------
					runingMode = 0;//set testdate to train. 80%---10%
					//---------------------------------------------

					//counterX++;
					//WHC(lamada,trainingSet);
					//IHCMatrix(lamada,trainingSet);
					//HHCMatrix(lamada,trainingSet);
					//hybirdHAndPNonLinaer(lamada,trainingSet);
					//RE_NBI(lamada,trainingSet);
					//Heter_NBI(lamada,trainingSet);
					//PD(lamada,trainingSet);
					//URA_NBI(lamada,trainingSet);
					//Biased_Heat(lamada,trainingSet);
					//Cold_Start(lamada,trainingSet);
					//NCF(lamada,trainingSet);
					//NCFNew(lamada,trainingSet);
					//MCF(lamada,trainingSet);
					//HeatS(trainingSet);
					//B_Rank(trainingSet);
					//ProbS(1,trainingSet);
					//temprs = RankingScoreNotCollect(learningSet);
					Heter_PD(lamada,sita,trainingSet);
					//SPD(lamada,sita,trainingSet);
					//Basied_PD_RE_MD(lamada,sita,gama,trainingSet);
					/*Heter_PD_RE_MD(lamada,sita,gama);*/
					temprs = getRankingScore();
					cout<<"lamada: "<<lamada<<"	sita: "<<sita<<" gama: "<<gama<<"	nowRS: "<<temprs<<"	lastRankingScore "<<lastRankingScore<<endl;

					if (temprs>lastRankingScore)
					{
						isWrongRange = true;
						cout<<"---------------------------------------"<<times<<"---------------------------------------------"<<endl;
						cout<<"lamada: "<<lamada<<"	sita: "<<sita<<" gama: "<<gama<<"	RS: "<<temprs<<endl;
						cout<<"last is the best bestlamada: "<<bestlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	temprs:	"<<temprs<<endl;

						lastRankingScore = 10.0;
						break;
					}

					lastRankingScore = temprs;	

					if ((temprs - bestRankingScore) <= wucha)
					{
						bestRankingScore = temprs;
						bestsita = sita;
						bestlamada = lamada;
						bestgama = gama;					
					}
				}
			}
		}

		stringstream errorMessagess;
		if (bestlamada >= maxlamada)
		{
			isWrongRange = true;
			errorMessagess<<"need bigger lamada: bestlamada "<<bestlamada<<"	maxlamada: "<<maxlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	bestrs:	"<<bestRankingScore<<endl;
		}else if (bestlamada <= minlamada)
		{
			isWrongRange = true;
			errorMessagess<<"need smaller lamada: bestlamada "<<bestlamada<<"	minlamada: "<<minlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	bestrs:	"<<bestRankingScore<<endl;
		}
		if (bestsita >= maxsita)
		{
			isWrongRange = true;
			errorMessagess<<"need bigger sita: bestlamada "<<bestlamada<<"	bestsita: "<<bestsita<<"	maxsita: "<<maxsita<<"	bestgama: "<<bestgama<<"	bestrs:	"<<bestRankingScore<<endl;
		}else if (bestsita <= minsita)
		{
			isWrongRange = true;
			errorMessagess<<"need smaller sita: bestlamada "<<bestlamada<<"	minsita: "<<minsita<<"	maxsita: "<<maxsita<<"	bestgama: "<<bestgama<<"	bestrs:	"<<bestRankingScore<<endl;
		}
		if (bestgama >= maxgama)
		{
			isWrongRange = true;
			errorMessagess<<"need bigger gama: bestlamada "<<bestlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	maxgama: "<<maxgama<<"	bestrs:	"<<bestRankingScore<<endl;
		}else if (bestgama <= mingama)
		{
			isWrongRange = true;
			errorMessagess<<"need smaller gama: bestlamada "<<bestlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	mingama: "<<mingama<<"	bestrs:	"<<bestRankingScore<<endl;
		}		
		if (isWrongRange)
		{
			//cout<<"---------------------------------------isWrongRange-----------------------------------------------"<<endl;
			string tempcontents = errorMessagess.str();
			cout<<tempcontents<<endl;
			cout<<"--------------------------------------isWrongRange finish this"<<times<<"------------------------"<<endl;
		}
		cout<<"bestlamada: "<<bestlamada<<"	bestsita: "<<bestsita<<"	bestgama: "<<bestgama<<"	bestrs:	"<<bestRankingScore<<endl;

		prametesArray[times][0] = bestlamada;
		prametesArray[times][1] = bestsita;
		prametesArray[times][2] = bestgama;
		prametesArray[times][3] = bestRankingScore;

		//---------------------------------------------
		runingMode = 1;//set testdate to testSet. 90%---10%
		//---------------------------------------------
		//WHC(prametesArray[times][0]);
		//IHCMatrix(prametesArray[times][0]);
		//HHCMatrix(prametesArray[times][0]);
		//hybirdHAndPNonLinaer(prametesArray[times][0]);
		//RE_NBI(prametesArray[times][0]);
		//Heter_NBI(prametesArray[times][0]);
		//PD(prametesArray[times][0]);
		//URA_NBI(prametesArray[times][0]);
		//Biased_Heat(prametesArray[times][0]);
		//Cold_Start(prametesArray[times][0]);
		//NCF(prametesArray[times][0],trainingSet);
		//NCFNew(prametesArray[times][0],trainingSet);
		//MCF(prametesArray[times][0],trainingSet);
		//HeatS();
		//B_Rank();
		//ProbS(1);
		Heter_PD(prametesArray[times][0],prametesArray[times][1]);
		//SPD(prametesArray[times][0],prametesArray[times][1]);
		//Basied_PD_RE_MD(prametesArray[times][0],prametesArray[times][1],prametesArray[times][2],trainingSet);
		/*Heter_PD_RE_MD(prametesArray[times][0],prametesArray[times][1],prametesArray[times][2],trainingSet);*/
		//rsArray[times] = RankingScoreNotCollect();

		rsAndOthersArray[0][times] = getRankingScore();
		rsAndOthersArray[1][times] = getLocalRankingScore();
		rsAndOthersArray[2][times] = Precision();
		rsAndOthersArray[3][times] = Recall();
		rsAndOthersArray[4][times] = IntraSimilarity();
		rsAndOthersArray[5][times] = HammingDistance();
		rsAndOthersArray[6][times] = Popularity();
		//rsAndOthersArray[7][times] = bestRankingScore;

		//prametesArray[times][3] = rsAndOthersArray[0][times];
		cout<<"RS on 9---1 dividing datab	"<<rsAndOthersArray[0][times]<<endl;
		sum+=rsAndOthersArray[0][times];
	}
	avg = sum/cishu;

	stringstream tempcontentstream;
	tempcontentstream<<"t	lamada	sita	gama	oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul"<<endl;
	writefile(resultfile,tempcontentstream.str());
	tempcontentstream.str("");

	stringstream oldParaFile;

	for (int times = 0; times<cishu; times++)
	{
		avgLamada += prametesArray[times][0];
		avgSita += prametesArray[times][1];
		avgGama += prametesArray[times][2];
		avgOldRS+= prametesArray[times][3];

		evaluationIndex[0] += rsAndOthersArray[0][times];
		evaluationIndex[1] += rsAndOthersArray[1][times];
		evaluationIndex[2] += rsAndOthersArray[2][times];
		evaluationIndex[3] += rsAndOthersArray[3][times];
		evaluationIndex[4] += rsAndOthersArray[4][times];
		evaluationIndex[5] += rsAndOthersArray[5][times];
		evaluationIndex[6] += rsAndOthersArray[6][times];

		stadardivation+=((rsAndOthersArray[0][times]-avg)*(rsAndOthersArray[0][times]-avg));
		//		cout<<"net "<<times<<" lamada "<<prametesArray[times][0]<<" rs is: "<<prametesArray[times][3]<<endl;
		tempcontentstream<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2];
		oldParaFile<<times<<"\t"<<prametesArray[times][0]<<"\t"<<prametesArray[times][1]<<"\t"<<prametesArray[times][2]<<"\r\n";
		cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;;
		tempcontentstream<<"\t"<<prametesArray[times][3]<<"\t"<<rsAndOthersArray[0][times]<<"\t"<<rsAndOthersArray[1][times]<<"\t"<<rsAndOthersArray[2][times]<<"\t"<<rsAndOthersArray[3][times]<<"\t"<<rsAndOthersArray[4][times]<<"\t"<<rsAndOthersArray[5][times]<<"\t"<<rsAndOthersArray[6][times];
		cout<<tempcontentstream.str()<<endl;
		writefile(resultfile,tempcontentstream.str());
		tempcontentstream.str("");
	}
	cout<<"oldParaFile.str()"<<oldParaFile.str()<<endl;;
	removeAndWritefile("oldparam.txt",oldParaFile.str());
	oldParaFile.str("");

	avgLamada = avgLamada/cishu;
	avgGama = avgGama/cishu;
	avgSita = avgSita/cishu;
	avgOldRS = avgOldRS/cishu;

	evaluationIndex[0] = evaluationIndex[0]/cishu;
	evaluationIndex[1] = evaluationIndex[1]/cishu;
	evaluationIndex[2] = evaluationIndex[2]/cishu;
	evaluationIndex[3] = evaluationIndex[3]/cishu;
	evaluationIndex[4] = evaluationIndex[4]/cishu;
	evaluationIndex[5] = evaluationIndex[5]/cishu;
	evaluationIndex[6] = evaluationIndex[6]/cishu;

	stadardivation = sqrt(stadardivation/cishu);

	stringstream avgParamStream;

	avgParamStream<<funcName<<endl;
	avgParamStream<<"minlamada: "<<minlamada<<" maxlamada: "<<maxlamada<<" lamadaIncremental: "<<lamadaIncremental<<endl;
	avgParamStream<<"minsita: "<<minsita<<" maxsita: "<<maxsita<<" sitaIncremental: "<<sitaIncremental<<endl;
	avgParamStream<<"mingama: "<<mingama<<" maxgama: "<<maxgama<<" gamaIncremental: "<<gamaIncremental<<endl;
	avgParamStream<<"oldRS	newRS	lrs	pricis	recall	intrSim	hamdis	Popul	lamada	sita	gama	standdivation"<<endl;
	avgParamStream<<"avg: "<<avgOldRS<<"\t"<<avg<<"\t"<<evaluationIndex[1]<<"\t"<<evaluationIndex[2]<<"\t"<<evaluationIndex[3]<<"\t"<<evaluationIndex[4]<<"\t"<<evaluationIndex[5]<<"\t"<<evaluationIndex[6]<<"\t"<<avgLamada<<"\t"<<avgGama<<"\t"<<avgSita<<"\t"<<stadardivation <<endl;

	cout<<"stadardivation:	"<<stadardivation <<" avg: "<<avg<<endl;
	cout<<"avgParam:"<<avgLamada<<"\t"<<avgSita<<"\t"<<avgGama<<"\t"<<" stadardivation:	"<<stadardivation <<endl<<"avg: "<<avgOldRS<<"\t"<<avg<<"\t"<<evaluationIndex[1]<<"\t"<<evaluationIndex[2]<<"\t"<<evaluationIndex[3]<<"\t"<<evaluationIndex[4]<<"\t"<<evaluationIndex[5]<<"\t"<<evaluationIndex[6]<<endl;

	writefile(resultfile,avgParamStream.str());
	return stadardivation;
}

void testHere(){
	const int cishu = 10;
	double rsArray[cishu];
	double avg = 0.0,sum = 0.0, stadardivation=0.0;
	double bestsita = 0.0, bestlamada = 0.0,bestgama = 0.0,bestRankingScore = 1.0, temprs = 0.0;
	double prametesArray[cishu][4];
	double lamamdas[cishu] = {0.17,0.18,0.19,0.19,0.18,0.18,0.18,0.18,0.18,0.18};
	memset(prametesArray,0,sizeof(prametesArray));
	oldNet.loadNetworkFromFile(inFileName);//这里计算rs的时候要用到oldNet的一些度，uncollectitem
	for (int times = 0; times<cishu; times++)
	{
		init(times);
		cout<<"bestlamada: "<<lamamdas[times]<<"	bestrs:	"<<bestRankingScore<<endl;
		hybirdHAndPNonLinaer(lamamdas[times]);
		rsArray[times] = RankingScoreNotCollect();
		prametesArray[times][3] = rsArray[times];
		prametesArray[times][1] = lamamdas[times];
		cout<<" rsArray[times]	"<<rsArray[times] <<endl;
		sum+=rsArray[times];
	}
	avg = sum/cishu;
	for (int times = 0; times<cishu; times++)
	{
		stadardivation+=((rsArray[times]-avg)*(rsArray[times]-avg));
		cout<<"net "<<times<<" lamada "<<prametesArray[times][1]<<" rs is: "<<prametesArray[times][3]<<endl;
	}
	stadardivation = sqrt(stadardivation/cishu);
	cout<<" stadardivation:	"<<stadardivation <<" avg: "<<avg<<endl;
}

void quick_sort(int unsort[],int begin,int end)
{
	if (begin>=end)
	{
		return;
	}
	int key = unsort[begin];
	int i = begin+1, j = end;

	while(1)
	{
		//cout << i<<"	"<<j<<endl;
		while(unsort[j]>key && i<j)
		{
			j--;
		}
		while(unsort[i]<key && i<j)
		{
			i++;
		}
		if (i>=j)
		{
			break;
		}
		swap(unsort[i],unsort[j]);
		if (unsort[i] == key)
		{
			j--;
		}else{
			i++;
		}
	}
	swap(unsort[begin],unsort[j]);

	if (begin<i-1)
	{
		quick_sort(unsort,begin,i-1);
	}
	if (j+1<end)
	{
		quick_sort(unsort,j+1,end);
	}
}

//C++'s array range should be [begin, end], the same as [begin, end+1)
//http://blog.csdn.net/kenden23/article/details/14643823
int partition(vector<int> &vect, int begin, int end)
{
	int key = vect[end];
	int i = begin-1;
	for (int j = begin; j < end; j++)
	{
		//这里的话因为比较因子key是最右边的一个，这边一直在左边往右循环，j对应的全部都是比key小的，而i对应的都是比key大的，i是从0开始递增，然后a[i]一直被换成比key小的，然后i移向下一个。a[i]之前就全部都是比key小的了。
		if(vect[j] <= key)
		{
			i++;
			//cout <<"swap a["<<i<<"]="<<vect[i]<<"	and a["<<j<<"]="<<vect[j]<<endl;
			if (i!=j)
			{
				swap(vect[i], vect[j]);
				for(int i=0;i<10;i++) {
					cout<<vect[i]<<" ";
				}
				cout<<endl;
			}
		}
	}
	cout << vect[i]<<"	i	"<<endl;
	swap(vect[i+1], vect[end]);
	for(int i=0;i<10;i++) {
		cout<<vect[i]<<" ";
	}
	cout<<endl;
	return i+1;
}

//C++'s array range should be [low, up], the same as [low, up+1)
void quickSort(vector<int> &vi, int low, int up)
{
	if(low < up)
	{
		int mid = partition(vi, low, up);

		//Watch out! The mid position is on the place, so we don't need to consider it again.
		//That's why below is mid-1, not mid! Otherwise it will occur overflow error!!!
		quickSort(vi, low, mid-1);
		quickSort(vi, mid+1, up);
	}
}

void qSort(vector<int> &vi)
{
	quickSort(vi, 0, vi.size()-1);
}

int main()
{
	//system("mode con cols=160");
	//getStandardDevation();//
	//test();

	intiPramate();

	long start=clock(),end(0);

	int flag = 3;

	if (flag == 0)
	{
		testifright();
	}else if(flag==1){
		train();
	}else if(flag==2){
		test();
	}
	else if (flag == 3){
		//calculataLocalRsAgain();

		calculata9010RsOne();
		//calculata9010RsTwo();
		//calculata9010RsThree();


		//getStandardDevatione();
		//getStandardDevationOne();
		//getStandardDevationTwo();
		//getStandardDevationThree();
	}else{
		getProbs();
//		testHere();
		//divideto3set10Times();
	}
	end=clock();
	long result=(end-start);
	cout<<" time is : "<<result<<endl;
	system("pause");
}