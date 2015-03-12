#include "SimpleNetwork.h"
#include "interSection.hpp"
//oriTopoNetwork(userNum,vector<int>(itemNum,0));
SimpleNetwork::SimpleNetwork(bool dirc)
{
	//memset(oriTopoNetwork,0,sizeof(oriTopoNetwork));
	vector<int> tempU;
	for (int i = 0; i<userNum;i++)
	{
		/*user_item_relation.push_back(tempU);*/
		user_item_relation[i] = tempU;
	}
	vector<int> tempN;
	for (int i = 0; i<itemNum;i++)
	{
		//item_user_relation.push_back(tempN);
		item_user_relation[i] = tempN;
	}
	//init();
}

SimpleNetwork::~SimpleNetwork(void)
{
}

void SimpleNetwork::initUserItemMatrix()
{
	//emptyVector(&oriTopoNetwork);
	for (int i1 =0;i1<userNum;i1++)
	{
		oriTopoNetwork[i1].clear();
	}

	for (int i = 0; i<userNum; i++)
	{
		int tsize = user_item_relation[i].size();
		if (tsize>0)
		{
			for (int j = 0; j < tsize; j++)
			{
				int tempItemId = user_item_relation[i][j];
				oriTopoNetwork[i][tempItemId] = 1;
			}
		}
	}
}

void SimpleNetwork::testSymmDegree()
{
	for (int i =0; i<userNum; i++)
	{
		for (int j = 0;j<userNum; j++)
		{

		}
	}
}

int SimpleNetwork::loadNetworkFromFile(string filename, bool startFromZero)
{
	int maxuser=0, maxitem=0;
	isStartFromZero = startFromZero;
	char buffer[256];
	if (filename.length()==0)
	{
		filename = inFileName;
	}

	ifstream infile(filename);
	if(!infile)
	{
		cout << "can not open this file" <<filename<< endl;
		return 0;
	}
	whichSet = filename.substr(filename.size()-15);
	cout<<whichSet<<"	degree:	";
	if (infile.is_open())
	{
		int a,b,score;
		int i=0,j=0;
		while (!infile.eof() )
		{
			infile.getline (buffer,33);
			sscanf_s(buffer,"%d %d %d",&a,&b,&score);
			//cout<<a<<" "<<b<<"	"<<score<<" end"<<endl;
			if (score<3)//&&sizeof(score)>0)
			{
				continue;
			}
			if (maxuser<a)
			{
				maxuser = a;
			}if (maxitem<b)
			{
				maxitem = b;
			}
			if (startFromZero)
			{
				addEdge(a,b);
			}else{
				addEdge(a-1,b-1);
			}
			i++;
		}
		infile.close();
	}
	else
	{
		cout<<"Error,can't open...\n";
		//exit(2);
	}
	//testSymmDegree();
	cout<<"loading network complete! has user: "<<maxuser<<" item : "<<maxitem<<endl;
	return 0;
}

int SimpleNetwork::loadNetworkFromFile(string filename)
{
	char buffer[256];
	int maxuser=0, maxitem=0;
	if (filename.length()==0)
	{
		filename = inFileName;
	}

	ifstream infile(filename);
	if(!infile)
	{
		cout << "can not open this file" <<filename<< endl;
		return 0;
	}
	whichSet = filename.substr(filename.size()-15);
	cout<<whichSet<<"	degree:	";
	if (infile.is_open())
	{
		int a,b,score;
		int i=0,j=0;
		while (!infile.eof() )
		{
			infile.getline (buffer,33);
			sscanf_s(buffer,"%d %d %d",&a,&b,&score);
			//cout<<a<<" "<<b<<"	"<<score<<" end"<<endl;
			if (score<3&&score>0)
			{
				continue;
			}
			if (maxuser<a)
			{
				maxuser = a;
			}if (maxitem<b)
			{
				maxitem = b;

			}
			/*if (maxitem>1700||maxuser>1000)
			{
				cout<<"line : "<<i<<"	"<<a<<" "<<b<<"	"<<score<<"	"<<c<<"	"<<d<<"	"<<maxuser<<"	"<<maxitem<<endl;
				break;
			}
			*/
			if (a == 0)
			{
				empty();
				loadNetworkFromFile(filename,true);
				cout<<"this file user start from 0"<<endl;
				return -1;
			}
			if ( b == 0)
			{
				empty();
				loadNetworkFromFile(filename,true);
				cout<<"this file item start from 0"<<endl;
				return -1;
			}
			addEdge(a-1,b-1);
			i++;
		}
		infile.close();
	}
	else
	{
		cout<<"Error,can't open...\n";
		//exit(2);
	}
	//testSymmDegree();
	//cout<<"maxitem	"<<maxitem<<" maxuser	"<<maxuser<<endl;
	//cout<<"loading network complete! has user: "<<maxuser<<" item : "<<maxitem<<endl;
	cout<<countDegree()<<endl;;
	return 0;
}

SimpleNetwork::SimpleNetwork(int a, int b)
{
	SimpleNetwork(true);
	//addEdge(a,b);
}

int split(string oldstring){

	//string test = "aa aa bbc cccd";
	vector<string> strvec;
	string strtemp;

	string::size_type pos1, pos2;
	pos2 = oldstring.find(' ');
	pos1 = 0;
	while (string::npos != pos2)
	{
		strvec.push_back(oldstring.substr(pos1, pos2 - pos1));

		pos1 = pos2 + 1;
		pos2 = oldstring.find(' ', pos1);
	}
	strvec.push_back(oldstring.substr(pos1));

	vector<string>::iterator iter1 = strvec.begin(), iter2 = strvec.end();
	while (iter1 != iter2)
	{
		cout << *iter1 << endl;
		++iter1;
	}
	return 0;
}

void SimpleNetwork::addEdge(int a, int b)
{
	/*UGraph->AddEdge(a,b);*/
	//if (oriTopoNetwork[a][b] == 0)
	//{
	//	oriTopoNetwork[a][b] = 1;
	//	//pointRelation[a].push_back(b);
	//	//origenPointRelation[a].push_back(b);
	//}
	user_item_relation[a].push_back(b);
	item_user_relation[b].push_back(a);
	//cout<<a<<"	"<<user_item_relation[a].back()<<endl;
	//cout<<"countdegessis : "<<countDegree(2)<<endl;
}

void SimpleNetwork::empty()
{
	for (int i = 0; i< userNum; i++)
	{
		user_item_relation[i].clear();
	}
	for (int i = 0; i< itemNum; i++)
	{
		item_user_relation[i].clear();
	}
}

void SimpleNetwork::addEdge(vector<int> *templineV)
{
	vector<int>::iterator itr = templineV->begin();
	int firstpoint =*itr;
	for (itr=templineV->begin()+1;itr!=templineV->end();itr++)
	{
		//cout<<"first is : "<<firstpoint<<" second is : "<<(*itr)<<endl;
		addEdge(firstpoint,*itr);
	}
}

void SimpleNetwork::addEdgeToTempNetwork(int a, int b)
{
	int counter = 0;

	if (oriTopoNetwork[a][b] == 1)
	{
		counter++;
		return;
	}
	//cout<<"add to net "<<a<<"	"<<b<<endl;
	//oriTopoNetwork[a][b] = 1;
	//pointRelation[a].push_back(b);
}

void SimpleNetwork::removePoint(int p)
{
}

int SimpleNetwork::removeEdge(int a, int b)
{
	int counter = 0;
	//cout<<"remove "<<user_item_relation[a]<<"	"<<b<<endl;

	for (vector<int>::iterator it = user_item_relation[a].begin(); it!= user_item_relation[a].end(); ++it)
	{
		if (*it == b)
		{
			//cout<<"before "<<user_item_relation[a].size()<<endl;
			user_item_relation[a].erase(it);
			//cout<<"after "<<user_item_relation[a].size()<<endl;
			counter++;
			break;
		}
	}
	for (vector<int>::iterator it1 = item_user_relation[b].begin(); it1!= item_user_relation[b].end(); ++it1)
	{
		if (*it1 == a)
		{
			item_user_relation[b].erase(it1);
			counter++;
			break;
		}
	}
	if (counter==0)
	{
		//cout<<"not found "<<a<<"	"<<b<<endl;
	}
	if (counter==1)
	{
		cout<<"only delete one direction edge "<<a<<"	"<<b<<endl;
	}
	return counter;
}

bool SimpleNetwork::haveUndealedEdge()
{
	for (int i=0;i<userNum;i++)
	{

	}
	return false;
}

//void writefile(char* filename,string content)
//{
//	/*FILE *F = fopen(filename, "ab+");
//	const char *p =  content.c_str();
//	fprintf(F,"%s\n", p);
//	fclose(F);*/
//}

void SimpleNetwork::printNetwork(const char* filename)
{
	if( remove( filename ) != 0 ){
		cout<<filename<<endl;
		perror( "Error deleting file" );
	}
	/*else
		puts( "File successfully deleted" );*/

	FILE *F;
	fopen_s(&F,filename, "a+");
	//FILE *F = fopen_s(filename, "ab+");
	if (isStartFromZero)
	{
		for (int i = 0; i< userNum; i++)
		{
			for (unsigned int j = 0; j<user_item_relation[i].size();j++ )
			{
				fprintf(F,"%d\t%d\n", i,user_item_relation[i][j]);
			}
		}
	}else
	{
		for (int i = 0; i< userNum; i++)
		{
			for (unsigned int j = 0; j<user_item_relation[i].size();j++ )
			{
				fprintf(F,"%d\t%d\n", i+1,user_item_relation[i][j]+1);
			}
		}
	}
	//const char *p =  content.c_str();
	fclose(F);
}

void SimpleNetwork::whoami()
{
	cout<<"I am : "<<whichSet<<endl;
}
int SimpleNetwork::countDegree(int flag)
{
	int dCounter = 0;
	if (flag == 1)
	{
		for (int i = 0;i<userNum;i++)
		{
			for (int j = 0;j<itemNum;j++)
			{
				//cout<<"oriTopoNetwork["<<i<<"]["<<j<<"]= " <<oriTopoNetwork[i][j]<<endl;
				if (oriTopoNetwork[i][j] == 1)
				{
					//cout<<"oriTopoNetwork["<<i<<"]["<<j<<"]= " <<oriTopoNetwork[i][j]<<endl;
					dCounter++;
				}
			}
		}
	}
	if (flag == 2)
	{
		for (int i = 0; i< userNum; i++)
		{
			dCounter+= user_item_relation[i].size();
		}
	}
	//cout<<"edge number is : "<<dCounter<<endl;
	return dCounter;
}

int SimpleNetwork::getUserMaxDegree()
{
	int tempmaxdegree = 0;
	for (int i=0;i<userNum;i++)
	{
		if (tempmaxdegree<user_item_relation[i].size())
		{
			tempmaxdegree = user_item_relation[i].size();
		}
	}
	return tempmaxdegree;
}

int SimpleNetwork::getItemMaxDegree()
{
	int tempmaxdegree = 0;
	for (int i=0;i<itemNum;i++)
	{
		if (tempmaxdegree<item_user_relation[i].size())
		{
			tempmaxdegree = item_user_relation[i].size();
		}
	}
	return tempmaxdegree;
}

void SimpleNetwork::resetgraph()
{
	for (int i = 0; i< userNum; i++)
	{
		//pointRelation[i].clear();
		//tempPointRelation[i] = origenPointRelation[i];
	}
	//emptyVector(&oriTopoNetwork);
}
