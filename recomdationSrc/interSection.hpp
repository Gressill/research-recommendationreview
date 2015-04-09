//#include "SimpleNetwork.h"
#include <iostream>
#include <vector>
using namespace std;
//初始设定表长10000.
template <typename _Type>
class SimpleHashTable
{
protected:
	int Length;//Hash table length
	int Count;//hash table current length
	_Type* Element;//Hash table

public:
	SimpleHashTable(int Length)   //构建哈希表，表长Length
	{
		Element = new _Type[Length];
		for(int i=0;i<Length;i++)
		{
			Element[i] = -1;
		}
		this->Length = Length;
		Count = 0;
	}

	~SimpleHashTable()
	{
		delete[] Element;
	}

	virtual int Hash(_Type Data)
	{
		return Data % this->Length;
	}

	//开放定址法再哈希
	virtual int ReHash(int Index,int Count)
	{
		return ((Index + Count) % Length); //
	}
	
	//查找元素，若已存在返回true，否则返回false
	virtual bool SerachHash(_Type Data,int& Index)
	{
		Index = Hash(Data);
		int Count = 0;

		while(Element[Index] != -1 && Element[Index] != Data)
		{
			Index = ReHash(Index,++Count);
		}

		return (Data == Element[Index] ? true :false);
	}

	virtual int SerachHash(_Type Data)
	{
		int Index = 0;
		if(SerachHash(Data,Index)) 
		{
			return Index;

		}
		else 
		{
			return -1;
		}
	}

	// 插入元素
	bool InsertHash(_Type Data)
	{
		int Index = 0;
		if(Count < Length && !SerachHash(Data,Index))
		{
			Element[Index] = Data;
			Count++;
			return true;
		}   
		//在插入的过程中，如果元素已经存在，即为交集元素则打印之.
		if(SerachHash(Data,Index))
		{
			//std::cout << Data << "\t";
			return false;
		}
		return false;
	}

	//手动设置表长
	void SetLength(int Length)
	{
		delete[] Element;
		Element = new _Type[Length];
		for(int i=0;i<Length;i++)
		{
			Element[i] = -1;
		}
		this->Length = Length;
	}

	//移除元素.
	void Remove(_Type Data)
	{
		int Index = SerachHash(Data);
		if(Index != -1)
		{
			Element[Index] = -1;
			Count--;
		}
	}

	void RemoveAll()
	{
		for (int i = 0;i<Length;i++)
		{
			Element[i] = -1;
		}
		Count = 0;
	}

	void Print()
	{
		for(int i=0;i<Length;i++)
		{
			printf("%d\t",Element[i]);
		}
		printf("\n");
	}
};

//自定义子类.
template <typename _Type>
class SimpleHashSet : public SimpleHashTable<_Type>
{
public:
	SimpleHashSet(int nLen):SimpleHashTable<_Type>(nLen){}
	~SimpleHashSet(){ }
	friend int hashInterSection(SimpleHashSet<_Type>* pHashSet, int a[], int m, int b[], int n);
	friend int hashInterSection(SimpleHashSet<_Type>* pHashSet, const vector<int> &a, int m, const vector<int> &b, int n);
	friend int hashInterSection(SimpleHashSet<_Type>* pHashSet, int a[], int m, const vector<int> &b, int n);
	friend int hashInterSection(SimpleHashSet<_Type>* pHashSet, const vector<int> &a, int m, int b[], int n);

	static const int getInterSectionSize(SimpleHashSet<_Type>* pHashSet, const vector<int>& a, int m, const vector<int>& b, int n)
	{
		//itemsCommonNeighbor.clear();
		int commonItemNumber = 0;
		for(int i = 0; i < m; i++)
		{
			if (!pHashSet->InsertHash(a[i]))
			{
				++commonItemNumber;
				//itemsCommonNeighbor.push_back(a[i]);
			}
		}      

		for(int j = 0; j < n; j++)
		{
			if (!pHashSet->InsertHash(b[j]))
			{
				++commonItemNumber;
				//itemsCommonNeighbor.push_back(b[j]);
			}
		}
		//cout<<commonItemNumber<< endl;
		return commonItemNumber;
	}
private: 
};

