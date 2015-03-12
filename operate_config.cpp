#include "operate_config.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "Constant.h"
using namespace std;

bool IsSpace(char c)//判断是不是空格
{
    if (' ' == c || '\t' == c)
        return true;
    return false;
}

bool IsCommentChar(char c)//判断是不是注释符
{
    switch(c) {
    case COMMENT_CHAR:
        return true;
    default:
        return false;
    }
}

void Trim(string & str)//去除字符串的首尾空格
{
    if (str.empty()) {
        return;
    }
    int i, start_pos, end_pos;
    for (i = 0; i < str.size(); ++i) {
        if (!IsSpace(str[i])) {
            break;
        }
    }
    if (i == str.size()) { // 全部是空白字符串
        str = "";
        return;
    }

    start_pos = i;

    for (i = str.size() - 1; i >= 0; --i) {
        if (!IsSpace(str[i])) {
            break;
        }
    }
    end_pos = i;

    str = str.substr(start_pos, end_pos - start_pos + 1);
}

//参数name，配置项的名字
//返回值，对应配置项name的value值
string operate_config::getValue(const string & name)
{
	//定义一个条目变量(实际是指针)
	map<string,string>::iterator it= configMap.find(name); 
	if(it == configMap.end()) {
		return "not found";
	}
	else {
		string value = (*it).second;
		Trim(value);
		return value;
	}	
}

//参数name，配置项的名字
//返回值，对应配置项name的value值
double operate_config::getNumber(const string & name)
{
	string value = getValue(name);
	if (value == "not found")
	{
		return notFound_number;
	}
	Trim(value);
	double n = atof(value.c_str());
	return  n;
}

operate_config::operate_config(const string & filename)
{
	infile=new ifstream();//filename.c_str()
    infile->open(filename.c_str());
    if (!infile->is_open())
	{
        cout << "无法打开配置文件" << endl;
    }else
    {
        cout << "open config file "<<filename.c_str()<< endl;

		string line;
		string new_line;
		while (getline(*infile, line))
		{
			if (line.empty())
			{
				//return 0;
				continue;
			}
			int start_pos = 0, end_pos = line.size() - 1, pos;
			if ((pos = line.find(COMMENT_CHAR)) != -1)
			{
				if (0 == pos)
				{  // 行的第一个字符就是注释字符
					continue;
					//return "";
				}
				end_pos = pos - 1;
			}
			new_line = line.substr(start_pos, start_pos + 1 - end_pos);  // 预处理，删除注释部分
			if ((pos = new_line.find('=')) == -1)
			{
				//			return "";  // 没有=号
				continue;
			}
			string na=new_line.substr(0, pos);
			Trim(na);
			string value=new_line.substr(pos + 1, end_pos + 1- (pos + 1));
			Trim(value);
			configMap[na] =value;
			//cout<<line<<" na "<<na<<"	"<<configMap[na]<<endl;
		}
    }
}

operate_config::operate_config(void)
{
}
operate_config::~operate_config(void)
{}
