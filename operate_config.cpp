#include "operate_config.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "Constant.h"
using namespace std;

bool IsSpace(char c)//�ж��ǲ��ǿո�
{
    if (' ' == c || '\t' == c)
        return true;
    return false;
}

bool IsCommentChar(char c)//�ж��ǲ���ע�ͷ�
{
    switch(c) {
    case COMMENT_CHAR:
        return true;
    default:
        return false;
    }
}

void Trim(string & str)//ȥ���ַ�������β�ո�
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
    if (i == str.size()) { // ȫ���ǿհ��ַ���
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

//����name�������������
//����ֵ����Ӧ������name��valueֵ
string operate_config::getValue(const string & name)
{
	//����һ����Ŀ����(ʵ����ָ��)
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

//����name�������������
//����ֵ����Ӧ������name��valueֵ
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
        cout << "�޷��������ļ�" << endl;
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
				{  // �еĵ�һ���ַ�����ע���ַ�
					continue;
					//return "";
				}
				end_pos = pos - 1;
			}
			new_line = line.substr(start_pos, start_pos + 1 - end_pos);  // Ԥ����ɾ��ע�Ͳ���
			if ((pos = new_line.find('=')) == -1)
			{
				//			return "";  // û��=��
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
