/****************************************************************************
*   ����:  ����
*   ����:  2013-4-14
*   Ŀ��:  ��ȡ�����ļ�����Ϣ����string����ʽ����
*   Ҫ��:  �����ļ��ĸ�ʽ����#��Ϊ��ע�ͣ����õ���ʽ��key = value���м���пո�Ҳ��û�пո�
*****************************************************************************/
#include <string>
#include <map>
#define COMMENT_CHAR '#'//ע�ͷ�
using namespace std;

class operate_config
{
private :
	ifstream *infile;
public:
	operate_config(void);
	//����filename�������ļ�������
	operate_config(const string & filename);
	//����name�������������
	//����ֵ����Ӧ������name��valueֵ
	string getValue();
	double getNumber();

	string getValue(const string & name);
	double getNumber(const string & name);
	map<string,string> configMap;
	~operate_config(void);
};
