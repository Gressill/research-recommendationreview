/****************************************************************************
*   作者:  符星
*   日期:  2013-4-14
*   目的:  读取配置文件的信息，以string的形式返回
*   要求:  配置文件的格式，以#作为行注释，配置的形式是key = value，中间可有空格，也可没有空格
*****************************************************************************/
#include <string>
#include <map>
#define COMMENT_CHAR '#'//注释符
using namespace std;

class operate_config
{
private :
	ifstream *infile;
public:
	operate_config(void);
	//参数filename，配置文件的名字
	operate_config(const string & filename);
	//参数name，配置项的名字
	//返回值，对应配置项name的value值
	string getValue();
	double getNumber();

	string getValue(const string & name);
	double getNumber(const string & name);
	map<string,string> configMap;
	~operate_config(void);
};
