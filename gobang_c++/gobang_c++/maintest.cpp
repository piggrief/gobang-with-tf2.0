# include <iostream>
# include <chrono>
# include "Python.h"
# include <windows.h>
# include "time_meassured.h"
using namespace std;

//把const char *c转 wchar_t * ，作为Py_SetPythonHome()参数匹配
wchar_t* GetWC(const char* c)
{
    size_t convertedChars = 0;
    const size_t cSize = strlen(c) + 1;
    wchar_t* wc = new wchar_t[cSize];
    mbstowcs_s(&convertedChars, wc, cSize, c, _TRUNCATE);

    return wc;
}

void test()
{
    Sleep(3000); //这里延迟500ms
}


int main()
{
    cout << time_meassured("s", test) << "s" <<endl;

    ////初始化(下面的方法可以在c:\APP\Anaconda3\include\pylifecycle.h中找到)
    ////Py_SetProgramName(0);
    ////很关键的一步，去掉导入numpy报错
    //Py_SetPythonHome(GetWC("C:/ProgramData/Anaconda3/envs/ker_spyder"));
    //Py_Initialize();
    ////执行import语句，把当前路径加入路径中，为了找到math_test.py
    //PyRun_SimpleString("import os,sys");

    //PyRun_SimpleString("sys.path.append('./')");
    ////测试打印当前路径
    //PyRun_SimpleString("print(os.getcwd())");


	return 0;
}