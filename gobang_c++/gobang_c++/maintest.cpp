# include <iostream>
# include <chrono>
# include "Python.h"
# include <windows.h>
# include "time_meassured.h"
using namespace std;

//��const char *cת wchar_t * ����ΪPy_SetPythonHome()����ƥ��
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
    Sleep(3000); //�����ӳ�500ms
}


int main()
{
    cout << time_meassured("s", test) << "s" <<endl;

    ////��ʼ��(����ķ���������c:\APP\Anaconda3\include\pylifecycle.h���ҵ�)
    ////Py_SetProgramName(0);
    ////�ܹؼ���һ����ȥ������numpy����
    //Py_SetPythonHome(GetWC("C:/ProgramData/Anaconda3/envs/ker_spyder"));
    //Py_Initialize();
    ////ִ��import��䣬�ѵ�ǰ·������·���У�Ϊ���ҵ�math_test.py
    //PyRun_SimpleString("import os,sys");

    //PyRun_SimpleString("sys.path.append('./')");
    ////���Դ�ӡ��ǰ·��
    //PyRun_SimpleString("print(os.getcwd())");


	return 0;
}