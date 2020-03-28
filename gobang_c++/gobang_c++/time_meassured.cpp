# include <chrono>
# include <string>
using namespace std;

/**
 * @brief ʱ���������
 * @param time    ʱ�侫���ַ�����֧��"ms","us","s","ns"
 * @param *func   ����ʱ�ĺ�����������void X(void)�ĺ���
 *
 * @return ����˵��
 *     ����ִ��ʱ��
 */
double time_meassured(string time, void (*func)(void))
{
    auto start_time = chrono::high_resolution_clock::now();
    (*func)();
    auto end_time = chrono::high_resolution_clock::now();

    if ("us" == time)
    {
        chrono::duration<double, micro> tm_us = end_time - start_time;
        return tm_us.count();
    }
    else if ("ms" == time)
    {
        chrono::duration<double, milli> tm_ms = end_time - start_time;
        return tm_ms.count();
    }
    else if ("ns" == time)
    {
        chrono::duration<double, nano> tm_ns = end_time - start_time;
        return tm_ns.count();
    }
    else if ("s" == time)
    {
        chrono::duration<double, deci> tm_s = end_time - start_time;
        return tm_s.count() / 10;
    }
    else
        return -1;
}