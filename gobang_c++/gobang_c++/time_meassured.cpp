# include <chrono>
# include <string>
using namespace std;

/**
 * @brief 时间测量函数
 * @param time    时间精度字符串，支持"ms","us","s","ns"
 * @param *func   待测时的函数，必须是void X(void)的函数
 *
 * @return 返回说明
 *     函数执行时间
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