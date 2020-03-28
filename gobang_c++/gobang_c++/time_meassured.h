#pragma once
# include <string>
/**
 * @brief 时间测量函数
 * @param time    时间精度字符串，支持"ms","us","s","ns"
 * @param *func   待测时的函数，必须是void X(void)的函数
 *
 * @return 返回说明
 *     函数执行时间
 */
double time_meassured(std::string time, void (*func)(void));