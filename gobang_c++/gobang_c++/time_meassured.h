#pragma once
# include <string>
/**
 * @brief ʱ���������
 * @param time    ʱ�侫���ַ�����֧��"ms","us","s","ns"
 * @param *func   ����ʱ�ĺ�����������void X(void)�ĺ���
 *
 * @return ����˵��
 *     ����ִ��ʱ��
 */
double time_meassured(std::string time, void (*func)(void));