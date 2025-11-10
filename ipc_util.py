# -*- coding: utf-8 -*-
# @Time    : 2025/2/12 13:50
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
import os
import signal

# 两个自定义信号量, 供用户处理事件, 通过SIGUSR1 和 SIGUSR2 信号操作其值
__s1 = 0  # 对应SIGUSR1 = 10
__s2 = 0  # 对应SIGUSR2 = 12


def _signal_handler(signum, frame):
    global __s1, __s2
    if signum == signal.SIGUSR1:
        __s1 = 1 - __s1  # 在0,1之间switch
        print(f"================= Received signal {signum}, set s1 = {__s1} ==================")
    elif signum == signal.SIGUSR2:
        __s2 = 1 - __s2  # 在0,1之间switch
        print(f"================= Received signal {signum}, set s2 = {__s2} ==================")


def register_signal_handler():
    """注册用户自定义signal_handler,在接收信号时触发触发对应信号量的自定义操作"""
    signal.signal(signal.SIGUSR1, _signal_handler)
    signal.signal(signal.SIGUSR2, _signal_handler)
    print(f"================= Current process ID: {os.getpid()} =================")


def get_s1():
    """对应信号:SIGUSR1 = 10"""
    return __s1


def get_s2():
    """对应信号:SIGUSR2 = 12"""
    return __s2


def switch_s1(num):
    """对应信号:SIGUSR1 = 10"""
    assert num == 0 or num == 1
    global __s1
    __s1 = num
    print(f"=================== use func to switch s1 to {__s1} ====================")


def switch_s2(num):
    """对应信号:SIGUSR2 = 12"""
    assert num == 0 or num == 1
    global __s2
    __s2 = num
    print(f"=================== use func to switch s2 to {__s2} ====================")


if __name__ == "__main__":

    import sys
    import time

    # 必须先注册, 信号操作才会生效
    register_signal_handler()

    for i in range(50):
        get_s2() > 0 and sys.exit("SIGUSR2 is active! This process exit!")
        for j in range(10):
            time.sleep(1)
            get_s1() and print("SIGUSR1 is active!")
            print(i, j)
