import soundfile as sf
from short_SHT import short_SHT

'''
对信号进行加窗，计算SHT
输入x:原始信号，fs：采样频率，win：窗长度，a:奇异值保留比例，s，保留分量个数（如果分量数不够用0填充）
输出insf_all：瞬时频率矩阵, inse_all：对应的瞬时能量矩阵, s_list：各个窗里分量的实际个数, t：时间向量
'''

data, fs = sf.read('tire.wav')
insf_all, inse_all, s_list, t = short_SHT(data, fs, win=2000, a = 0.05, s = 20, isPlot = True)