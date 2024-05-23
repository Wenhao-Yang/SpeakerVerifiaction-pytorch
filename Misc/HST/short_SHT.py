#coding=utf8
from numpy import linalg
import numpy as np
from scipy.fftpack import fft
from scipy.signal import hilbert
import matplotlib.pyplot as plt


def fast_svd(x,a):

    # 对信号进行快速SVD分解
    # x为被分解信号，a为保留奇异值的比例
    # 返回分解后的分量矩阵m*n,m=len(x)/2*a,为分解后的分量个数，n=len(x)
    
    L = len(x)
    N = int(L/2)
    M = N + 1
    
    Signal = np.zeros((N,M))
    for i in range(N):
        Signal[i,:] = x[i:M+i]
        
    U,S,V = linalg.svd(Signal)
    
    #除去较小特征值
    V  =V
    da = int(N*a)
    Ua = U[:,0:da]
    Va = V[0:da,:]
    Sa = S[0:da]
    Sa = np.diag(Sa)
    #重构
    US = Ua[0,:]*Sa
    SV = Sa*Va[:,-1]

    Sf = np.dot(US.T,Va)
    Sb = np.dot(SV,Ua.T)

    St = np.concatenate((Sf[:,0:-1],Sb),axis = 1)
    
    return St

def get_f_gram(St,fs):

    #计算各个分量的峰值频率
    #St为分量矩阵，fs为采样频率
    #输出为峰值频率的列表，维度为分量的个数
    
    f_maxlist = []
    a = St.shape
    da = a[0]
    L = a[1]
    
    N =np.power(2,np.ceil(np.log2(L)))      # 下一个最近二次幂
    N = N.astype(int)
    f = np.arange(int(N/2))*fs/N        # 频率坐标
    
    for i in range(da):
        St_singal = St[i,:]
        y = np.abs(fft(St_singal,N))/L*2 
        y = y[range(int(N/2))]
        m = np.argmax(y,axis=0)
        f_max = f[m]
        f_maxlist.append(f_max)
        
    return f_maxlist

def hht(St,fs):

    #对信号进行hilbert变换
    #输出insf：瞬时频率矩阵；inse：瞬时能量矩阵
    z = hilbert(St)
    a = np.abs(z)
    phase = np.unwrap(np.angle(z))
    insf = (np.diff(phase)/(2.0*np.pi) * fs)
    inse = np.square(a)
    insf0 = insf[:,0]
    insf0=np.expand_dims(insf0,axis = 1)
    insf = np.concatenate((insf0,insf),axis = 1)  
    return insf,inse

def SHT(x,fs,a=0.1, isPlot = False):

    #对信号进行SHT分解
    #输出St2：分量矩阵, insf：瞬时频率矩阵，inse：瞬时能量矩阵

    St = fast_svd(x,a)
    f_maxlist = get_f_gram(St,fs)
    da = St.shape[0]
    L = len(x)
    t_end = L/fs
    t = np.arange(0, t_end, 1/fs)
    
    St2 = []
    f0 = f_maxlist[0]
    St0 = St[0,:]
    
    for i in range(da-1):
        
        if abs(f_maxlist[i]-f_maxlist[i+1])<100:
            St0 = St0 + St[i+1,:]
        else:
            St2.append(St0)
            St0 = St[i+1,:]
            
    insf, inse = hht(St2,fs)
    
#     if isPlot:
#         num = len(insf)
#         for i2 in range(num):

#             insf_draw = insf[i2,:]
#             inse_draw = inse[i2,:]/10
#             plt.scatter(t, insf_draw, marker='o', s =inse_draw, c=inse_draw, cmap='coolwarm')
#             plt.ylim(0,4000)
#             plt.xlim(0,t_end)
    
    return St2, insf, inse

def short_SHT(x, fs, win, a = 0.1, s=50, isPlot = False):

    #对信号进行加窗，计算SHT
    #输入x:原始信号，fs：采样频率，win：窗长度，a:奇异值保留比例，s，保留分量个数（如果分量数不够用0填充）
    #输出insf_all：瞬时频率矩阵, inse_all：对应的瞬时能量矩阵, s_list：各个窗里分量的实际个数, t：时间向量
    
    L = len(x)
    t_end = L/fs
    t = np.arange(0, t_end, 1/fs)
    iters = int(L/win)
    
    insf_all = np.zeros((s, L))
    inse_all = np.zeros((s, L))
    s_list = []
    
    for i in range(iters):
        
        x_short = x[win * i : win * (i + 1)]+0.0000001
        
        _, insf, inse = SHT(x_short, fs, a)
        
        s0 = len(insf)
        
        if s0>s:
            insf_all[:,win * i : win * (i + 1)] = insf[0:s]
            inse_all[:,win * i : win * (i + 1)] = inse[0:s]
        else:
            insf_all[0:s0,win * i : win * (i + 1)] = insf
            inse_all[0:s0,win * i : win * (i + 1)] = inse
        
        s_list.append(s0)
        
    if isPlot == True:
        for i2 in range(iters):
            for i3 in range(min(s,s_list[i2])):
                
                insf_draw = insf_all[i3,win * i2 : win * (i2 + 1)]
                inse_draw = inse_all[i3,win * i2 : win * (i2 + 1)]/100+0.0000001
                t_short = t[win * i2 : win * (i2 + 1)]
                plt.scatter(t_short, insf_draw, marker='o', s =inse_draw, c=inse_draw, cmap='coolwarm')
        
        plt.ylim(0,4000)
        plt.xlim(0,t_end)       
                
       
    return insf_all, inse_all, s_list, t