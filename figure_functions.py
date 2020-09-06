from scipy.stats import skewnorm
from ipywidgets import interactive
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr

def convert_to_alpha(s):
    d=(np.pi/2*((abs(s)**(2/3))/(abs(s)**(2/3)+((4-np.pi)/2)**(2/3))))**0.5 
    a=((d)/((1-d**2)**.5))
    if s < 0:
        a = -a
    return a

def realign_data(y,mean,stdev):
    y = y - np.mean(y)
    y = y/np.std(y)
    y = y*stdev
    y = y+mean
    return y

def plotpdf(mean=0.0,stdev=1.0,skew=0.0,npts=250):
    x = np.linspace(-5,5,100)
    pdf = skewnorm(convert_to_alpha(skew),loc=mean,scale=stdev)
    y = pdf.pdf(x)
    data = pdf.rvs(npts)
    data = realign_data(data,mean,stdev)
    mn,var=pdf.stats(moments='mv')
    xtrans = x-mn
    xtrans = mean+(xtrans*stdev/np.sqrt(var))
    plt.figure(1,figsize=(12,5))
    plt.subplot(121)
    nbins=np.min([npts//10,50])
    (N,xbins,_)=plt.hist(data,bins=nbins)
    midx = xbins[:-1]+(np.diff(xbins)/2)
    midy = pdf.pdf(midx)
    mult=np.sum(np.diff(xbins)*N)/np.sum(np.diff(xbins*stdev/np.sqrt(var))*midy)
    plt.plot(xtrans,y*mult,'k-')
    mline=plt.axvline(np.mean(data),color='r',linestyle='--',linewidth=2,label='Mean')
    dline=plt.axvline(np.median(data),color='k',linestyle='--',linewidth=2,label='Median')
    plt.xlim(-4.5,4.5)
    plt.ylim(0,npts//8)
    plt.legend()
    plt.title('Histogram of data')
    plt.subplot(122)
    plt.plot(data[::5],'k.-')
    plt.axhline(np.mean(data),color='r',linestyle='--')
    plt.ylim(-5,5)
    plt.title('Sample of randomly-generated data\n with the given parameters')
    return

def plotquantiles(mean=0.0,stdev=1.0,npts=250,pc_anomalies=0):
    np.random.seed(42)
    xcore = np.random.normal(loc=mean,scale=stdev,size=npts)
    N,xbins=np.histogram(xcore,bins=30)
    xd = np.mean(np.diff(xbins))
    xbins = np.arange(-4,12,xd)
    if pc_anomalies > 0:
        xanom = np.random.normal(loc=mean+(6*stdev), scale=2*stdev,
                                 size=np.round(npts*pc_anomalies//100).astype(int))
        data = np.append(xcore,xanom)
    else:
        data = xcore
    plt.figure(1,figsize=(6,5))
    plt.hist(data,bins=xbins)
    mline=plt.axvline(np.mean(data),color='r',linestyle='--',linewidth=2,label='Mean')
    dline=plt.axvline(np.median(data),color='k',linestyle='--',linewidth=2,label='Median')
    mslow=plt.axvline(np.mean(data)-np.std(data),color='r',linestyle='-',linewidth=2,label='1 s.d.')
    mshi=plt.axvline(np.mean(data)+np.std(data),color='r',linestyle='-',linewidth=2)
    q3=plt.axvline(np.percentile(data,75),color='k',linestyle='-',linewidth=2,label='Q1/3')
    q1=plt.axvline(np.percentile(data,25),color='k',linestyle='-',linewidth=2)
    plt.ylim(0,npts//8)
    plt.xlim(-4,12)
    plt.legend()
    return
