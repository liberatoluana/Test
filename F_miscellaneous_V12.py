# F_miscellaneous.py>

from astropy.table import Table, Column, MaskedColumn
import glob, operator, re, math, sys, csv, os
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.signal import find_peaks
from datetime import datetime
from numpy.linalg import inv
import scipy.signal as ssig
import numpy as np
import random



###Function to fit the period found in the data accounting for the errorbars
def LSfitw(y,M,dy):

    Sigma_inv = np.diagflat(1./np.array(dy))
    #finds beta such that ||y-M*beta||*2 is minimal but with diagonal weight
    #Sigma_inv should be Sigma^^(-1/2) so iv std matrix
    Mt = np.dot(Sigma_inv,M)
    yt = np.dot(Sigma_inv,y)
    A =np.dot(Mt.T,Mt) # this and the following steps implement the LS iversion formula
    B = np.dot(Mt.T,yt)
    betaest = np.dot(np.linalg.inv(A),B)
    yest = np.dot(M,betaest)
    return yest, betaest
    


#------------------------------------------------------------------------------------------------------
##Function to get the frequency interval of search based on the lenght of the data points. Not very useful anymore.
def freq_window(body,time):
    tdif=[]

    for i in range(len(time)-1):
        diff = time[i+1] - time[i]
        tdif.append(float(diff))
    totdif=time[-1]-time[0]

    Pmax= 3*totdif
    fmin = 1.0/(Pmax)
    
#    Pmin= min(tdif)#days
    fmax = 7.99 #rot/day

    return fmin,fmax,0
#------------------------------------------------------------------------------------------------------
## Function to obtain the center and peak of observations in each 10 days window
def windows_of_search(conv):
    mid_peak,peak_val=[],[]
    flag_peak=0
    t_in=0
    t_out=0
    peak=0
    for i in range(1,len(conv)):		#iterating in the range of the windows from the convolution
        if(conv[i]>0 and flag_peak==0):		#find where the window with observations begin
            flag_peak=1
            t_in=i
        if(conv[i]>conv[i-1] and flag_peak==1):	#find the peak
            peak=conv[i]
        if(conv[i]==0 and flag_peak==1):	#find where the window with observations ends
            t_out=i
            peak_val.append(peak)		#save number of observations in this peak
            flag_peak=0
            center=((t_out-t_in)/2)+t_in	#find the center of the window
            mid_peak.append(center)		#and save it
    mid=np.array(mid_peak, dtype=object)	
    peak=np.array(peak_val, dtype=object)
    return mid,peak				#return the arrays with the center and the heights of the windows found with data
 
#------------------------------------------------------------------------------------------------------ 
### Function to find the windows of search in the data set
def get_sample_window(t,body,path):
    twindow = 10.0 # desired lenght of the search window (in days)

    t_start=min(t)                # first date in the array
    t_range = max(t)-t_start      # range of dates
    signal = np.zeros(int((t_range+2.)*100)) 	# increase resolution to 1/100. of a day, with 2 extra days added for the extremes
                                               # in "signal" all values are zero now
    for tt in t:           # populate with "ones" the time values in sigma that are approx. those in res_transit_ref['day']
        it = int((tt-(t_start-1.))*100)
        signal[it]=1.0
        
    t_filter = np.ones(int(twindow*100))     # window to search for data points
    t_conv = twindow*100*ssig.convolve(signal,t_filter,method='direct')/sum(t_filter)   # convolve the window with the signal
  
    mid,peak=windows_of_search(t_conv)   #call function to get the center and the peak for each window
    
    x=np.linspace(0,len(t_conv),len(t_conv))  #create interval between zero and the lenght of the length of the convolution function lenght
    fig=plt.figure()  #create new figure
    plt.xlabel("Time between observations (days)") #add label to x axis
    plt.ylabel("Number of data points")	#add label to y axis
    plt.title(str("Body {} - Density of observations - \n Windows of {} days ".format(body,twindow)))	#add title
    plt.plot(x/100,t_conv)		#plot the convolution function over the time
    plt.scatter(mid/100,peak,s=2,color='black')		#plot location of the peak and center of the windows found
    plt.savefig(str("{}/Results/{}/{}_data_density.jpeg".format(path,body,body)), dpi=150)		#save figure
    plt.close(fig)	#close figure
    time_end=((mid)/100)+twindow+t_start	# get the final time of the window of search
    time_beg=((mid)/100)-twindow+t_start	#get the initial time of the window of search
    cont=0					#initialize count of preferencial windows of search
    for i in range(len(peak)):			#for each peak found
        if (peak[i]>=10): cont+=1		#check if the number of data points is larger than 10
						#if so, add one to the count of windows of search that will be used
    return time_beg,time_end,peak,cont		#return the times, the peaks and the number of windows


#------------------------------------------------------------------------------------------------------
##Function to check the distribution of data points and clean the outliers. It finds the data points with values and/or errors that are beyond 3 sigma from the distribution.
def check_outliers(res,dres,body):
    res_mean=np.mean(res)
    res_std=np.std(res)
    dres_mean=np.mean(dres)
    dres_std=np.std(dres)
    cont=0
    idx=[]
    for i in range(len(res)):		#find and save the indexes of the outliers beyond 3-sigma
        if ((np.abs(res[i]-res_mean)>np.abs(3*res_std)) or (np.abs(dres[i]-dres_mean)>np.abs(3*dres_std))):
            idx=np.append(idx,int(i))
            cont+=1

        
    if (cont>0):
        f=open("Clean_sample.out","a")
        f.write("{} lost {} points\n".format(body,cont))
        f.close()    
    idx=np.array(idx)
    return idx.astype('int')		#return outliers' indexes


#------------------------------------------------------------------------------------------------------
##Function to read the input data from files.

def real_data(filename):		
    data = Table.read(filename,format='csv')
    t=np.array(data['epoch'])
    res=np.array(data['Res_AL_aver(mas)'])
    dres=np.array(data['Res_AL_std(mas)'])
    fov=np.array(data['FOV'])
    mag=np.array(data['G'])
    return t,res,dres,fov,mag
    
#------------------------------------------------------------------------------------------------------
## Function to obtain the indexes of the data points from the window of search that will be used.
def get_idx(t0,tf,t): #t0 and tf are the initial and final times of the window, and t is the time array
    idx=[]
    for i in range(len(t)):
        if(t[i]>=t0 and t[i]<=tf):
            idx.append(i)
    return idx        #return indexes for the data points in the current window of search
#------------------------------------------------------------------------------------------------------                        
##Function to test the existence of a file. 
##If it doesn't exist, create a new one. If if does exist, open in append mode.
def test_existence(filename):

    if not(os.path.exists(filename)):
        return open(filename, "w+")
    else:
        return open(filename, "a")
        
#------------------------------------------------------------------------------------------------------        
##Function to calculate the SNR of a fit with respect to the distribution of error in the data points.
def SNR_fit(y_fit,dy):
    A=np.abs(max(y_fit)-min(y_fit))/2
    sigma=np.mean(dy)
    SNR=A/sigma
    return SNR
    
#------------------------------------------------------------------------------------------------------    
#Function to obtain the highest power, the best frequency from the highest power and clean the periodogram from any spurious peaks
def get_best_frequency(y,freq):		
    peaks, properties = find_peaks(y, width=3)	#find all peaks with width larger than 3 points
    #check for each peak found if the the values before and after are too different and replace them by the same value right before or after the anomaly
    for i in range(len(peaks)):			
        if(peaks[i]==0):
            y[peaks[i]]=y[peaks[i]+1]
        elif(peaks[i]==len(y)-1):
            dif_b=y[peaks[i]]/y[peaks[i]-1]
            if(dif_b>1.05):
                y[peaks[i]]=y[peaks[i]-1]
        else:
            dif_b=y[peaks[i]]/y[peaks[i]-1]
            dif_a=y[peaks[i]]/y[peaks[i]+1]
            if(dif_a>1.05):
                y[peaks[i]]=y[peaks[i]-1]
            if(dif_b>1.05):
                y[peaks[i]]=y[peaks[i]+1]
    #sort the peaks in crescent order and get the last value, which is the largest
    peak_y_order=np.sort(y[peaks])
    peak_x_order=np.argsort(y[peaks])
    return y,freq[peaks[peak_x_order][-1]],peak_y_order[-1]  #return the clean powers, the best frequency and the highest power
#------------------------------------------------------------------------------------------------------    
"""
def coverage(y_fit,y,phase):
    A=np.abs(max(y_fit)-min(y_fit))
    delta_obs=np.abs(max(y)-min(y))
    delta_phase=np.abs(max(phase)-min(phase))
    C=A*delta_phase/delta_obs
    return C
""" 
