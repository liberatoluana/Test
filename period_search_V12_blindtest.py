#!/usr/bin/env python

import glob, operator, re, math, sys, csv, os

from astropy.table import Table, Column, MaskedColumn
from astropy.timeseries import LombScargle
from astropy.time import Time

from scipy.stats import norm, invgauss, rayleigh,skewnorm
from scipy.signal import find_peaks
import scipy.signal as ssig

from matplotlib.patches import Rectangle
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import matplotlib.cbook

from datetime import datetime
import multiprocessing as mp
import numpy as np
import statistics 
import pathlib
import random

from F_MC_Quality_check_David_V12_normal import *
from F_MC_Noise_check_V12 import *
from F_miscellaneous_V12 import *
#from F_Fake_signal import *


###Main function to perform the period search

def Run_main(m):

    body=number[m]
    print('{}\n'.format(body))
    os.system(str("mkdir -p Results/{}".format(body)))     #create a new folder for this object, if it doesn't exist already
#####
  #  t,res,dres,fovs,snr=Generate_signal(path,body)    #obtaining data from an artificial signal.
####

    path_gaia_files=str("../files_blind_test/{}.gaia".format(body)) #full path to the file containing the residuals for each object.

#####
#    path_gaia_files=str("{}/gaia_files/{}.gaia".format(path,body))   #Use this if the folder with the gaia files is in the same directory as you're running the script
#####

    t,res,dres,fovs,mag=real_data(path_gaia_files)    #obtain real data from files

#-------------------------------------------------------------------------------------------------------------           
    #Getting the data set
    t0,tf,peak,n_windows=get_sample_window(t,body,path)		#find the windows with data for period search
    for w in range(len(t0)):					#run the period search in each window found
        if (peak[w] >=10.0):					#if the number of points in current window is >=10, then run the search
            time1, y1, dy1,index,fov1=[],[],[],[],[]
            time, y, dy,fov=[],[],[],[]
            index=get_idx(t0[w],tf[w],t)			#get indexes of the data points inside the window
            for a in range(len(index)):				#get the arrays with the data for the current window
                b=int(index[a])	
#                print(t[b],res[b],dres[b])
                time1.append(t[b])				#time array
                y1.append(res[b])				#residual array
                dy1.append(dres[b])				#error bars array
                fov1.append(fovs[b])				#field of view array
            
            outliers=check_outliers(y1,dy1,body)		#check the data in the window for outliers
 #           print(body,outliers,'\n')
            if(len(outliers)!=0):				#if there is any outlier, delete it from the data set
                time=np.delete(time1,outliers)
                y=np.delete(y1,outliers)
                dy=np.delete(dy1,outliers)
                fov=np.delete(fov1,outliers)
            else:						#if not, just rename for the arrays that will be used
                time=time1
                y=y1
                dy=dy1
                fov=fov1
                
            if(len(time)<10):					#if the outlier removal leaves a sample of less than 10 points, skip this window.
            	break
#-------------------------------------------------------------------------------------------------------------                           
            ######
            #separate the datapoints in FOV1 and FOV2, for plotting later.
            y_fov1,y_fov2,t_fov1,t_fov2,dy_fov1,dy_fov2=[],[],[],[],[],[]
            for i, item in enumerate(time):			
                t[i] = t[i]-t0[w]
                if (fov[i]==1): 
                    y_fov1.append(y[i])
                    t_fov1.append(time[i])
                    dy_fov1.append(dy[i])
                    
                else:
                    y_fov2.append(y[i])
                    t_fov2.append(time[i])
                    dy_fov2.append(dy[i])
#-------------------------------------------------------------------------------------------------------------           
            #######
            ###plot the sample of data used in the search in the current window, separated by FOV 1 and 2
            fig=plt.figure()
            plt.errorbar(t_fov1,y_fov1,yerr=dy_fov1,fmt='o',color='blue',label='FOV 1')
            plt.errorbar(t_fov2,y_fov2,yerr=dy_fov2,fmt='o',color='darkgreen',label='FOV 2')            
            plt.title(str("Body {} - Sample of data selected for analysis\nWindow {} from {:.2f} to {:.2f} Gaia date ({:.2f} days)\n {} observations".format(body,w,time[0],time[-1],time[-1]-time[0],len(time))))
            plt.xlabel('Observation date (days)')
            plt.ylabel("Average AL residuals (mas)")
            plt.legend(loc='upper right')
            plt.savefig(str("{}/Results/{}/{}_window-{}_Sample.jpeg".format(path,body,body,w)), dpi=150)
            plt.close(fig)
 #-------------------------------------------------------------------------------------------------------------                      
            #######
            #FIRST PERIOD SEARCH
            
            fmin,fmax,estim=freq_window(body,time)		#get the limits of frequencies that will be explored

            freq = np.linspace(fmin, fmax, 10000)		#create a range of frequencies
            
            ls=LombScargle(time,y,dy,fit_mean=True)	#Pass the data to the Lomb-Scargle. Fit_mean=True is the default.
            powers=ls.power(freq,method='cython')	#Run the search within the range of frequencies given. Method 'cython' used because it is a similar implementation as method 'slow', but much faster.
            power,best_frequency,best_power = get_best_frequency(powers,freq) #get highest power and best frequency from periodogram and clean periodogram from spurious outliers
            ########
#-------------------------------------------------------------------------------------------------------------           
            ######
            #plot the first periodogram
            fig=plt.figure()
            plt.plot(freq,power,color='red',linestyle='solid',label='Data')
            plt.scatter(best_frequency,best_power,c='black')
            plt.ylabel('GLS Power')
            plt.grid(True,which="both")
            plt.xlabel('Frequency (rotations/day)')
            plt.savefig(str("{}/Results/{}/{}_First_GLS_freq_{}_window_.jpeg".format(path,body,body,w)), dpi=150)    
            plt.close(fig)            
            #######
#-------------------------------------------------------------------------------------------------------------                       
            ###### Try to do the rest of the steps and rise error if something goes wrong. Useful to do not stop the program if there is an issue with some of the objects.
            try:
                un = np.ones(np.shape(time))
                M1 = (np.concatenate(([un], [np.cos(2*np.pi*np.array(time)*best_frequency)], [np.sin(2*np.pi*np.array(time)*best_frequency)]), axis=0)).T
                y_fit,betaest = LSfitw(np.array(y),M1,np.array(dy))			#find the amplitude of the sinusoid that best fit this set of data
                x_time=np.array(np.linspace(0,1/best_frequency,1000))
                y_amp=(betaest[1]*np.cos(2*np.pi*x_time*best_frequency)+betaest[2]*np.sin(2*np.pi*x_time*best_frequency))+betaest[0]
                wob=np.abs((np.max(y_amp)-np.min(y_amp))/2)			#calculate the amplitude of the signal and store it
                MC_pval=check_MC(time,dy,wob,fmin,fmax,best_power,w,body,path)	#run MC on noise to obtain p-value
                snr=SNR_fit(y_amp,dy)
                if (MC_pval>10 and n_windows==1):				#If there is just one window of search and the p-value from the MC on noise is already larger than 5, then skip this body, but save this info in log file.
                    log.write(str("{}\t{}\t{}\t{:.5f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(body,time[0],time[-1],24/best_frequency,wob,MC_pval,0.0,snr,0.0,0.0,0.0,0.0)))
                    log.flush()   
                    continue
                else:								#But if the p-value is smaller than 5, then run the rest of the program.

                    period = 1.0/best_frequency					#get the first estimate on the best period in unit of days

                    phase_fov1=(t_fov1/ period) % 1				#get the phase for each data point if FOV 1
                    phase_fov2=(t_fov2/ period) % 1                       	#and in FOV 2
                    
                    ls2=LombScargle(time,y_fit,dy,fit_mean=True)		#run the GLS for the sinusoid from the fit with the first period estimated 
                    power2=ls2.power(freq,method='cython')			#using the same methods and GLS parameters as previously
                    

                    y_fit3=np.ones(len(time))					#create an array full of ones to use as the amplitudo of the spectral window
                    
                    ls3=LombScargle(time,y_fit3,fit_mean=False, center_data=False)	#and run the GLS fro the spectral window, but now not centering the data neither fitting the mean.
                    power3=ls3.power(freq,method='cython')
                    #-------------------------------------------------------------------------------------------------------------
                    ########
                    #Plot the periodogram obtained as a funcition of the period 
                    fig2, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]},sharex=True)
                    plt.subplots_adjust(hspace = .001)
                    ax1.set_title(str("Checking the Periods for body {} \nWindow {} from {:.2f} to {:.2f} Gaia date ({:.2f} days)".format(body,w,time[0],time[-1],time[-1]-time[0])))
                    ax1.plot(24/freq,power,color='red',linestyle='solid',label='Data')
                    ax1.scatter(24/best_frequency,best_power,color='black',marker='o',label='best value')
                    ax1.set_ylabel('GLS Power')
                    ax1.set_xscale('log')
                    ax1.grid(True,which="both")
                    ax1.legend(loc=0)
                    ax2.set_xscale('log')
                    ax2.plot(24/freq,power3,color='green',linestyle='solid',label='Spectral Window')
                    ax2.plot(24/freq,power2,color='blue',linestyle='solid',label='GLS of the main sinusoid')
                    ax2.grid(True,which="both")                                
                    ax2.set_xlabel('Period (hours)')
                    ax2.legend(loc=0)
                    plt.savefig(str("{}/Results/{}/{}_window-{}_GLS_period.jpeg".format(path,body,body,w)), dpi=150)    
                    plt.close(fig2)
                    
                    ########
                    #Plot the periodogram obtained as a funcition of the frequency
                   
                    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]},sharex=True)
                    plt.subplots_adjust(hspace = .001)
                    ax1.set_title(str("Checking the Frequencies for body {} \nWindow {} from {:.2f} to {:.2f} Gaia date ({:.2f} days)".format(body,w,time[0],time[-1],time[-1]-time[0])))
                    ax1.plot(freq,power,color='red',linestyle='solid',label='Data')
                    ax1.set_ylabel('GLS Power')
                    ax1.scatter(best_frequency,best_power,color='black',marker='o',label='best value')
                    ax1.grid(True,which="both")
                    ax1.legend(loc=0)
                    ax2.plot(freq, power3, color='green', linestyle='solid', label='Spectral Window')
                    ax2.plot(freq, power2, color='blue', linestyle='solid', label='GLS of the main sinusoid')
                    ax2.grid(True,which="both")                                
                    ax2.set_xlabel('Frequency (rotations/day)')
                    ax2.legend(loc=0)
                    plt.savefig(str("{}/Results/{}/{}_window-{}_GLS_freq.jpeg".format(path,body,body,w)), dpi=150)    
                    plt.close(fig)
                    #-------------------------------------------------------------------------------------------------------------
                    if(MC_pval<10):
                        print(body," Best Frequency: ",best_frequency, " p-value=",MC_pval)
                        qual,p_fit,std_p,ci_p,amp,std_amp,ci_amp=check_quality(time,wob,dy,fmin,fmax,best_frequency,w,body,path) 	#Run the quality check on the fit and save the output to the log file
                        print(body," quality: ",qual, " amp=",amp)                        
                        peri=p_fit/24
                        std_peri=std_p/24
                        ci_peri=ci_p/24

                        ###Get the fit of the data points based on the best period found in the MC quality distribution                   
                        xtime=np.array(np.linspace(0,peri,1000))
                        phasefit = (xtime / peri)  
                        M2 = (np.concatenate(([un], [np.cos(2*np.pi*np.array(time)/peri)], [np.sin(2*np.pi*np.array(time)/peri)]), axis=0)).T
                        y_fit2,betaest2 = LSfitw(np.array(y),M2,np.array(dy))			#find the amplitude of the sinusoid that best fit this set of data
                        y_fit_all=(betaest2[1]*np.cos(2*np.pi*xtime/peri)+betaest2[2]*np.sin(2*np.pi*xtime/peri))+betaest2[0]
                        
                        phase2_fov1=(t_fov1/ peri) % 1				#get the phase for each data point if FOV 1
                        phase2_fov2=(t_fov2/ peri) % 1                       	#and in FOV 2

                        snr=SNR_fit(y_fit_all,dy)					#get the SNR of the fit with respect to the data points distribution

					  
                        #-------------------------------------------------------------------------------------------------------------
                        #plot the data fit
                        fig = plt.figure()
                        plt.title(str("Body {} Astrometry residuals\nPeriod fitted:{:.2f}+-{:.2f} hours\n Amplitude={:.2f}+-{:.2f}mas | SNR={:.2f}\np-value = {:.3f}%").format(body,24*peri,24*std_peri,amp,std_amp,snr,MC_pval),pad=10)
                        plt.errorbar(phase2_fov1,y_fov1,yerr=dy_fov1,color='blue',fmt='o',label='FOV 1')
                        plt.errorbar(phase2_fov2,y_fov2,yerr=dy_fov2,color='darkgreen',fmt='o',label='FOV 2')
                        plt.plot(phasefit, y_fit_all,color='darkorange',zorder=2,label='Fit')
                        plt.subplots_adjust(top=0.8) 
                        plt.legend(loc='upper right')
                        plt.xlabel('Phase')
                        plt.ylabel('Average AL residuals (mas)')
                        plt.savefig(str("{}/Results/{}/{}_window-{}_Fit_data.jpeg".format(path,body,body,w)), dpi=150)
                        plt.close(fig)
		           #-------------------------------------------------------------------------------------------------------------
                        
                                                                      
                        #If the p-value is smaller than 5%, the quality factor is larger than 4, the wobbling is smaller than 15 mas (to exclude cases with wrong and too large fits), and the best frequency is not at the limit of the interval (which would mean that the search has failed), then it is selected as a cadidate and saved in the file of best cases.
                        if(float(MC_pval)<=5.0 and best_frequency>fmin and best_frequency<fmax and wob < 20 and qual > 0.5):	
                            best.write(str("{}\t{}\t{}\t{:.5f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(body,time[0],time[-1],24*peri,amp,MC_pval,qual,snr,24*std_peri,std_amp,24*ci_peri,ci_amp)))
                            best.flush()
                            os.system(str("cp -R Results/{} Selected/".format(body)))
                        else:
                            log.write(str("{}\t{}\t{}\t{:.5f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(body,time[0],time[-1],peri*24,amp,MC_pval,qual,snr,std_peri*24,std_amp,24*ci_peri,ci_amp)))
                            log.flush() 
                    else:
                        log.write(str("{}\t{}\t{}\t{:.5f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(body,time[0],time[-1],24/best_frequency,wob,MC_pval,0.0,snr,0.0,0.0,0.0,0.0)))
                        log.flush()        
	#-------------------------------------------------------------------------------------------------------------   
                     
            except ValueError:  #raised usually if `power` is empty, or in more rare cases if something went wrong with the quality check.
                print("Problems to perform period search in {} data set\n".format(body))
                pass
        
        plt.close("all")
        log.flush()
        best.flush()
    search_control.write("{}\n".format(body))		#write in the control file that the period search has finished for the current object
    search_control.flush()
    
    return str("{} Done!\n".format(body))



#-------------------------------------------------------------------------------------------------------------


path=pathlib.Path(__file__).parent.resolve()		#get path to the current directory where the code is running

number,bodies=[],[]

list_bodies=open("Bodies_gaia.in","r")			#get list of object that will be explored
for j in list_bodies:
    bodies.append(int(j))


############
# Check if the file that logs the objects already explored exists. If it doesn't exist, create a new file. 
#But if if exists, read list of objects already done and remove them from the list of objects that will be explored
control=str(path)+str("/Search_control.con")
if (os.path.exists(control)==False or os.stat(control).st_size == 0):
    finish=open(control,"w+")
    finish.close()
    number=bodies
else:
    fp=open(control, "r")
    read=fp.readlines()
    fp.close()
    
    
    for i in range(len(bodies)):
        flag=0
        for j in range(len(read)):
            if(int(bodies[i])==int(read[j])):
                flag=1
        if(flag==0):
            number.append(int(bodies[i]))


######        
        
        
search_control=open(str("Search_control.con"),"a+")	#open control file in append mode

###
#call function to test existence of the output files
best=test_existence("Best_periods.out")			
log=test_existence("Log_output_rest.out")		
large_wob=test_existence("Large_wobbling_bodies.out")
clear=test_existence("Clean_sample.out")
###


os.system(str("mkdir -p Results Selected Large_Wobbling"))	#create folders for the results, in case it doesn't exist already

####
#initiating the period search routine for 10 different objects in parallel
"""
for m in range(len(number)):
	Run_main(m)
"""
m=np.arange(len(number))
pool = mp.Pool(processes=25)       #If you want to change the number 10 of parallel runs, change the parameter "processes" to the value you desire
results = pool.map(Run_main, m)	   #Call main function
####

#close files used 
search_control.close()        
best.close()
log.close()
large_wob.close()

#run the bash script to organize the results in a pdf
#os.system("./Organize_figures_latex.sh")

