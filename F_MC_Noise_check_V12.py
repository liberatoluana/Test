# F_MC_Quality_check.py

###Function to perform MC simulations and check the p-value probability of the fit given by GLS for the frequency of the highest peak. It checks the probability of the signal found being created only due to noise.

from scipy.stats import norm, invgauss, rayleigh,skewnorm
from astropy.timeseries import LombScargle
from matplotlib.patches import Rectangle
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import numpy as np
import random

def check_MC(t,dy,wob,f_min,f_max,power_max,w,body,path):
    high_power,power_freqs=[],[]        #array to keep the highest power and best frequencies in each MC simulation
    dy=np.array(dy) 			#making sure that dy is in a numpy array
    frequency = np.linspace(f_min, f_max, 10000)  #creating the range of frequencies that will be explored
    N=10000				#Number of MC simulations
    flag=0   		 #flag to know when to stop counting the p-value
    pval=100  		 #initializing the probability as 100% of false alarm


    for i in range(N):            #Starts the MC simulations
        y_boot=[]                #create one new array of values of y for each MC
        random.seed()		 #get a new seed for random function
        shift=wob*np.random.laplace(0,0.27/np.sqrt(2))
        for k in range(len(dy)):
            y=shift + (dy[k]*np.random.normal(0,1))  #estimate the amplitude of the noise that will be used as signal. Get a random normal distribution around 0 with std=1 and scale with the original error bars
            y_boot=np.append(y_boot,y)	

        ls = LombScargle(t,y_boot,dy,fit_mean=True,nterms=1)  #set the LS with the signal. The real time sampling, the real error bars and the amplitudes generated from the sinusoid with noise.
        power=ls.power(frequency,method='cython')	#Run the periodogram in the chosen range of frequencies
        high_power.append(np.max(power))		#find and save the highest power in each MC simulation
        power_freqs.append(frequency[np.argmax(power)])	#find and save the frequency from the highest power in each MC simulation


    x = np.linspace(0,1,1000)		#create a linear range of powers from 0 to 1

    pval_precision=2			#precision of 2 decimals
    L=100*(10**pval_precision)		#calculate number of bins that will be used to check the p-value
    for i in range(L):			#iterating in each bin
        perc=(np.percentile(high_power,[i/100]))	#i/100 represents each bin in the percentile range and high_power is the distribution of data. This function gives you where in the distribution you find the percentile you input. The smaller the bin, the higher is the precision.
        if (power_max<=perc):		#when the location of the percentile becomes larger than the value of power from the best period found in the period search, we save it
            pval=100.0-(i/100.0)			#The p-value will be 100% minus the current percentile				
            flag=1					#flag to know we can stop counting
            break					#and stop the counting
        if(i==N-1 and flag==0):				#but if it is at the last bin and still haven't stopped counting
            pval=0.000					#then set the p-value to 0
        
    fig, ax = plt.subplots(figsize=(6, 3))		#create new figure
    plt.title("MC test on noise")			#add title
    plt.tick_params(left = False,labelleft = False)    	#disable ticks and label on the left 
    n,b,xxx=ax.hist(high_power,density=True,histtype='stepfilled', alpha=0.2,bins=100,color='green',range=[0,1])	#plot distribution histogram
    plt.xlabel("Highest power value")			#add label to x axis
    plt.plot([power_max,power_max], [0,max(n)], 'k--', lw=1, label='max power location\n with P-value={:.3f}%'.format(pval))	#plot location of highest power from best period
    fig.tight_layout()					#adjust tightness of plot
    plt.legend(loc=0)
    plt.savefig(str("{}/Results/{}/{}_window-{}_MC_BS.jpeg".format(path,body,body,w)), dpi=150)		#save figure
    plt.close(fig)					#close figure
    
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.suptitle("")
    plt.tick_params(left = False,labelleft = False)    
    n,bins,patches=ax.hist(power_freqs,density=True,histtype='stepfilled', alpha=0.2,bins=100,color='green')    
    ax.set_xlabel("Frequency of highest power")
    fig.tight_layout()
    fig.tight_layout()
    plt.savefig(str("{}/Results/{}/{}_window-{}_MC_BC_freq.jpeg".format(path,body,body,w)), dpi=150)
    plt.close(fig)
    """
    
    return pval		#return the p-value

