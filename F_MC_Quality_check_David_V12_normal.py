# F_MC_Quality_check_David.py

###Function to perform MC simulations and check the quality of the signal privided


from scipy.stats import norm, invgauss, rayleigh,skewnorm
from astropy.timeseries import LombScargle
from matplotlib.patches import Rectangle
from F_miscellaneous_V12 import *
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import numpy as np
import random 
    
def get_std(n,xbins):

    f1,f2,f3,f4,f5=0,0,0,0,0
    s=0
    for k in range(len(n)):
       s=s+n[k]
       per=100*s/sum(n)
       if(per>=2.5 and f1==0):
           lmin_95=(xbins[k+1]+xbins[k])/2
           f1=1
       if(per>=15.7 and f2==0):
           lmin=(xbins[k+1]+xbins[k])/2
           f2=1
       if(per>=50 and f3==0):
           lmid=(xbins[k+1]+xbins[k])/2
           f3=1          
       if(per>=84.3 and f4==0):
           lmax=(xbins[k+1]+xbins[k])/2
           f4=1
       if(per>=97.5 and f5==0):
           lmax_95=(xbins[k+1]+xbins[k])/2
           f5=1
    mu=lmid
    std_max=lmax-lmid
    std_min=-(lmid-lmin)
    std=np.max([std_max,std_min])
    CI=np.max([lmax_95-lmid,np.abs(lmid-lmin_95)])

    return mu, std,CI
###----------------------------------------------------------------------------------

def check_quality(time,wob_amp,dy,f_min,f_max,freq_best,w,body,path):  ##arguments are the time of observations, the first signal amplitude estimated, the error bars, the range of frequencies for the period search, the first best frequency estimated, the number of the window of search, the ID of the object and the path to the folder in which the plots will be saved

    MC_freqs,amp,MC_peris=[],[],[]
    Aest=[]	
    dy=np.array(dy)		#making sure that the error bars are in a numpy array
    N=5000			# Number of MC simulations
    frequency = np.linspace(f_min, f_max, 10000)  #creating the range of frequencies that will be explored
    t=np.array(time)
    un = np.ones(np.shape(t))
    for i in range(N):	#For each MC simulation do
        y_err=[]	#set a new list of amplitudes
        random.seed()	#get a new seed for random function
        phase=2*np.pi*random.random()	#get a random phase
        shift=float(wob_amp)*np.random.laplace(0,0.27/np.sqrt(2))
        sig=float(wob_amp)*np.cos(phase+(2*np.pi*freq_best*t))		#create the sinusoid based on the first best frequency found
        y=sig+shift								#make the amplitude of the sinusoid compatible with the first amplitude estimated for the signal
        for k in range(len(dy)):	#for each data point
            y_err=np.append(y_err,dy[k]*np.random.normal(0,1))	#estimate the noise that will be added to the sinusoidal signal. Get a random normal distribution around 0 with std=1 and scale with the original error bars
        y_boot=np.add(y,y_err)		#add the noise to the signal
        ls=LombScargle(t,y_boot,dy,fit_mean=True,nterms=1)	#set the LS with the signal. The real time sampling, the real error bars and the amplitudes generated from the sinusoid with noise.
        powers=ls.power(frequency,method='cython')	#Run the periodogram in the chosen range of frequencies
        freq_power=float(frequency[np.argmax(powers)])         #find and save the best frequency in each MC simulation
        MC_freqs=np.append(MC_freqs,freq_power)     
        MC_peris=np.append(MC_peris,24/freq_power) 
        M1 = (np.concatenate(([un], [np.cos(2*np.pi*t*frequency[np.argmax(powers)])], [np.sin(2*np.pi*t*frequency[np.argmax(powers)])]), axis=0)).T
        yest,betaest = LSfitw(y_boot,M1,dy)			#find the amplitude of the sinusoid that best fit this set of data
        xtime=np.array(np.linspace(0,1/freq_power,100))
        y_fit_all=(betaest[1]*np.cos(2*np.pi*xtime*freq_power)+betaest[2]*np.sin(2*np.pi*xtime*freq_power))+betaest[0]
        amp=np.append(amp,(np.max(y_fit_all)-np.min(y_fit_all))/2)		#calculate the amplitude of the signal and store it
    #--------------------------------------  
    

    fig, ax = plt.subplots(figsize=(6, 3))		#create new figure
    fig.suptitle("MC test on noise + data")		#add title
    n,bins,patches=ax.hist(MC_freqs,density=True,histtype='stepfilled', alpha=0.6,bins=100,range=[f_min,f_max])    	#plot distribution of frequencies from MC and store the info
    index=np.searchsorted(bins,freq_best)-1	#find the bin that matches with the best frequency found previously
    if(index<0):index=0
    sort=np.argsort(n)				#sort the distribution in crescent order
    width=bins[index]-bins[index-1]		#get the width of the bins	
    q_a=np.where((MC_freqs>freq_best-0.15) & (MC_freqs<freq_best+0.15))[0]
    Q=len(q_a)/len(MC_peris)

    plt.tick_params(left = False,labelleft = False)    #disable left ticks and label
    ax.set_xlabel("Frequency of highest power")		#set x axis label
    ax.plot([freq_best,freq_best], [0,max(n)], 'k--', lw=0.8)	#plot location of best frequency
    ax.add_patch( Rectangle((bins[index]-width,0),0.25,n[index]*1.05,fc ='none',ec ='red',lw =0.2,label='quality={:.2f}'.format(Q)))	#plot red rectangle around best frequency
    ax.legend(loc=0,borderaxespad=0.)	#set location of the legend
    fig.tight_layout()		#set plot tightness
    plt.savefig(str("{}/Results/{}/{}_window-{}_MC_quality.jpeg".format(path,body,body,w)), dpi=150)	#save figure
    plt.close(fig)		#close figure
    
    
    #------------------------------------
    
    rmin=wob_amp-2
    rmax=2+wob_amp
    if(rmin<0):	rmin=0
    binwidth=0.01
    bins=np.arange(rmin, rmax + binwidth, binwidth)
    fig, ax = plt.subplots(figsize=(6, 3))		#create new figure

    n,xbins,xxx=ax.hist(amp,density=True,histtype='stepfilled', alpha=0.4,bins=bins,range=[rmin,rmax])    #plot an histogram of the distribution of amplitudes
   
    mu,std,ci=get_std(n,xbins)
    fig.suptitle("MC test on noise + data\n" r"Amplitude={:.2f}$\pm${:.2f} | 95% CI={:.2f}$\pm${:.2f}mas ".format(mu,std,mu,ci))		#add title
    ax.plot([mu,mu], [0,max(n)*1.02], 'k--', lw=0.8)	#plot the location of the center of the distribution 
    ax.plot([mu-std,mu-std], [0,max(n)*0.95], color='black',ls=':', lw=0.8)	#plot the location of -sigma 
    ax.plot([mu+std,mu+std], [0,max(n)*0.95], color='black',ls=':',lw=0.8)	#plot the location of +sigma 
    plt.plot([mu-std,mu+std],[np.max(n),np.max(n)],lw=0.5,color='black')
    plt.text(mu,np.max(n)*1.05,r'$\sigma$')
    ax.plot([wob_amp,wob_amp], [0,max(n)*0.9], color='red', lw=0.8)	#plot location of best period
    plt.tick_params(left = False,labelleft = False)    	#disable left ticks and label
    ax.set_xlabel("Amplitude distribution (mas)")		#add label to x axis
    plt.ylim(0,np.max(n)*1.2)
    plt.xlim(rmin,rmax)
    fig.tight_layout()					#fix tightness of the plot
    plt.savefig(str("{}/Results/{}/{}_window-{}_MC_amp.jpeg".format(path,body,body,w)), dpi=150) #save the figure
    plt.close(fig)					#close the figure

    #------------------------------------
    """   
    f_best=freq_best					#get the bes period(in hours) from the best frequency
    mid=np.median(MC_freqs)
    std=np.std(MC_freqs)
    rmin=mid-5*std
    rmax=mid+5*std
    if(rmin<0):	rmin=0
    if(rmax>f_max): rmax=f_max
 
    f=np.array(MC_freqs[np.where((MC_freqs>rmin) & (MC_freqs<rmax))[0]])				#get the distribution of MC periods (in hours) from the MC frequencies
   #------------------------------------

    fig, ax = plt.subplots(figsize=(6, 3))		#create new figure
    binwidth=0.001
    bins=np.arange(np.min(f),np.max(f) + binwidth, binwidth)
    n,xbins,patches=ax.hist(f,density=True,histtype='stepfilled', alpha=0.4,bins=bins)    #plot histogram of the distribution around best period

    mu_f,std_f,ci_f=get_std(n,xbins)

    ax.plot([mu_f,mu_f], [0,max(n)*1.05], 'k--', lw=0.8)	#plot the location of the center of the distribution 
    ax.plot([mu_f-std_f,mu_f-std_f], [0,max(n)*0.95], color='black',ls=':', lw=0.8)	#plot the location of -sigma 
    ax.plot([mu_f+std_f,mu_f+std_f], [0,max(n)*0.95], color='black',ls=':', lw=0.8)	#plot the location of +sigma 
    fig.suptitle("MC test on noise + data\n" r"Frequency={:.2f}$\pm${:.2f} | 95% CI={:.2f}$\pm${:.2f} cycles/day".format(mu_f,std_f,mu_f,ci_f))		#add title
    plt.plot([mu_f-std_f,mu_f+std_f],[np.max(n),np.max(n)],lw=0.5,color='black')
    plt.text(mu_f,np.max(n)*1.05,r'$\sigma$')
    plt.ylim(0,np.max(n)*1.2)
    plt.xlim(rmin,rmax)
    plt.tick_params(left = False,labelleft = False)    	#disable left ticks and label
    ax.set_xlabel("Frequency distribution")		#add x axis label
    ax.plot([f_best,f_best], [0,max(n)], color='red', lw=0.8)	#plot location of best period
    fig.tight_layout()					#adjust plot tightness
    plt.savefig(str("{}/Results/{}/{}_window-{}_MC_freq.jpeg".format(path,body,body,w)), dpi=150) #save figure
    plt.close(fig)					#close figure
    """
   #------------------------------------
    peri_best=24/freq_best					#get the bes period(in hours) from the best frequency
    rmin=24/(freq_best+0.25)
    rmax=24/(freq_best-0.25)
    if(rmin<0):	rmin=0
    if(rmax>24/f_min or rmax<0): rmax=24/f_min
 
    f=np.array(MC_peris[np.where((MC_peris>rmin) & (MC_peris<rmax))[0]])				#get the distribution of MC periods (in hours) from the MC frequencies


    fig, ax = plt.subplots(figsize=(6, 3))		#create new figure
  #  binwidth=0.1
  #  bins=np.arange(np.min(f),np.max(f) + binwidth, binwidth)
    n,xbins,patches=ax.hist(f,density=True,histtype='stepfilled', alpha=0.8,bins=1000, range=[rmin,rmax])    #plot histogram of the distribution around best period

    mu_p,std_p,ci_p=get_std(n,xbins)
    ax.plot([mu_p,mu_p], [0,max(n)*1.05], 'k--', lw=0.8)	#plot the location of the center of the distribution 
    ax.plot([mu_p-std_p,mu_p-std_p], [0,max(n)*0.95], color='black',ls=':', lw=0.8)	#plot the location of -sigma 
    ax.plot([mu_p+std_p,mu_p+std_p], [0,max(n)*0.95], color='black',ls=':', lw=0.8)	#plot the location of +sigma 
    fig.suptitle("MC test on noise + data\n" r"Period={:.2f}$\pm${:.2f} | 95% CI={:.2f}$\pm${:.2f} hours".format(mu_p,std_p,mu_p,ci_p))		#add title
    plt.plot([mu_p-std_p,mu_p+std_p],[np.max(n),np.max(n)],lw=0.5,color='black')
    plt.text(mu_p,np.max(n)*1.05,r'$\sigma$')
    plt.ylim(0,np.max(n)*1.2)
    plt.xlim(rmin,rmax)
    plt.tick_params(left = False,labelleft = False)    	#disable left ticks and label
    ax.set_xlabel("Period distribution (hours)")		#add x axis label
    ax.plot([peri_best,peri_best], [0,max(n)], color='red', lw=0.8)	#plot location of best period
    fig.tight_layout()					#adjust plot tightness
    plt.savefig(str("{}/Results/{}/{}_window-{}_MC_peris.jpeg".format(path,body,body,w)), dpi=150) #save figure
    plt.close(fig)					#close figure


    return Q,mu_p,std_p,ci_p,mu,std,ci		#return the quality factor, the mean and std for the amplitude and frequency of the wobbling.

