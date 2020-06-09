import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.mlab as mlab
import subprocess
import os
import pandas as pd
from sympy import pretty_print as pp, latex

#Get resolution from sim_specs file
with open("data/sim_specs.dat") as file:
    for line in file:
        if line[0:11] == "#resolution":
            resolution = int(line.replace('#resolution: ', ""))

#Get amount of k-points from sim_specs file
with open("data/sim_specs.dat") as file:
    for line in file:
        if line[0:20] == "#number of k_points:":
            k_points_amount = int(line.replace('#number of k_points: ', ""))

def harminv(path):    
    
    #Set the frequency range(s) to be used by harminv
    #Sometimes harminv gives better results when the frequency range is more narrow.
    #To get the full range, harminv can then be executed multiple times in steps
    #In this case, it will run only once and look between 0.000 and 0.036
    freq_step = 0.036
    freq_low = 0.000
    freq_high = 0.037
    
    dt = 1/(2*resolution)
    
    i=0
    final_output = []
    
    with open(path) as file:
            for line in file: 
                
                print("k_point_number " + str(i+1) + " of " + str(k_points_amount))
                print("\n")
                
                range1 = freq_low
                range2 = freq_low + freq_step
                
                F=open("temp_file.dat","w")
                F.write(line)
                F.close()
                
                while range2 <= freq_high:# and 13 <= i <=15:
                    #This is the command used to execute harminv and where its parameters can be adjusted. 
                    #See https://github.com/NanoComp/harminv/blob/master/doc/harminv-man.md for all options.
                    cmd = 'harminv -t ' + str(dt) + ' -Q 0 -F -d 2 ' + str(range1) + '-' + str(range2) +' < temp_file.dat | tee temp_log_file >/dev/null'
                    os.system(cmd)
                    
                    j=0
                    with open('temp_log_file') as temp_file:
                        for temp_line in temp_file: 
                            if j != 0:
                                final_output.append((i,np.float(temp_line.split(',')[0]),np.float(temp_line.split(',')[1]),np.float(temp_line.split(',')[2]),np.float(temp_line.split(',')[3]),np.float(temp_line.split(',')[4]),np.float(temp_line.split(',')[5])))
                                print(temp_line)
                            j=1
                    range1 += freq_step
                    range2 += freq_step
                print('\n')
                i+=1
    
    os.system('rm temp_file.dat')
    os.system('rm temp_log_file')
    
    return final_output

def k_points_load(path):
    output = []
    
    with open(path) as file:
        for line in file:
            if line[0:9] == "#k_points":
                all_points = line.replace('#k_points: [Vector3<', "").replace('>,', "").replace('>]', "")
    
    all_points = all_points.split('Vector3<')
    
    for i in range(0,len(all_points)):
        output.append((np.float(all_points[i].split(',')[0]),np.float(all_points[i].split(',')[1]),np.float(all_points[i].split(',')[2])))
    
    return np.array(output)

#Get full range of calculated k-points from data file
k_points = k_points_load("data/sim_specs.dat")

#Execute harminv on single point in space (raw_field.dat produced by first script)
#This creates a new harminv_field.dat file with harminv data.
output_field = harminv("data/raw_field1.dat")
F=open("data/harminv_field1.dat","w")
for i in range(0,len(output_field)):
    F.write(str(output_field[i][0]) + "," + str(output_field[i][1]) + "," + str(output_field[i][2]) + "," + str(output_field[i][3]) + "," + str(output_field[i][4]) + "," + str(output_field[i][5]) + "," + str(output_field[i][6]) + "\n")
F.close()

#Filtering unwanted modes
#set parameters for filtering
quality_factor = 20
minimum_amplitude = 1
max_error = 10**(-6)

#Load harminv data
modes = []
with open("data/harminv_field1.dat") as file:
    for line in file: 
        modes.append((np.float(line.split(",")[0]),np.float(line.split(",")[1]),np.float(line.split(",")[2]),np.float(line.split(",")[3]),np.float(line.split(",")[4]),np.float(line.split(",")[5]),np.float(line.split(",")[6])))
modes = np.array(modes)

#Make filtered dataframe according to given parameters (panda dataframe used)
plot_data = []
for i in range(0,len(modes)):
    if np.abs(modes[i][3]) >= quality_factor and modes[i][4] >= minimum_amplitude and  max_error >= np.abs(modes[i][6]):
        plot_data.append((modes[i][0],modes[i][1],modes[i][2],modes[i][3],modes[i][4],modes[i][5],modes[i][6]))
plot_data = list(dict.fromkeys(plot_data))
plot_data = pd.DataFrame.from_dict(plot_data)
plot_data.columns = ['k_point','frequency','attenuation','Q-factor','amplitude','phase','error']


#Make the plot
fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 16})
plt.axvline(x=16, linewidth = 0.2, color = 'gray')
plt.axvline(x=38, linewidth = 0.2, color = 'gray')


plt.scatter(plot_data['k_point'],750*plot_data['frequency'], marker = '.', s=2, color = 'b')#750 is conversion number from meep to THz (when 1 meep length unit = 400nm)

ax.yaxis.set_tick_params(labelsize=12)

plt.xticks([0,18,39,50], labels = ["M","Γ","K","M"], size = 12)
plt.xlabel("Wave vector k", size = 'small')
plt.ylabel("Frequency (THz)", size = 'small')

#Functions used for secundary axis (cm^-1 vs THz)
def HtoW(x):
    return x*33.35641

def WtoH(x):
    return x/33.35641

ax.set_ylim(0,30)

plt.rcParams.update({'font.size': 12})
secaxy = ax.secondary_yaxis('right', functions=(HtoW, WtoH))
plt.rcParams.update({'font.size': 16})
secaxy.set_ylabel('wavenumber ' + latex("(cm$^{-1}$)"), size = 'small')

plt.savefig("band_diagram.png", bbox_inches='tight', dpi=300)
