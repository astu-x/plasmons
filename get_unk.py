import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.mlab as mlab
import subprocess
import os
import sys

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

def load_array_from_file(path,type):
    data = []
    i = 0
    #print("loading file: " + path)
    with open(path) as file:
        for line in file:
            if type == "raw":
                data.append(np.array([p.replace('(', '').replace(')', '').replace('j','i').split(", ") for p in line[2:-3].split('], [')]))
            if type == "harm":
                
                data = np.array([p.replace("], [","]], [[").split("], [") for p in line[2:-3].replace('j','i').replace(']], [[',']]], [[[').split(']], [[')])
                
                data = np.array([[b[1:-1].replace('), (',')), ((').split("), (") for b in a] for a in data])
                
                data = [[[np.fromstring(c[1:-1], dtype = "float", sep = ",") for c in b] for b in a] for a in data]
            i+=1
    return np.array(data)

def UNK2unk(cell,k):
    output = np.zeros((cell.shape[0],cell.shape[1]),dtype = 'complex')
    for i in range(0,cell.shape[0]):
        for j in range(0,cell.shape[1]):
            output[i][j] = cell[i][j]*np.exp(np.complex(0,-2*np.pi*k[0]*(i/cell.shape[0]) - 2*np.pi*k[1]*(j/cell.shape[1])))
    return output


#Load K_points
k_points = k_points_load("data/sim_specs.dat")

#Load any existing uc_harm file to get the size
data = load_array_from_file("data/uc_harm/uc_harm0.dat","harm")

#Make array to hold UNK's (Capital UNK means Bloch phase still included)
UNK = np.zeros((len(k_points),np.shape(data)[0],np.shape(data)[1]),dtype = 'complex')

#Create list with frequencies belonging to the band of interest (from 2nd script)
#Or leave 0 to skip (example only fills in a single k point)
band_frequencies = np.zeros(len(k_points))
band_frequencies[0] = 0.0221 #meep units

#Give parameters to filter "bad" modes
max_error = 10**(-6)
min_Q = 20
min_amp = 0.001
max_freq_dev = 0.1 #Meep frequency units

for k in range(0,len(k_points)):
    if band_frequencies[k] != 0:
        print(k)
        data = load_array_from_file("data/uc_harm/uc_harm"+str(k)+".dat","harm")
        for i in range(0,np.shape(data)[0]):
            for j in range(0,np.shape(data)[1]):
                current_freq = 0
                n_nearest = 0
                for n in range(0,np.shape(data[i][j])[0]):
                    if np.abs(data[i][j][n][1] - band_frequencies[k]) <= np.abs(band_frequencies[k] - current_freq):
                        current_freq = data[i][j][n][1]
                        n_nearest = n
                if data[i][j][n][6] < max_error and np.abs(band_frequencies[k] - current_freq) < max_freq_dev and np.abs(data[i][j][n_nearest][3]) > min_Q and data[i][j][n_nearest][4] > min_amp:
                    UNK[k][i][j] = data[i][j][n_nearest][4]*np.exp(np.complex(0,data[i][j][n_nearest][5]))
                else:
                    UNK[k][i][j] = 0

#Create array containing unk's (Corresponding Bloch phases removed, they become unit cell periodic)
unk = np.zeros((len(k_points),np.shape(data)[0],np.shape(data)[1]),dtype = 'complex')

for k in range(0,len(k_points)):
    unk[k] = UNK2unk(UNK[k],k_points[k])


#Now both UNK and unk are extracted and can be used for further investigation
#For instance creating images of the real part, phase and amplitudes, like so:

point2plot = 0

plt.figure()
plt.imshow(np.real(unk[point2plot]), origin = 'lower')
plt.colorbar()
plt.savefig("unk_real"+str(point2plot)+".png", dpi=300)
plt.close()

plt.figure()
plt.imshow(np.angle(unk[point2plot]), origin = 'lower')
plt.colorbar()
plt.savefig("unk_phase"+str(point2plot)+".png", dpi=300)
plt.close()

plt.figure()
plt.imshow(np.abs(unk[point2plot]), origin = 'lower')
plt.colorbar()
plt.savefig("unk_amp"+str(point2plot)+".png", dpi=300)
plt.close()