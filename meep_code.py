import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import h5py as h5py
import time as time
import multiprocessing
import subprocess
import sys
import io
import os
import gc

#Determine full or single-point simulation (depending on passed argument, 
#e.g. "meep_code.py 23" executes 23 only, "meep_code.py" executes all specified k_points)
#This is useful for parallel processing (start new process for each k_point)
number_of_simulation = "Not Set"
if __name__ == "__main__":
    try:
        number_of_simulation = int(sys.argv[1])
    except(IndexError):
        pass


#Simulation parameters
resolution = 70
a = 1 #Unitcell length
z = 1 #thickness of simulation cell
pmls = 0.2 #Thickness of pmls or absorber
radius = 0.25*a #Hole radius
a_actualsize = 4*np.float128(1*10**(-7)) #in meters, parameter used for scaling values of dispersive material's dielectric
time_until = 5000 #length of simulation (time domain simulation)

#Useful constants
e =np.float128(1.6*10**(-19)) #Coulomb
Ef = np.float128(3.2*10**(-20)) #Joule (0.2Ev)
hbar = np.float128(1.05*10**(-34)) #Joule*s
c = np.float128(300000000) #m/s
epsilon_0 = np.float128(8.85*10**(-12))
f_c = np.float128(6.9*10**(11)) # Tesla/second


#Define global array for eigenfield calculations
UNK_ram = []

def simulation(number):
    
    #print number of simulation
    k_point = k_points[number]
    print("\n\n\n\n\n")
    print("run ", number+1, " of ", len(k_points))
    print("\n\n\n\n\n")
    
    
    #Define materials and geometries
    
    #single sheet of graphene with hexagonal holes
    #Drudian dispersion of Graphene (Jin), with gyrotropic term for magnetic field
    magnetic_field_strength = 10 #In Tesla
    eps_graphene = 1
    frequency = 1
    sigma = (resolution/a_actualsize)*np.float128((e**2*Ef)/(epsilon_0*hbar**2*np.pi))*np.float128(a_actualsize/c)**2/(2*np.pi)**2 #In dimensionless Meep units
    gamma = 0
    
    susceptibilities = [mp.GyrotropicDrudeSusceptibility(frequency=frequency, gamma=gamma, sigma=sigma, bias = mp.Vector3(0,0,10*f_c*np.float128(a_actualsize/c)))]
    graphene = mp.Medium(epsilon=eps_graphene, E_susceptibilities=susceptibilities)
    
    #Create simulations cell
    cell    = mp.Vector3(a,np.sqrt(3)*a,z + 2*pmls + 1/resolution)
    
    #Add rectangle with single pixel thickness to simulate Graphene
    geometry = [mp.Block(size=mp.Vector3(mp.inf,mp.inf,1/resolution), center=mp.Vector3(), material=graphene)]
    
    #Define holes in plane by adding columns with epsilon = 1
    
    #middle hole
    geometry.append(mp.Cylinder(radius, center=mp.Vector3(),material=mp.Medium(epsilon=1)))
    
    #corner holes
    geometry.append(mp.Cylinder(radius, center=mp.Vector3(0.5*a,0.5*a*np.sqrt(3)), material=mp.Medium(epsilon=1)))
    geometry.append(mp.Cylinder(radius, center=mp.Vector3(0.5*a,-0.5*a*np.sqrt(3)), material=mp.Medium(epsilon=1)))
    geometry.append(mp.Cylinder(radius, center=mp.Vector3(-0.5*a,0.5*a*np.sqrt(3)), material=mp.Medium(epsilon=1)))
    geometry.append(mp.Cylinder(radius, center=mp.Vector3(-0.5*a,-0.5*a*np.sqrt(3)), material=mp.Medium(epsilon=1)))
    
    
    #Alternative geometry with substrates and gold islands in orthogonal lattice (comment out "single sheet of graphene with hexagonal holes" part and uncomment this part)
    '''
    #Graphene           Drudian dispersion (Jin)
    epsilon_inf = 1
    frequency = 1
    sigma = (resolution/a_actualsize)*np.float128((e**2*Ef)/(epsilon_0*hbar**2*np.pi))*np.float128(a_actualsize/c)**2/(2*np.pi)**2
    gamma = 0#gamma_value*np.float128(a_actualsize/c)/(2*np.pi)
    susceptibilities = [mp.DrudeSusceptibility(frequency=frequency, gamma=gamma, sigma=sigma)]
    #susceptibilities = [mp.GyrotropicDrudeSusceptibility(frequency=frequency, gamma=gamma, sigma=sigma, bias = mp.Vector3(0,0,4*f_c*np.float128(a_actualsize/c)))]
    graphene = mp.Medium(epsilon=epsilon_inf, E_susceptibilities=susceptibilities)
    
    #Hexaboron nitride                  Beiranvand 2015, Theoretical, might not be good for IR regime
    epsilon_inf = 2.1
    hBN = material=mp.Medium(epsilon=epsilon_inf)
    
    #Silicon dioxide        Fit from Fei supplementary
    
    #Single peak
    epsilon_inf = 2.5
    
    f0 = 1071.52*3*10**10*np.float128(a_actualsize/c)
    sigma = (np.float128((721.25*3*10**10)**2)*np.float128(a_actualsize/c)**2)/(f0**2)    #Devide by f0 squared to get the numerator right (cancel the f0**2 factor)
    gamma = 32.5*3*10**10*np.float128(a_actualsize/c)
    
    susceptibilities = [mp.LorentzianSusceptibility(frequency=f0, gamma=gamma, sigma=sigma)]
    SiO2 = mp.Medium(epsilon=epsilon_inf, E_susceptibilities=susceptibilities)
    
    #Gold 
    epsilon_inf = 1
    frequency = 1
    sigma = np.float128((2*10**15)**2)*np.float128(a_actualsize/c)**2
    gamma = 0#4.5*10**12*np.float128(a_actualsize/c)
    
    susceptibilities = [mp.DrudeSusceptibility(frequency=frequency, gamma=gamma, sigma=sigma)]
    gold = mp.Medium(epsilon=epsilon_inf, E_susceptibilities=susceptibilities)
    
    #Geometry setup
    #create supercell
    cell    = mp.Vector3(a,a,z + 2*pmls + 1/resolution) #vertical orientation
    
    geometry = []
    
    #Gold cylinders (Gold first to make sure they don't pierce the other objects)
    #middle cylinder
    geometry.append(mp.Cylinder(radius, height = gold_height, center=mp.Vector3(0.15,0.15, gold_height/2),material=gold))#shifted from center
    
    #corner cylinders
    geometry.append(mp.Cylinder(radius, height = gold_height, center=mp.Vector3(0.5*a,0.5*a,gold_height/2), material=gold))
    geometry.append(mp.Cylinder(radius, height = gold_height, center=mp.Vector3(0.5*a,-0.5*a,gold_height/2), material=gold))
    geometry.append(mp.Cylinder(radius, height = gold_height, center=mp.Vector3(-0.5*a,0.5*a,gold_height/2), material=gold))
    geometry.append(mp.Cylinder(radius, height = gold_height, center=mp.Vector3(-0.5*a,-0.5*a,gold_height/2), material=gold))
    
    #SiO2 block
    geometry.append(mp.Block(size=mp.Vector3(mp.inf,mp.inf,z/2), center=mp.Vector3(0,0,-2/resolution -z/4), material=SiO2))
    
    #hBN rectangle
    geometry.append(mp.Block(size=mp.Vector3(mp.inf,mp.inf,1/resolution), center=mp.Vector3(0,0,0), material=hBN))
    
    #Graphene rectangle
    geometry.append(mp.Block(size=mp.Vector3(mp.inf,mp.inf,1/resolution), center=mp.Vector3(0,0,-1/resolution), material=graphene))
    '''
    
    
    #Add absorber or pml to avoid numerical reflection.
    #pml_layers = [mp.PML(pmls, direction=mp.Z)]
    pml_layers = [mp.Absorber(pmls, direction=mp.Z)]
    
    #initialize 2 output lists for 2 sources
    output1 = []
    output2 = []
    
    #Specify frequency range (for initiation source)
    fcen = 0.018 #center frequency
    df = 0.036 #width of frequency range
    
    #Add sources (2 sources with Bloch phase difference for hexagonal unitcell)
    sources = []
    
    #Specify source center and its periodic lattice copy (one pixel above graphene, as field is better defined there)
    center1 = mp.Vector3(0.2,-0.4,1/resolution)*a
    center1_lcp = center1 - mp.Vector3(0.5*a,-0.5*a*np.sqrt(3),0)
    
    sources.append(mp.Source(mp.GaussianSource(fcen,fwidth=df, cutoff = 5), component=mp.Ez, center=center1, amplitude = np.exp(np.complex(0,2*np.pi*k_point.dot(center1)))))
    sources.append(mp.Source(mp.GaussianSource(fcen,fwidth=df, cutoff = 5), component=mp.Ez, center=center1_lcp, amplitude = np.exp(np.complex(0,2*np.pi*k_point.dot(center1_lcp)))))
    
    #Set harmonic inversion functions for source and its lcp, for on the fly harminv directly done by meep
    #Options used by harminv can't be altered this way.
    h1 = mp.Harminv(mp.Ez,center1,fcen,df)
    h2 = mp.Harminv(mp.Ez,center1_lcp,fcen,df)
    
    #initialze simulation
    sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    boundary_layers=pml_layers,
                    sources=sources,
                    resolution=resolution,
                    Courant = 0.5)
    
    #Set k_point for bloch periodicity
    sim.k_point = k_point
    
    #turning eps averaging off (not really necessary...)
    sim.eps_averaging = False
    
    #Chunk splitting can disabled (Can give better load balancing in parallel)
    #sim.split_chunks_evenly=False
    
    #set variable for raw field data, which will be outputted later for analasys (harmonic inversion with own options)
    field1 = []
    field1_lcp = []
    
    #Function to get field values at source point at every time step for 
    def write_field(i):
        if mp.am_master():  #If statement only needed for parallel execution meep (but doesn't hurt serial)
            field1.append(sim.get_array(center=center1, size=mp.Vector3(), component=mp.Ez))
            field1_lcp.append(sim.get_array(center=center1_lcp, size=mp.Vector3(), component=mp.Ez))
    
    
    def UNK(sim):
        #Get field data at current time step
        frame1 = sim.get_array(center=mp.Vector3(0,0,1/resolution), size=mp.Vector3(a,np.sqrt(3)*a,0), component=mp.Ez) #in plane
        gc.collect()
        
        #Extract unit cell from supercell
        if mp.am_master():
            UNK = []
            for i in range(1,resolution+1): # The 1 to res + 1 is to remove k-vector boarder
                UNK_temp = []
                for j in range(1,int(0.5*np.sqrt(3)*(resolution+1))):
                    UNK_temp.append(frame1[i][int((0.25*np.sqrt(3))*(resolution+1))+j])
                UNK.append(UNK_temp)
                UNK_temp = []
            UNK = np.array(UNK)
            
            #possibility to reduce resolution of cell by averaging 2x2 pixels to one (resolution vs memory trade off)
            '''
            data_reduced = np.zeros((int(np.shape(UNK)[0]/2),int(np.shape(UNK)[1]/2)), dtype = "complex")
            for i in range(0,np.shape(UNK)[0]):
                for j in range(0,np.shape(UNK)[1]):
                    if (i+1)%2 == 0 and (j+1)%2 == 0:
                        data_reduced[int(i/2)][int(j/2)] = UNK[i][j]            
            UNK = np.array(data_reduced)
            '''
            
            #Add the unit cell to UNK_ram, which will hold the unit cell time evolution
            #Will be used later to extract eigen fields
            UNK_ram.append(UNK)
    
    #Write sim_specs file before start simulation (just terminate the simulation if you just want the specs file)
    if number_of_simulation == "Not Set" or number_of_simulation == 0:
        F=open("data/sim_specs.dat","w")
        F.write("#Simulation of heaxagonal graphene hole structure\n")
        F.write("#resolution: " + str(resolution) + "\n")
        F.write("#time_until: " + str(time_until) + "\n")
        F.write("#a: " + str(a) + "\n")
        F.write("#z: " + str(z) + "\n")
        F.write("#a_actualsize: " + str(a_actualsize) + "\n")
        F.write("#sigma: " + str(sigma) + "\n")
        F.write("#gamma: " + str(gamma) + "\n")
        F.write("#fcen: " + str(fcen) + "\n")
        F.write("#df: " + str(df) + "\n")
        F.write("#number of k_points: " + str(len(k_points)) + "\n")
        F.write("#k_points: " + str(k_points) + "\n")
        F.close()
    
    
    #Run the simulation (use mp.after_sources(UNK) for full eigen fields (high memory demand) or mp.after_sources(write_field) for eigen value only)
    sim.run(mp.after_sources(h1), mp.after_sources(h2), mp.after_sources(write_field), until = time_until)#mp.after_sources(UNK)
    
    
    #combine field1 output to one line (harminv format) (field at center of source)
    output_line2 = ""
    for i in range(0,len(field1)):
        output_line2 += str(field1[i].real)+"+"+str(field1[i].imag)+"i "
    output_line2 += "\n"
    
    #combine field1_lcp output to one line (harminv format) (field at center of source lattice copy)
    output_line3 = ""
    for i in range(0,len(field1_lcp)):
        output_line3 += str(field1_lcp[i].real)+"+"+str(field1_lcp[i].imag)+"i "
    output_line3 += "\n"
    
    #add "on the fly" harminv data as single lines
    output1.append(h1.modes)
    output2.append(h2.modes)
    
    #create output line for own analysss (analysis.py format)
    output_line1 = "source1:"
    output_line1 += "position="+str(center1)
    output_line1 += "k_point="+str(k_point)
    output_line1 += str(output1)
    output_line1 += "\n"
    output_line1 += "source2:"
    output_line1 += "position="+str(center1_lcp)
    output_line1 += "k_point="+str(k_point)
    output_line1 += str(output2)
    output_line1 += "\n"
    
    #All output data to single output list
    output = [output_line1,output_line2,output_line3,UNK_ram]
    
    return output

def load_array_from_file(path,type):
    data = []
    i = 0
    print("loading file: " + path)
    with open(path) as file:
        for line in file:
            if type == "raw":
                data.append(np.array([p.replace('(', '').replace(')', '').replace('j','i').split(", ") for p in line[2:-3].split('], [')]))
            if type == "harm":
                
                data = np.array([p.replace("], [","]], [[").split("], [") for p in line[2:-3].replace('j','i').replace(']], [[',']]], [[[').split(']], [[')])
                
                data = np.array([[b[1:-1].replace('), (',')), ((').split("), (") for b in a] for a in data])
                
                data = [[[np.fromstring(c[1:-1], dtype = "float", sep = ",") for c in b] for b in a] for a in data]
            i+=1
    data = np.array(data)
    print("%d bytes" % (data.size * data.itemsize))
    return data

def UC_harminv(file_number):
    
    UNK = []
    
    #From ram UNK_ram
    for l in range(0,len(UNK_ram[0][0])):
        UNK_temp1 = []
        for k in range(0,len(UNK_ram[0])):
            F=open("temp_file"+str(file_number)+".dat","w")
            line = ""
            for i in range(0,len(UNK_ram)):
                line += str(np.real(UNK_ram[i][k][l])) + "+" +  str(np.imag(UNK_ram[i][k][l])) + "i "
            F.write(line)
            F.close()
            
            dt = 1/(2*resolution)
            
            #frequency range to search for modes
            freq_low = 0
            freq_high = 0.036
            
            cmd = 'harminv -t ' + str(dt) + ' -F -f 300 ' + str(freq_low) + '-' + str(freq_high) +' < temp_file'+str(file_number)+'.dat | tee temp_log_file'+str(file_number)+' >/dev/null'
            os.system(cmd)
            m=0
            UNK_temp2 = []
            with open('temp_log_file'+str(file_number)) as temp_file:
                for temp_line in temp_file:
                    if m != 0:
                        UNK_temp2.append((file_number,np.float(temp_line.split(',')[0]),np.float(temp_line.split(',')[1]),np.float(temp_line.split(',')[2]),np.float(temp_line.split(',')[3]),np.float(temp_line.split(',')[4]),np.float(temp_line.split(',')[5])))
                    m+=1
            UNK_temp1.append(UNK_temp2)
        UNK.append(UNK_temp1)
        print("line: " + str(l+1) + " of " + str(len(UNK_ram[0][0])))
    
    
    #remove temp files
    os.system('rm temp_file'+str(file_number)+'.dat')
    os.system('rm temp_log_file'+str(file_number))
    
    UNK = np.array(UNK)
    
    #write result array to file
    F=open("data/uc_harm/uc_harm"+str(file_number)+".dat","w")
    F.write(str(np.ndarray.tolist(np.array(UNK))) + "\n")
    F.close()



#Initialize and start the simulations here

#Set k_points to be calculated
#k-points along symmetry lines IBZ (hexagonal lattice)
Gamma = mp.Vector3()
M = mp.Vector3(0,1/np.sqrt(3))/a
K = mp.Vector3(1/3, 1/np.sqrt(3))/a
k_points_amount = 30*a

k_points = mp.interpolate(int(k_points_amount*(M-Gamma).norm()), [M,Gamma])
k_points = k_points[:-1]
k_points = k_points + mp.interpolate(int(k_points_amount*(Gamma-K).norm()), [Gamma,K])
k_points = k_points[:-1]
k_points = k_points + mp.interpolate(int(k_points_amount*(K-M).norm()), [K,M])

#18 is gamma point
#39 is K point

#points to cover entire brilloiuine zone (hexagonal lattice)
'''
#rhombic shape
Gamma1 = mp.Vector3()
Gamma2 = mp.Vector3(0,2/np.sqrt(3))/a
Gamma3 = mp.Vector3(1/3, 1/np.sqrt(3))/a + mp.Vector3(2/3, 0)/a
Gamma4 = Gamma3 - Gamma2

N = 35 #(36x36 brillouine zone sampling)

shift = (Gamma2 + Gamma4)/(2*(N+1))

Gamma1 -= shift
Gamma2 -= shift
Gamma3 -= shift
Gamma4 -= shift

bz_line1 = mp.interpolate(N, [Gamma1,Gamma2])
bz_line2 = mp.interpolate(N, [Gamma4,Gamma3])

bz_surface_points = []

for i in range(0,len(bz_line1)):
    bz_surface_points += mp.interpolate(N, [bz_line1[i],bz_line2[i]])

k_points = bz_surface_points
'''


#Run the simulation function from here
if number_of_simulation == "Not Set":#Serial execution
    #All points calculation
    result = []
    for i in range(0, len(k_points)):
        result.append(simulation(i))
        number_of_simulation = ""
    
    #Export all useful data to files!
    
    #print harminv results to file
    F=open("data/output.dat","w")
    for i in range(0,len(result)):
        F.write(result[i][0])
    F.close()
    
    #print raw field1 data from source1
    F=open("data/raw_field1.dat","w")
    for i in range(0,len(result)):
        F.write(result[i][1])
    F.close()
    
    #print raw field1_lcp data from source2
    F=open("data/raw_field1_lcp.dat","w")
    for i in range(0,len(result)):
        F.write(result[i][2])
    F.close()
else:
    #single k_point calculation
    result = []
    try:
        result.append(simulation(number_of_simulation))
    except Exception as message:
        print(type(message))
        print(message)
    
    
    #Export all useful data to files!
    
    #print harminv results to file
    F=open("data/output"+str(number_of_simulation)+".dat","w")
    for i in range(0,len(result)):
        F.write(result[i][0])
    F.close()
    
    #print raw field1 data from source1
    #Numbering is for individual k_point execution, all files can be combined to a single raw_field1.dat file afterwards. 
    F=open("data/raw_field1_"+str(number_of_simulation)+".dat","w")
    for i in range(0,len(result)):
        F.write(result[i][1])
    F.close()
    
    #print raw field1_lcp data from source2
    F=open("data/raw_field1_lcp_"+str(number_of_simulation)+".dat","w")
    for i in range(0,len(result)):
        F.write(result[i][2])
    F.close()
    
    
    #Harmonic inversion on unit cell if used
    if UNK_ram != []:
        UC_harminv(number_of_simulation)