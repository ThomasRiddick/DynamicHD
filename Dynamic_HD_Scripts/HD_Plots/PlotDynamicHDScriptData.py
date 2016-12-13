'''
This is the original plotting script; it is now largely kept only for archival purposes
Created on Jan 5, 2016

@author: m300468
'''

from scipy.io import FortranFile #@UnresolvedImport
import numpy as np
import matplotlib.pyplot as plt
import Dynamic_HD_Scripts.iohelper as iohelper
import Dynamic_HD_Scripts.dynamic_hd as dynamic_hd
import math
import Dynamic_HD_Scripts.flow_to_grid_cell as flow_to_grid_cell

def read_fortran_file(filename,data,data_type):        
    directory ='/Users/thomasriddick/Documents/data/temp' 
    with FortranFile(directory+'/'+filename,'r') as f: 
        data.append(f.read_record(dtype=data_type).reshape((360*6,720*6)))
        
def read_netCDF4_file(filename,data):
    directory ='/Users/thomasriddick/Documents/data/HDdata/orographys/' 
    filename = directory + filename
    data.append(iohelper.NetCDF4FileIOHelper.load_field(filename, 'LatLong5min',unmask=False))

def read_txt_file(filename,data):
    directory ='/Users/thomasriddick/Documents/data/HDdata/orographys/' 
    filename = directory + filename
    data.append(iohelper.TextFileIOHelper.load_field(filename, 'HD'))

def read_file(filename,data,data_type):
    print "Reading file {0}".format(filename)
    filetype = dynamic_hd.get_file_extension(filename)
    if filetype == '.datx':
        read_fortran_file(filename, data, data_type)
    elif filetype == '.nc':
        read_netCDF4_file(filename, data)
    elif filetype == '.txt':
        read_txt_file(filename,data)
    else:
        print "{0} is not a recognised file type".format(filetype)

def main():
    
    #floatfiles= ["base_test_orography.dat", "new_test_orography.dat"]
    #floatfiles= ["topo_hd_vs1_9_data_from_stefan.txt", "topo_created_with_uwe_scripts_method_with_mods.nc"]
    #floatfiles= ["topo_TEST_new.nc"]
    #floatfiles= ["base_blank_orography.dat","MMMTOPOIN.nc"]
    #floatfiles= ["topo_TEST.nc"]
    floatfiles= ["topo_hd_vs1_9_data_from_stefan.txt",'ice5g_0k_5min.nc']
    #intfiles  = ["updated_river_directions_from_real_current_orog_test_against_blank.dat"]
    intfiles  = ["updated_river_directions_from_real_current_orog_single_orog_test_5minres.datx"]
    #intfiles  = ["base_river_directions.dat", "updated_river_directions.dat"]
    #intfiles  = ["base_blank_river_directions.dat","updated_river_directions_from_real_test_against_blank.dat"] 
    #intfiles  = ["updated_river_directions_from_real_test_against_blank.dat"] 
    rdir_data = []
    orog_dat = []
    pmap_data = []
    for filename in floatfiles:
        read_file(filename,orog_dat,np.float64)
    for filename in intfiles:
        read_file(filename,rdir_data,np.int64)
    orog_dat[0] = np.ma.masked_less(orog_dat[0],0.0001)
    for array in orog_dat:
    #    for j in range(720):
    #        print array[:,j]
        levels = np.linspace(-100.0, 9900.0, 100, endpoint=True)
        plt.figure()
        plt.contourf(array,levels)
        plt.colorbar()
        ax = plt.gca()
        ax.invert_yaxis()
    levels = np.linspace(-100.0,500.0, 100, endpoint=True)
    #diff = orog_dat[0] - orog_dat[1]
    #plt.figure()
    #plt.contourf(diff,levels)
    #plt.colorbar()
    #ax = plt.gca()
    #ax.invert_yaxis()
    for array in rdir_data:
    #    for j in range(720):
    #        print array[:,j]
        lsmask = np.ma.getmaskarray(orog_dat[0])
        pmap_data.append(flow_to_grid_cell.create_hypothetical_river_paths_map(array,lsmask=lsmask))
        print array[30:40,30:40]
        #pmap_data.append(flow_to_grid_cell.create_hypothetical_river_paths_map(array,nlat=360*6,
                                                                                #nlong=720*6))
        #plt.figure()
        #levels = np.arange(-1.5,10,1)
        #plt.contourf(array,levels)
        #plt.colorbar()
        #ax = plt.gca()
        #ax.invert_yaxis()
    for array in pmap_data:
        lsmask = np.ma.getmaskarray(orog_dat[0])
        #print lsmask
        array=np.add.reduceat(array,np.linspace(0,360*6,num=360,endpoint=False,dtype=np.int64))
        array=np.add.reduceat(array,np.linspace(0,720*6,num=720,endpoint=False,dtype=np.int64),axis=1)
        array = np.rot90(array,k=2)
        array_masked = np.ma.array(array,mask=lsmask)
        print array_masked.shape
        plt.figure()
        #levels = np.logspace(math.log10(0.5), math.log10(500*36), 50)
        levels = np.logspace(math.log10(0.5*36), math.log10(8000), 50)
        print levels
        #plt.contourf(array_masked,levels,mask=lsmask)
        plt.contourf(array_masked,levels)
        plt.colorbar()
        ax = plt.gca()
        ax.invert_yaxis()
    plt.show()
        
if __name__ == '__main__':
    main()