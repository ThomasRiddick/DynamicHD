import numpy as np
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.base import grid
from Dynamic_HD_Scripts.base.field import Field

def prepare_index_fields():
    nlat= 48
    nlon = 96
    lat_index_filename= "/Users/thomasriddick/Documents/data/temp/lat_index_file.nc"
    lon_index_filename= "/Users/thomasriddick/Documents/data/temp/lon_index_file.nc"
    lat_indices = np.arange(1,nlat+1)
    lon_indices = np.arange(1,nlon+1)
    lat_indices_field = np.transpose(np.tile(lat_indices,(nlon,1)))
    lon_indices_field = np.tile(lon_indices,(nlat,1))
    gaussian_grid = grid.makeGrid("T31")
    iodriver.advanced_field_writer(lat_index_filename,
                                   Field(lat_indices_field,
                                        grid=gaussian_grid),
                                   fieldname="lat_index")
    iodriver.advanced_field_writer(lon_index_filename,
                                   Field(lon_indices_field,
                                         grid=gaussian_grid),
                                   fieldname="lon_index")

def main():
    prepare_index_fields()

if __name__ == '__main__':
    main()
