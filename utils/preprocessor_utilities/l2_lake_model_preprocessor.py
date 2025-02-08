import regex as re
import sys

input_file=sys.argv[1]
output_file=sys.argv[1]

with open(input_file,'r') as f:
    txt = f.read()

re.sub(r'/_DIMS_/',':,:',txt)
re.sub(r'/_DEF_NPOINTS_HD_','integer :: nlat_hd,nlon_hd')
re.sub(r'/_DEF_NPOINTS_LAKE_','integer :: nlat_lake,nlon_lake')
re.sub(r'/_DEF_NPOINTS_SURFACE_','integer :: nlat_surface,nlon_surface')

with open(output_file,'w') as f:
    f.write(txt)
