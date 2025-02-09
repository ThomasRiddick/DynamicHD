import re
import sys

input_file=sys.argv[1]
output_file=sys.argv[2]

print(f"Preprocessing file: {input_file}")

with open(input_file,'r') as f:
    lines = f.readlines()

dim1="lat"
dim2="lon"
modified_txt = ""
extended_line = None
in_lake_grid_loop = False
in_hd_grid_loop = False
in_surface_grid_loop = False
for line in lines:
    if extended_line is not None:
        extended_line += line.rstrip('\n')
        if not re.match(".*&",line):
            if not re.match(r'\s*_ASSIGN_.*&',extended_line):
                raise RuntimeError("Preprocessor logic failure")
            extended_line = re.sub(r'(\s*)_ASSIGN_?(.*)_COORDS_(.*)_\s*=\s*(&?)\s*(.*)_COORDS_(.*)_',
                                   f'\\1\\2\\3_{dim1} = \\4\\5\\6_{dim1}\\n'
                                   f'\\1\\2\\3_{dim2} = \\4\\5\\6_{dim2}\\n',
                                   extended_line)
            extended_line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                                   r'INDEX_NAME(.*)_\s*=\s*(.*)_COORDS_(\w+)_',
                                   f'\\1\\2\\3{dim1}\\4 = \\5\\6_{dim1}\\n'
                                   f'\\1\\2\\3{dim2}\\4 = \\5\\6_{dim2}\\n',
                                   extended_line)
            extended_line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                                   r'INDEX_NAME(.*)_\s*=\s*(.*)_INDICES_LIST_(\w*)'
                                   r'INDEX_NAME(.*)_',
                                   f'\\1\\2\\3{dim1}\\4 = \\5\\6{dim1}\\7\\n'
                                   f'\\1\\2\\3{dim2}\\4 = \\5\\6{dim2}\\7\\n',
                                   extended_line)
            extended_line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                                   r'INDEX_NAME(.*)_\s*=\s*(.*)',
                                   f'\\1\\2\\3{dim1}\\4 = \\5\\n'
                                   f'\\1\\2\\3{dim2}\\4 = \\5\\n',
                                   extended_line)
            extended_line_spacing = re.match(r'(\s*)',extended_line).group(1)
            extended_line_as_list = extended_line.split("&")
            modified_txt += f"&\n{extended_line_spacing}  ".join(extended_line_as_list)
            extended_line = None
            continue
    #Temporarily strip out comments
    line = re.sub(r'!.*',r'',line)

    #Order of operations is important for correct parsing of preproc statements
    line = re.sub(r'_DIMS_',':,:',line)
    line = re.sub(r'(\s*)_GET_COARSE_COORDS_FROM_FINE_COORDS_\s+(\w*)%_COORDS_(\w*)_'
                  r'\s+_NPOINTS_(\w*)_\s+_NPOINTS_(\w*)_\s+_COORDS_(\w*)_',
                  lambda m : f'{m.group(1)}fine_cells_per_coarse_cell_{dim1}'
                             f' = n{dim1}_{m.group(4)}/n{dim1}_{m.group(5)}\n'
                             f'{m.group(1)}fine_cells_per_coarse_cell_{dim2}'
                             f' = n{dim2}_{m.group(4)}/n{dim2}_{m.group(5)}\n'
                             f'{m.group(1)}{m.group(2)}%{m.group(3)}_{dim1}'
                             f' = ceiling(real({m.group(6)}_{dim1})'
                             f'/real(fine_cells_per_coarse_cell_{dim1}))\n'
                             f'{m.group(1)}{m.group(2)}%{m.group(3)}_{dim2}'
                             f' = ceiling(real({m.group(6)}_{dim2})'
                             f'/real(fine_cells_per_coarse_cell_{dim2}))'.lower(),
                  line)
    m = re.match(r'\s*_LOOP_OVER_(\w+)_GRID_END_\s',line)
    if m:
        if m.group(1) == "HD":
            in_hd_grid_loop = False
        elif m.group(1) == "LAKE":
            in_lake_grid_loop = False
        elif m.group(1) == "SURFACE":
            in_surface_grid_loop = False
        else:
            raise RuntimeError("Unknown loop type")
    if in_hd_grid_loop:
        line = "  " + line
    if in_lake_grid_loop:
        line = "  " + line
    if in_surface_grid_loop:
        line = "  " + line
    m = re.match(r'\s*_LOOP_OVER_(\w+)_GRID_\s',line)
    if m:
        if m.group(1) == "HD":
            in_hd_grid_loop = True
        elif m.group(1) == "LAKE":
            in_lake_grid_loop = True
        elif m.group(1) == "SURFACE":
            in_surface_grid_loop = True
        else:
            raise RuntimeError("Unknown loop type")
    if re.match(r'\s*_ASSIGN_.*&',line):
        extended_line = line.rstrip('\n')
        continue
    line = re.sub(r'(\s*)_LOOP_OVER_(\w+)_GRID_END_',r'\1  end do\n\1end do',line)
    line = re.sub(r'(\s*)_LOOP_OVER_(\w+)_GRID_ _COORDS_(\w+)_',
                  lambda m : f'{m.group(1)}do {dim1}_{m.group(3)} = 1,n{dim1}_{m.group(2)}\n'
                             f'{m.group(1)}  do {dim2}_{m.group(3)} = 1,n{dim2}_{m.group(2)}'.lower(),
                  line)
    line = re.sub(r'(\s*)_GET_COORDS_\s+_COORDS_(\w+)_\s+_FROM_\s+_INDICES_FIELD_(\w*)'
                  r'INDEX_NAME(\w*)_\s+_COORDS_(\w+)_',
                  lambda m : f'{m.group(1)}{m.group(2)}_{dim1} = '
                             f'{m.group(3)}{dim1}{m.group(4)}'
                             f'({dim1}_{m.group(5)},lon_{m.group(5)})\n'
                             f'{m.group(1)}{m.group(2)}_{dim2} = '
                             f'{m.group(3)}{dim2}{m.group(4)}'
                             f'({dim1}_{m.group(5)},{dim2}_{m.group(5)})'.lower(),
                  line)
    line = re.sub(r'_DEF_NPOINTS_HD_ _INTENT_(\w+)_',
                  f'integer, intent(\\1) :: n{dim1}_hd, n{dim2}_hd',line)
    line = re.sub(r'_DEF_NPOINTS_LAKE_ _INTENT_(\w+)_',
                  f'integer, intent(\\1) :: n{dim1}_lake, n{dim2}_lake',line)
    line = re.sub(r'_DEF_NPOINTS_SURFACE_ _INTENT_(\w+)_',
                  f'integer, intent(\\1) :: n{dim1}_surface, n{dim2}_surface',line)
    line = re.sub(r'_DEF_NPOINTS_HD_',f'integer :: n{dim1}_hd, n{dim2}_hd',line)
    line = re.sub(r'_DEF_NPOINTS_LAKE_',f'integer :: n{dim1}_lake, n{dim2}_lake',line)
    line = re.sub(r'_DEF_NPOINTS_SURFACE_',f'integer :: n{dim1}_surface, n{dim2}_surface',line)
    line = re.sub(r'_DEF_LOOP_INDEX_HD_',
                  f'integer :: {dim1}_hd, {dim2}_hd',line)
    line = re.sub(r'_DEF_LOOP_INDEX_LAKE_',
                  f'integer :: {dim1}_lake, {dim2}_lake',line)
    line = re.sub(r'_DEF_LOOP_INDEX_SURFACE_',
                  f'integer :: {dim1}_surface, {dim2}_surface',line)
    line = re.sub(r'_DEF_COORDS_(\w*)_',
                  f'integer :: \\1_{dim1}, \\1_{dim2}',line)
    line = re.sub(r'_ASSIGN_?(.*)_NPOINTS_(\w+)_ = _NPOINTS_(\w+)_',
                  lambda m : f'{m.group(1)}n{dim1}_{m.group(2)} '
                             f'= n{dim1}_{m.group(3)}\n'
                             f'{m.group(1)}n{dim2}_{m.group(2)} '
                             f'= n{dim2}_{m.group(3)}'.lower(),
                  line)
    line = re.sub(r'_NPOINTS_HD_',f'n{dim1}_hd,n{dim2}_hd',line)
    line = re.sub(r'_NPOINTS_LAKE_',f'n{dim1}_lake,n{dim2}_lake',line)
    line = re.sub(r'_NPOINTS_SURFACE_',f'n{dim1}_surface,n{dim2}_surface',line)
    line = re.sub(r'(\s*)allocate\((.*)_INDICES_LIST_(\w*)INDEX_NAME(.*)_\)',
                  f'\\1allocate(\\2\\3{dim1}\\4)\n'
                  f'\\1allocate(\\2\\3{dim2}\\4)',
                  line)
    line = re.sub(r'(\s*)_ASSIGN_?(.*)_COORDS_(.*)_\s*=\s*(.*)_COORDS_(\w*)_',
                  f'\\1\\2\\3_{dim1} = \\4\\5_{dim1}\\n'
                  f'\\1\\2\\3_{dim2} = \\4\\5_{dim2}',
                  line)
    line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                  r'INDEX_NAME(.*)_\s*=\s*(.*)_COORDS_(\w+)_',
                  f'\\1\\2\\3{dim1}\\4 = \\5\\6_{dim1}\\n'
                  f'\\1\\2\\3{dim2}\\4 = \\5\\6_{dim2}\\n',
                  line)
    line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                  r'INDEX_NAME(.*)_\s*=\s*(.*)_INDICES_LIST_(\w*)'
                  r'INDEX_NAME(.*)_',
                  f'\\1\\2\\3{dim1}\\4 = \\5\\6{dim1}\\7\\n'
                  f'\\1\\2\\3{dim2}\\4 = \\5\\6{dim2}\\7\\n',
                  line)
    line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                  r'INDEX_NAME(.*)_\s*=\s*(.*)',
                  f'\\1\\2\\3{dim1}\\4 = \\5\\n'
                  f'\\1\\2\\3{dim2}\\4 = \\5\\n',
                  line)
    line = re.sub(r'_INDICES_LIST_(\w+)_FIRST_DIM_',f'{dim1}_\\1',line)
    line = re.sub(r'(\s*)_DEF_INDICES_FIELD_(\w*)INDEX_NAME(\w*)_(\s*_INTENT_\w*_)?',
                 f'\\1integer, dimension(:,:), allocatable :: \\2{dim1}\\3\\4\\n'
                 f'\\1integer, dimension(:,:), allocatable :: \\2{dim2}\\3\\4',line)
    line = re.sub(r'(\s*)_DEF_INDICES_LIST_(\w*)INDEX_NAME(\w*)_(\s*_INTENT_\w*_)?',
                  f'\\1integer, dimension(:), allocatable :: \\2{dim1}\\3\\4\\n'
                  f'\\1integer, dimension(:), allocatable :: \\2{dim2}\\3\\4',line)
    line = re.sub(r'(\s*)_DEF_COORDS_(\w*)_( _INTENT_\w*_)?',
                  f'\\1integer :: \\2_{dim1}\\3\\n\\1integer :: \\2_{dim2}\\3',line)
    line = re.sub(r'_INDICES_(?:FIELD|LIST)_(\w*)INDEX_NAME(\w*)_',
                  f'\\1{dim1}\\2,\\1{dim2}\\2',
                  line)
    line = re.sub(r'_COORDS_LAKE_',f'{dim1}_lake,{dim2}_lake',line)
    line = re.sub(r'_COORDS_HD_',f'{dim1}_hd,{dim2}_hd',line)
    line = re.sub(r'_COORDS_SURFACE_',f'{dim1}_surface,{dim2}_surface',line)
    line = re.sub(r'_COORDS_ARG_(\w*)_',
                  f'\\1_{dim1},\\1_{dim2}',line)
    line = re.sub(r'(.*)\s+::(.*) _INTENT_(\w*)_',r'\1, intent(\3) ::\2',line)
    modified_txt += line

with open(output_file,'w') as f:
    f.write(modified_txt)
