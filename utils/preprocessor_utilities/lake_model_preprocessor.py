import re
import sys

input_file=sys.argv[1]
output_file=sys.argv[2]

swapindices=False
print(f"Preprocessing file: {input_file}")
if swapindices:
  print("Indices swapped")

with open(input_file,'r') as f:
    lines = f.readlines()
if swapindices:
  dim1="lon"
  dim2="lat"
else:
  dim1="lat"
  dim2="lon"
modified_txt = ""
extended_line = None
in_lake_grid_loop = False
in_hd_grid_loop = False
in_surface_grid_loop = False
ignore_following_lines = False
filtered_lines = []
for line in lines:
    if re.match(r'\s*_IF_USE_LONLAT_',line):
        ignore_following_lines = not swapindices
        continue
    elif re.match(r'\s*_ELSE_IF_NOT_USE_LONLAT_',line):
        ignore_following_lines = swapindices
        continue
    elif re.match(r'\s*_END_IF_USE_LONLAT_',line):
        ignore_following_lines = False
        continue
    elif ignore_following_lines:
        continue
    else:
        filtered_lines.append(line)
ignore_following_line = False
lines = filtered_lines
filtered_lines = []
for line in lines:
    if re.match(r'\s*_IF_USE_SINGLE_INDEX_',line):
        ignore_following_lines = True
        continue
    elif re.match(r'\s*_ELSE_',line):
        ignore_following_lines = False
        continue
    elif re.match(r'\s*_END_IF_USE_SINGLE_INDEX_',line):
        ignore_following_lines = False
        continue
    elif ignore_following_lines:
        continue
    else:
        filtered_lines.append(line)

for line in filtered_lines:
    if extended_line is not None:
        extended_line += line.rstrip('\n')
        if not re.match(".*&",line):
            if not re.match(r'\s*_ASSIGN_.*&',extended_line):
                raise RuntimeError("Preprocessor logic failure")
            extended_line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                                   r'INDEX_NAME(.*)_\s*=\s*(.*)_COORDS_(HD|LAKE|SURFACE)_',
                                   lambda m : f'{m.group(1)}{m.group(2)}{m.group(3)}{dim1}'
                                              f'{m.group(4)} = {m.group(5)}{dim1}_{m.group(6)}\n'
                                   f'{m.group(1)}{m.group(2)}{m.group(3)}{dim2}'
                                   f'{m.group(4)} = {m.group(5)}{dim2}_{m.group(6)}\n'.lower(),
                                   extended_line)
            extended_line = re.sub(r'(\s*)_ASSIGN_?(.*)_COORDS_(.*)_\s*='
                                   r'\s*(&?)\s*(.*)_COORDS_(HD|LAKE|SURFACE)_',
                                   lambda m : f'{m.group(1)}{m.group(2)}{m.group(3)}'
                                   f'_{dim1} = {m.group(4)}{m.group(5)}{dim1}_{m.group(6)}\n'
                                   f'{m.group(1)}{m.group(2)}{m.group(3)}_{dim2} = '
                                   f'{m.group(4)}{m.group(5)}{dim2}_{m.group(6)}\n'.lower(),
                                   extended_line)
            extended_line = re.sub(r'(\s*)_ASSIGN_?(.*)_COORDS_(.*)_\s*='
                                   r'\s*(&?)\s*(.*)_COORDS_(.*)_',
                                   f'\\1\\2\\3_{dim1} = \\4\\5\\6_{dim1}\\n'
                                   f'\\1\\2\\3_{dim2} = \\4\\5\\6_{dim2}\\n',
                                   extended_line)
            extended_line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                                   r'INDEX_NAME(.*)_\s*=\s*(.*)_COORDS_(\w+)_',
                                   f'\\1\\2\\3{dim1}\\4 = \\5\\6_{dim1}\\n'
                                   f'\\1\\2\\3{dim2}\\4 = \\5\\6_{dim2}\\n',
                                   extended_line)
            extended_line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_(?:LIST|FIELD)_(\w*)'
                                   r'INDEX_NAME(.*)_\s*(=>?)\s*(.*)_INDICES_(?:LIST|FIELD)_(\w*)'
                                   r'INDEX_NAME(.*)_',
                                   f'\\1\\2\\3{dim1}\\4 \\5 \\6\\7{dim1}\\8\\n'
                                   f'\\1\\2\\3{dim2}\\4 \\5 \\6\\7{dim2}\\8\\n',
                                   extended_line)
            extended_line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                                   r'INDEX_NAME(.*)_\s*(=?)\s*(.*)',
                                   f'\\1\\2\\3{dim1}\\4 \\5 \\6\\n'
                                   f'\\1\\2\\3{dim2}\\4 \\5 \\6\\n',
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
    line = re.sub(r'(\s*)_END_FOR_FINE_CELLS_IN_COARSE_CELL_',r'\1  end do\n\1end do',line)

    line = re.sub(r'(\s*)_FOR_FINE_CELLS_IN_COARSE_CELL_ _COORDS_(\w+)_ _COORDS_(HD|SURFACE|LAKE)_ _SCALE_FACTORS_(\w+)_',
                  lambda m: f'{m.group(1)}do {m.group(2)}_{dim1} = '
                            f'1+({dim1}_{m.group(3)}-1)*{dim1}_{m.group(4)},'
                            f'{dim1}_{m.group(3)}*{dim1}_{m.group(4)}\n'
                            f'{m.group(1)}  do {m.group(2)}_{dim2} = '
                            f'1+({dim2}_{m.group(3)}-1)*{dim2}_{m.group(4)},'
                            f'{dim2}_{m.group(3)}*{dim2}_{m.group(4)}'.lower(),
                  line)
    line = re.sub(r'(\s*)_LOOP_OVER_(\w+)_GRID_END_',r'\1  end do\n\1end do',line)
    line = re.sub(r'(\s*)_LOOP_OVER_(\w+)_GRID_ _COORDS_(\w+)_(?:\s+_(.*)_)?',
                  lambda m : f'{m.group(1)}do {dim1}_{m.group(3)} ='
                             f' 1,{m.group(4) if m.group(4) else ""}n{dim1}_{m.group(2)}\n'
                             f'{m.group(1)}  do {dim2}_{m.group(3)} = '
                             f'1,{m.group(4) if m.group(4) else ""}n{dim2}_{m.group(2)}'.lower(),
                  line)
    line = re.sub(r'(\s*)_GET_SWITCHED_COORDS_\s+_COORDS_(\w+)_\s+_FROM_\s+_ARRAY_(\w*)_\s*_OFFSET_(\d*)_',
                  f'\\1\\2_{dim2} = \\3(1+\\4)\\n'
                  f'\\1\\2_{dim1} = \\3(2+\\4)',
                  line)
    line = re.sub(r'(\s*)_GET_SWITCHED_COORDS_\s+_COORDS_(\w+)_\s+_FROM_\s+_ARRAY_(\w*)_',
                  f'\\1\\2_{dim2} = \\3(1)\\n'
                  f'\\1\\2_{dim1} = \\3(2)',
                  line)
    line = re.sub(r'(\s*)_GET_COORDS_\s+_COORDS_(\w+)_\s+_FROM_\s+_ARRAY_(\w*)_\s*_OFFSET_(\d*)_',
                  f'\\1\\2_{dim1} = \\3(1+\\4)\\n'
                  f'\\1\\2_{dim2} = \\3(2+\\4)',
                  line)
    line = re.sub(r'(\s*)_GET_COORDS_\s+_COORDS_(\w+)_\s+_FROM_\s+_ARRAY_(\w*)_',
                  f'\\1\\2_{dim1} = \\3(1)\\n'
                  f'\\1\\2_{dim2} = \\3(2)',
                  line)
    line = re.sub(r'(\s*)_GET_COORDS_\s+_COORDS_(\w+)_\s+_FROM_\s+_INDICES_FIELD_(\w*)'
                  r'INDEX_NAME(\w*)_\s+_COORDS_(HD|LAKE|SURFACE)_',
                  lambda m : f'{m.group(1)}{m.group(2)}_{dim1} = '
                             f'{m.group(3)}{dim1}{m.group(4)}'
                             f'({dim1}_{m.group(5)},{dim2}_{m.group(5)})\n'
                             f'{m.group(1)}{m.group(2)}_{dim2} = '
                             f'{m.group(3)}{dim2}{m.group(4)}'
                             f'({dim1}_{m.group(5)},{dim2}_{m.group(5)})'.lower(),
                  line)
    line = re.sub(r'(\s*)_GET_COORDS_\s+_COORDS_(\w+)_\s+_FROM_\s+(.*)_INDICES_FIELD_?(\w*)'
                  r'INDEX_NAME(\w*)_\s+(.*)_COORDS_(\w+)_',
                  f'\\1\\2_{dim1} = '
                  f'\\3\\4{dim1}\\5'
                  f'(\\6\\7_{dim1},\\6\\7_{dim2})\\n'
                  f'\\1\\2_{dim2} = '
                  f'\\3\\4{dim2}\\5'
                  f'(\\6\\7_{dim1},\\6\\7_{dim2})',
                  line)
    line = re.sub(r'(\s*)_GET_COORDS_\s+_COORDS_(\w+)_\s+_FROM_\s+(.*)_INDICES_LIST_(\w*)'
                  r'INDEX_NAME(\w*)_\s+(.*)',
                  f'\\1\\2_{dim1} = '
                  f'\\3\\4{dim1}\\5'
                  f'(\\6)\\n'
                  f'\\1\\2_{dim2} = '
                  f'\\3\\4{dim2}\\5'
                  f'(\\6)',
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
    line = re.sub(r'_DEF_SCALE_FACTORS_(\w+)_',
                  f'integer :: {dim1}_\\1, {dim2}_\\1',line)
    line = re.sub(r'(\s*)_ASSIGN_?(.*)_NPOINTS_(\w+)_ = _NPOINTS_(\w+)_',
                  lambda m : f'{m.group(1)}{m.group(2)}n{dim1}_{m.group(3)} '
                             f'= n{dim1}_{m.group(4)}\n'
                             f'{m.group(1)}{m.group(2)}n{dim2}_{m.group(3)} '
                             f'= n{dim2}_{m.group(4)}'.lower(),
                  line)
    line = re.sub(r'(\s*)_CALCULATE_SCALE_FACTORS_(\w+)_\s*_?(.*)_NPOINTS_(\w+)_\s*_?(.*)_NPOINTS_(\w+)_',
                  lambda m : f'{m.group(1)}{dim1}_{m.group(2)} ='
                             f' {m.group(3)}n{dim1}_{m.group(4)}/{m.group(5)}n{dim1}_{m.group(6)}\n'
                             f'{m.group(1)}{dim2}_{m.group(2)} ='
                             f' {m.group(3)}n{dim2}_{m.group(4)}/{m.group(5)}n{dim2}_{m.group(6)}'.lower(),
                  line)
    line = re.sub(r'(\w+%)?_NPOINTS_HD_',f'\\1n{dim1}_hd,\\1n{dim2}_hd',line)
    line = re.sub(r'(\w+%)?_NPOINTS_LAKE_',f'\\1n{dim1}_lake,\\1n{dim2}_lake',line)
    line = re.sub(r'(\w+%)?_NPOINTS_SURFACE_',f'\\1n{dim1}_surface,\\1n{dim2}_surface',line)
    line = re.sub(r'(\w+%)?_NPOINTS_TOTAL_HD_',f'\\1n{dim1}_hd*\\1n{dim2}_hd',line)
    line = re.sub(r'(\w+%)?_NPOINTS_TOTAL_LAKE_',f'\\1n{dim1}_lake*\\1n{dim2}_lake',line)
    line = re.sub(r'(\w+%)?_NPOINTS_TOTAL_SURFACE_',f'\\1n{dim1}_surface*\\1n{dim2}_surface',line)
    line = re.sub(r'_INDICES_LIST_(\w*)INDEX_NAME(\w*)_FIRST_DIM_',f'\\1{dim1}\\2',line)
    line = re.sub(r'(\s*)deallocate\((.*)_INDICES_LIST_(\w*)INDEX_NAME(.*)_\)',
                  f'\\1deallocate(\\2\\3{dim1}\\4)\n'
                  f'\\1deallocate(\\2\\3{dim2}\\4)',
                  line)
    line = re.sub(r'(\s*)allocate\((.*)_INDICES_LIST_(\w*)INDEX_NAME(.*)_\)',
                  f'\\1allocate(\\2\\3{dim1}\\4)\n'
                  f'\\1allocate(\\2\\3{dim2}\\4)',
                  line)
    line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                                   r'INDEX_NAME(.*)_\s*=\s*(.*)_COORDS_(HD|LAKE|SURFACE)_',
                                   lambda m : f'{m.group(1)}{m.group(2)}{m.group(3)}{dim1}'
                                              f'{m.group(4)} = {m.group(5)}{dim1}_{m.group(6)}\n'
                                   f'{m.group(1)}{m.group(2)}{m.group(3)}{dim2}'
                                   f'{m.group(4)} = {m.group(5)}{dim2}_{m.group(6)}\n'.lower(),
                                   line)
    line = re.sub(r'(\s*)_ASSIGN_?(.*)_COORDS_(.*)_\s*=\s*_VALUE_(.*)_',
                                   f'\\1\\2\\3_{dim1} = \\4\\n'
                                   f'\\1\\2\\3_{dim2} = \\4\\n',
                                   line)
    line = re.sub(r'(\s*)_ASSIGN_?(.*)_COORDS_(.*)_\s*='
                                   r'\s*(&?)\s*(.*)_COORDS_(HD|LAKE|SURFACE)_',
                                   lambda m : f'{m.group(1)}{m.group(2)}{m.group(3)}'
                                   f'_{dim1} = {m.group(4)}{m.group(5)}{dim1}_{m.group(6)}\n'
                                   f'{m.group(1)}{m.group(2)}{m.group(3)}_{dim2} = '
                                   f'{m.group(4)}{m.group(5)}{dim2}_{m.group(6)}\n'.lower(),
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
    line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_(?:LIST|FIELD)_(\w*)'
                  r'INDEX_NAME(.*)_\s*=\s*(.*)_INDICES_(?:LIST|FIELD)_(\w*)'
                  r'INDEX_NAME(.*)_',
                  f'\\1\\2\\3{dim1}\\4 = \\5\\6{dim1}\\7\\n'
                  f'\\1\\2\\3{dim2}\\4 = \\5\\6{dim2}\\7\\n',
                  line)
    line = re.sub(r'(\s*)_ASSIGN_?(.*)_INDICES_LIST_(\w*)'
                  r'INDEX_NAME(.*)_\s*(=>?)\s*(.*)',
                  f'\\1\\2\\3{dim1}\\4 \\5 \\6\\n'
                  f'\\1\\2\\3{dim2}\\4 \\5 \\6\\n',
                  line)
    line = re.sub(r'(\s*)_DEF_INDICES_FIELD_(\w*)INDEX_NAME(\w*)_(\s*_INTENT_\w*_)?',
                 f'\\1integer, dimension(:,:), pointer :: \\2{dim1}\\3\\4\\n'
                 f'\\1integer, dimension(:,:), pointer :: \\2{dim2}\\3\\4',line)
    line = re.sub(r'(\s*)_DEF_INDICES_LIST_(\w*)INDEX_NAME(\w*)_(\s*_INTENT_\w*_)?',
                  f'\\1integer, dimension(:), pointer :: \\2{dim1}\\3\\4\\n'
                  f'\\1integer, dimension(:), pointer :: \\2{dim2}\\3\\4',line)
    line = re.sub(r'(\s*)_DEF_COORDS_(\w*)_( _INTENT_\w*_)?',
                  f'\\1integer :: \\2_{dim1}\\3\\n\\1integer :: \\2_{dim2}\\3',line)
    line = re.sub(r'(\w+%)?_INDICES_(?:FIELD|LIST)_(\w*)INDEX_NAME(\w*)_',
                  f'\\1\\2{dim1}\\3,\\1\\2{dim2}\\3',
                  line)
    line = re.sub(r'(\s*)(.*)_EQUALS_?(\w+%)?(\w+%)?_COORDS_(\w*)_ == _COORDS_(HD|LAKE|SURFACE)_',
                  f'\\1\\2(\\3\\4\\5_{dim1} == {dim1}_\\6) .and. &\\n'
                  f'\\1    (\\3\\4\\5_{dim2} == {dim2}_\\6)',
                  line)
    line = re.sub(r'_COORDS_LAKE_',f'{dim1}_lake,{dim2}_lake',line)
    line = re.sub(r'_COORDS_HD_',f'{dim1}_hd,{dim2}_hd',line)
    line = re.sub(r'_COORDS_SURFACE_',f'{dim1}_surface,{dim2}_surface',line)
    line = re.sub(r'(\w+%)?(\w+%)?_COORDS_ARG_(\w*)_',
                  f'\\1\\2\\3_{dim1},\\1\\2\\3_{dim2}',line)
    line = re.sub(r'(\s*)(.*)_EQUALS_?(\w+%)?(\w+%)?_COORDS_(\w*)_ == (\w+%)?(\w+%)?_COORDS_(\w*)_',
                  f'\\1\\2(\\3\\4\\5_{dim1} == \\6\\7\\8_{dim1}) .and. &\\n'
                  f'\\1    (\\3\\4\\5_{dim2} == \\6\\7\\8_{dim2})',
                  line)
    line = re.sub(r'(\s*)(.*)_NEQUALS_(\w+%)?(\w+%)?_COORDS_(\w*)_ /= (\w+%)?(\w+%)?_COORDS_(\w*)_',
                  f'\\1\\2(\\3\\4\\5_{dim1} /= \\6\\7\\8_{dim1}) .or. &\\n'
                  f'\\1    (\\3\\4\\5_{dim2} /= \\6\\7\\8_{dim2})',
                  line)
    line = re.sub(r'(.*)\s+::(.*) _INTENT_(\w*)_',r'\1, intent(\3) ::\2',line)
    if re.match(r'^$',line):
        modified_txt += line
    else:
        for split_line in line.split("\n"):
            if re.match(r'^$',split_line):
                continue
            if len(split_line) > 120:
                split_line = f'{split_line[0:120]}&\n&{split_line[120:]}'
            modified_txt += f'{split_line}\n'

with open(output_file,'w') as f:
    f.write(modified_txt)
