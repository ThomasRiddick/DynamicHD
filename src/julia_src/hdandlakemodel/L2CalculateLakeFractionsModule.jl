module L2CalculateLakeFractionsModule

RENAME TRUE LAKE PIXELS
ADD COUNTS (OTHER SORT RECOUNTS MANY TIMES)

struct LakeProperties
  pixel_count::Integer
  lake::LakePrognostics
  cell_list::Array{LakeCell}
  pixel_list::Array{Pixel}
end

struct Pixel
  id::Integer
  fine_grid_coords::CartesianIndex
  true_coarse_grid_coords::CartesianIndex
  assigned_coarse_grid_coords:CartesianIndex
  transferred::Bool
end

struct LakeCell
  coarse_grid_coords::CartesianIndex
  pixels::Array{CartesianIndex}
  total_all_lake_pixel_count::Integer
  lake_pixels::Dict{Pixel}
  lake_pixels_added::Dict{Pixel}
  pixels_count::Integer
  lake_pixels_count::Integer
  lake_pixels_added_count::Integer
  max_pixels_from_lake::Int64
end

function insert_pixel(cell::LakeCell,pixel::Pixel)
  pixel%assigned_coarse_grid_coords = cell%coarse_grid_coords
  if pixel%true_coarse_grid_coords != cell%coarse_grid_coords
    cell%lake_pixels_added[pixel%id] = pixel
    cell%lake_pixels_added_count += 1
    pixel%transferred = true
  end
  cell%lake_pixels[pixel%id] = pixel
  cell%lake_pixels_count += 1
end

function extract_pixel(cell::LakeCell,pixel::Pixel)
  pixel%assigned_coarse_grid_coords = nothing
  if pixel%true_coarse_grid_coords != cell%coarse_grid_coords
    delete!(cell%lake_pixels_added,pixel%id)
    cell%lake_pixels_added_count -= 1
    pixel%transferred = false
  end
  delete!(cell%lake_pixels,pixel%id)
  cell%lake_pixels_counts -= 1
end

function extract_any_pixel(cell::LakeCell)
  if cell%lake_pixels_count <= 0
    error("No pixels to extract")
  end
  pixel = cell%lake_pixels%keys[1]
  extract_pixel(cell,pixel)
  return pixel
end

function move_pixel(pixel,source_cell,target_cell)
  extract_pixel(source_cell,pixel)
  insert_pixel(target_cell,pixel)
end

dims_lake = (nlat_lake,nlon_lake)
dims_surface = (nlat_surface,nlon_surface)

all_lake_pixels_mask::Array{Bool} = zeros(dims_lake)
s_lake_binary_lake_mask::Array{Bool} = zeros(dims_surfrace)

for lake in lakes
  potential_pixel_list = get_potential_lake_pixel_list(lake)
  cell_list = get_lake_cell_list(lake)
  for pixel in potential_pixel_list
    all_lake_pixels_mask[pixel%fine_grid_coords] = true
  end
  if length(potential_pixel_list) < max(map(x->potential_pixel_count[x],cell_list))
    continue
  pixel_list = get_lake_pixel_list(lake)
  setup lake
end
sort!(lakes,by=x->length(x%pixel_list),rev=true)
calculate_total_all_lake_pixel_counts(lakes)

CELLS MUST BE SORTED BY FRACTION
for lake in lakes
  sort!(lake%cell_list,by=x->x%lake_pixels_count)
  j = length(lake%cell_list)
  unprocessed_cells_total_pixel_count = 0
  for cell in lake%cell_list
    unprocessed_cells_total_pixel_count += cell%lake_pixels_count
  end
  if unprocessed_cells_total_pixel_count < 0.5*cell%pixels_count
    continue
  end
  for i = 1:length(lake%cell_list)
    if i == j
      break
    end
    cell = lake%cell_list[i]
    unprocessed_cells_total_pixel_count -= cell%lake_pixels_count
    cell%max_pixels_from_lake = length(cell%true_pixels) + cell%pixels_count -
                                cell%total_all_lake_pixel_count
    if cell%max_pixels_from_lake == cell%lake_pixels_count
      continue
    end
    if cell%max_pixels_from_lake < 0.5*cell%pixels_count
      continue
    end
    if unprocessed_cells_total_pixel_count + cell%lake_pixels_count < 0.5*cell%pixels_count
      break
    end
    while true
      least_filled_cell = lake%cell_list[j]
      if cell%lake_pixels_count + least_filled_cell%lake_pixels_count  <= cell%max_pixels_from_lake
        for pixel in least_filled_cell%lake_pixels%values
          move_pixel(pixel,least_filled_cell,cell)
          unprocessed_cells_total_pixel_count -= least_filled_cell%lake_pixels_count
        end
        j -= 1
        if cell%lake_pixels_count == cell%max_pixels_from_lake
          break
        end
      else
        pixels_to_transfer = cell%max_pixels_from_lake - cell%lake_pixels_count
        for pixel_id in least_filled_cell%lake_pixels%keys[1:pixels_to_transfer]
          pixel = least_filled_cell%lake_pixels[pixel_id]
          move_pixel(pixel,least_filled_cell,cell)
          unprocessed_cells_total_pixel_count -= pixels_to_transfer
        break
      end
    end
  end
end

MOST FILLED CELL NEEDS TO BE BY FRACTION

function add_pixel(cell,pixel)
  if ANY CELL IN LAKE ARE ABOVE IN SLAKE MASK
  if cell%max_pixels_from_lake == cell%lake_pixels_count
    other_pixel = nothing
    other_pixel_origin_cell = nothing
    max_other_pixel_origin_cell_lake_pixel_count = -1
    for pixel in cell%lake_pixels_added%values
      working_pixel_origin_cell = get_lake_cell_from_coords(lake,pixel%true_coarse_grid_coords)
      if working_pixel_origin_cell%lake_cell_count > max_other_pixel_origin_cell_lake_pixel_count
        other_pixel = pixel
        other_pixel_origin_cell =  working_pixel_origin_cell
        max_other_pixel_origin_cell_lake_pixel_count = working_pixel_origin_cell%lake_cell_count
      end
    end
    if isnothing(other_pixel)
      error("No added pixel to return - logic error")
    end
    add_pixel(other_pixel_origin_cell,other_pixel)
    insert_pixel(cell,pixel)
  else if cell%lake_pixels_count >= 0.5*cell%pixels_count
    insert_pixel(cell,pixel)
  else
    most_filled_cell = nothing
    most_filled_cell_pixel_count = 0
    for other_cell in lake%cells_list
      if other_cell%lake_pixels_count > most_filled_cell_pixel_count &&
         other_cell%lake_pixels_count < other_cell%max_pixels_from_lake
        most_filled_cell = other_cell
        most_filled_cell_pixel_count = other_cell%lake_pixels_count
      end
    end
    if isnothing(most_filled_cell)
      insert_pixel(cell,pixel)
    else if most_filled_cell%lake_pixels_count >= 0.5*most_filled_cell%pixels_count
      insert_pixel(most_filled_cell,pixel)
    else
      insert_pixel(cell,pixel)
    end
  end
end

use in_slake_mask consistently
LEAST MUST BE BY FRACTION
function remove_pixel(cell,pixel)
  extract_pixel(cell,pixel)
  if cell%lake_pixels_count >= 0.5*cell%pixels_count
    for other_cell in lake%cell_list
      least_filled_cell_pixel_count = total_fine_grid_points
      least_filled_cell = nothing
      if other_cell%lake_pixels_count < least_filled_cell_pixel_count &&
         ! in_slake_mask(other_cell) && other_cell%lake_pixels_count
         least_filled_cell = other_cell%lake_pixels_count
         least_filled_cell = other_cell
      end
    end
    if ! isnothing(least_filled_cell)
      pixel = extract_any_pixel(least_filled_cell)
      insert_pixel(cell,pixel)
    end
  end
end
