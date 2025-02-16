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
  pixel_id::Integer
  fine_grid_coords::CartesianIndex
  true_coarse_grid_coords::CartesianIndex
  assigned_coarse_grid_coords:CartesianIndex
  transferred::Bool
end

struct LakeCell
  coarse_grid_coords::CartesianIndex
  pixels::Array{CartesianIndex}
  total_all_lake_pixel_count::Integer
  lake_pixels::Array{Pixel}
  lake_pixels_added::Array{Pixel}
  max_pixels_from_lake::Int64
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
  sort!(lake%cell_list,by=x->length(x%lake_pixels))
  j = length(lake%cell_list)
  unprocessed_cells_total_pixel_count = 0
  for cell in lake%cell_list
    unprocessed_cells_total_pixel_count += length(cell%lake_pixels)
  end
  if unprocessed_cells_total_pixel_count < 0.5*length(cell%pixels)
    continue
  end
  for i = 1:length(lake%cell_list)
    if i == j
      break
    end
    cell = lake%cell_list[i]
    unprocessed_cells_total_pixel_count -= length(cell%lake_pixels)
    cell%max_pixels_from_lake = length(cell%true_pixels) + length(cell%pixels) -
                                length(total_all_lake_pixel_count)
    if cell%max_pixels_from_lake == length(cell%lake_pixels)
      continue
    end
    if cell%max_pixels_from_lake < 0.5*length(cell%pixels)
      continue
    end
    if unprocessed_cells_total_pixel_count + length(cell%lake_pixels) < 0.5*length(cell%pixels)
      break
    end
    while true
      least_filled_cell = lake%cell_list[j]
      if length(cell%lake_pixels) + length(least_filled_cell%lake_pixels)  <= cell%max_pixels_from_lake
        for pixel in least_filled_cell%lake_pixels
          move_pixel(pixel,least_filled_cell,cell)
          unprocessed_cells_total_pixel_count -= length(least_filled_cell%lake_pixels)
        end
        j -= 1
        if length(cell%lake_pixels) == cell%max_pixels_from_lake
          break
        end
      else
        pixels_to_transfer = cell%max_pixels_from_lake - length(cell%lake_pixels)
        for pixel in least_filled_cell%lake_pixels[1:pixels_to_transfer]
          move_pixel(pixel,least_filled_cell,cell)
          unprocessed_cells_total_pixel_count -= pixels_to_transfer
        break
      end
    end
  end
end


REMOVE PIXEL AT TOP OF ADDED PIXEL LIST (FROM LEAST FILLED CELL)
MOST FILLED CELL NEEDS TO BE BY FRACTION

function add_pixel(cell,pixel)
  if ANY CELL IN LAKE ARE ABOVE IN SLAKE MASK
  if cell%max_pixels_from_lake == length(cell%lake_pixels)
    other_pixel = remove_top_added_pixel(cell)
    other_pixel_origin_cell = get_origin(other_pixel)
    add_pixel(other_pixel_origin_cell,other_pixel)
    insert_pixel(cell,pixel)
  else if length(cell%lake_pixels) => 0.5*length(cell%pixels)
    insert_pixel(cell,pixel)
  else
    most_filled_cell = nothing
    most_filled_cell_pixel_count = 0
    for other_cell in lake%cells_list
      if length(other_cell%lake_pixels) > most_filled_cell_pixel_count &&
         length(other_cell%lake_pixels) < other_cell%max_pixels_from_lake
        most_filled_cell = other_cell
        most_filled_cell_pixel_count = length(other_cell%lake_pixels)
      end
    end
    if most_filled_cell EQUALS NOTHING??
      insert_pixel(cell,pixel)
    else if length(most_filled_cell%lake_pixels) >= 0.5*length(most_filled_cell%pixels)
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
  if length(cell%lake_pixels) >= 0.5*length(cell%pixels)
    for other_cell in lake%cell_list
      least_filled_cell_pixel_count = total_fine_grid_points
      least_filled_cell = nothing
      if length(other_cell%lake_pixels) < least_filled_cell_pixel_count &&
         ! in_slake_mask(other_cell)
         least_filled_cell = length(other_cell%lake_pixels)
         least_filled_cell = other_cell
      end
    end
    if ! least_filled_cell EQUALS NOTHING
      pixel = extract_any_pixel(least_filled_cell)
      insert_pixel(cell,pixel)
    end
  end
end
