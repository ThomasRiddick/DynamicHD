module L2CalculateLakeFractionsModule

using GridModule: Grid, for_all,for_all_fine_cells_in_coarse_cell,get
using GridModule: find_coarse_cell_containing_fine_cell
using FieldModule: Field,set!,elementwise_divide
using L2LakeModelGridSpecificDefsModule: GridSpecificLakeModelParameters
using L2LakeModelGridSpecificDefsModule: get_corresponding_surface_model_grid_cell

#pass in total all lake cell pixel count from outside then skip
#lakes that are too small to change

#look into breaking when not more elgible cells are left ...
#is this okay or should it be continue?

mutable struct Pixel
  id::Int64
  lake_number::Int64
  filled::Bool
  fine_grid_coords::CartesianIndex
  original_coarse_grid_coords::CartesianIndex
  assigned_coarse_grid_coords::CartesianIndex
  transferred::Bool
end

mutable struct LakeCell
  coarse_grid_coords::CartesianIndex
  all_lake_potential_pixel_count::Int64
  lake_potential_pixel_count::Int64
  lake_pixels::Dict{Int64,Pixel}
  lake_pixels_added::Dict{Int64,Pixel}
  pixel_count::Int64
  lake_pixel_count::Int64
  lake_pixels_added_count::Int64
  max_pixels_from_lake::Int64
  in_binary_mask::Bool
  function LakeCell(coarse_grid_coords::CartesianIndex,
                    pixel_count::Int64,
                    potential_lake_pixel_count::Int64,
                    lake_pixels_original::Dict{Int64,Pixel})
    return new(coarse_grid_coords,-1,
               potential_lake_pixel_count,
               lake_pixels_original,
               Dict{Int64,Pixel}(),
               pixel_count,
               length(lake_pixels_original),
               0,
               -1,
               false)
  end
end

mutable struct LakeInput
  lake_number::Int64
  lake_pixel_coords_list::Vector{CartesianIndex}
  potential_lake_pixel_coords_list::Vector{CartesianIndex}
  cell_coords_list::Vector{CartesianIndex}
end

mutable struct LakeProperties
  lake_number::Int64
  cell_list::Vector{LakeCell}
  lake_pixel_count::Int64
  has_cells_in_binary_mask::Bool
  function LakeProperties(lake_number::Int64,
                          cell_list::Vector{LakeCell},
                          lake_pixel_count::Int64)
    return new(lake_number,cell_list,lake_pixel_count,false)
  end
end

struct LakeFractionCalculationPrognostics
  lakes::Vector{LakeProperties}
  pixel_numbers::Field{Int64}
  pixels::Vector{Pixel}
  primary_lake_numbers::Field{Int64}
  lake_index::Field{Int64}
  non_lake_filled_pixel_count_field::Field{Int64}
  grid_specific_lake_model_parameters::GridSpecificLakeModelParameters
  function LakeFractionCalculationPrognostics(
      lakes::Vector{LakeProperties},
      pixel_numbers::Field{Int64},
      pixels::Vector{Pixel},
      primary_lake_numbers::Field{Int64},
      grid_specific_lake_model_parameters::GridSpecificLakeModelParameters,
      lake_grid::Grid)
    return new(lakes,pixel_numbers,pixels,
               primary_lake_numbers,
               Field{Int64}(lake_grid,0),
               Field{Int64}(lake_grid,0),
               grid_specific_lake_model_parameters)
  end
  function LakeFractionCalculationPrognostics(
      grid_specific_lake_model_parameters::GridSpecificLakeModelParameters,
      lake_grid::Grid)
    return new(LakeProperties[],
               Field{Int64}(lake_grid,0),
               Pixel[],
               Field{Int64}(lake_grid,0),
               Field{Int64}(lake_grid,0),
               Field{Int64}(lake_grid,0),
               grid_specific_lake_model_parameters)
  end
end

function insert_pixel(cell::LakeCell,pixel::Pixel)
  pixel.assigned_coarse_grid_coords = cell.coarse_grid_coords
  if pixel.original_coarse_grid_coords != cell.coarse_grid_coords
    cell.lake_pixels_added[pixel.id] = pixel
    cell.lake_pixels_added_count += 1
    pixel.transferred = true
  end
  cell.lake_pixels[pixel.id] = pixel
  cell.lake_pixel_count += 1
  ##println("inserting pixel")
end

function extract_pixel(cell::LakeCell,pixel::Pixel)
  if pixel.original_coarse_grid_coords != cell.coarse_grid_coords
    delete!(cell.lake_pixels_added,pixel.id)
    cell.lake_pixels_added_count -= 1
    pixel.transferred = false
  end
  delete!(cell.lake_pixels,pixel.id)
  cell.lake_pixel_count -= 1
  #println("extracting pixel")
end

function extract_any_pixel(cell::LakeCell)
  if cell.lake_pixel_count <= 0
    error("No pixels to extract")
  end
  local pixel::Pixel
  for key in keys(cell.lake_pixels)
    pixel_id::Int64 = key
    pixel = cell.lake_pixels[pixel_id]
    break
  end
  extract_pixel(cell,pixel)
  return pixel
end

function move_pixel(pixel,source_cell,target_cell)
  extract_pixel(source_cell,pixel)
  insert_pixel(target_cell,pixel)
end

function get_lake_cell_from_coords(lake::LakeProperties,
                                   coords::CartesianIndex)
  for cell in lake.cell_list
    if cell.coarse_grid_coords == coords
      return cell
    end
  end
end

function setup_cells_lakes_and_pixels(lakes::Vector{LakeInput},
                                      cell_pixel_counts::Field{Int64},
                                      all_lake_potential_pixel_mask::Field{Bool},
                                      lake_properties::Vector{LakeProperties},
                                      pixel_numbers::Field{Int64},
                                      pixels::Vector{Pixel},
                                      all_lake_potential_pixel_counts::Field{Int64},
                                      grid_specific_lake_model_parameters::GridSpecificLakeModelParameters,
                                      lake_grid::Grid)
  all_lake_total_pixels::Int64 = 0
  for lake::LakeInput in lakes::Vector{LakeInput}
    cell_coords_list = lake.cell_coords_list
    lake_cells::Vector{LakeCell} = LakeCell[]
    for pixel_coords in lake.potential_lake_pixel_coords_list
      all_lake_total_pixels += 1
      set!(pixel_numbers,pixel_coords,all_lake_total_pixels)
      containing_cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel_coords,
                                                  grid_specific_lake_model_parameters)
      pixel::Pixel = Pixel(all_lake_total_pixels,lake.lake_number,false,
                           pixel_coords,deepcopy(containing_cell_coords),
                           deepcopy(containing_cell_coords),false)
      push!(pixels,pixel)
      set!(all_lake_potential_pixel_mask,pixel.fine_grid_coords,true)
    end
    for pixel_coords in lake.lake_pixel_coords_list
      pixels[pixel_numbers(pixel_coords)].filled = true
    end
    for cell_coords in cell_coords_list
      pixels_in_cell::Dict{Int64,Pixel} = Dict{Int64,Pixel}()
      potential_lake_pixel_count::Int64 = 0
      for_all(lake_grid; use_cartesian_index=true) do fine_coords::CartesianIndex
        coarse_coords::CartesianIndex =
          get_corresponding_surface_model_grid_cell(fine_coords,
                                                    grid_specific_lake_model_parameters)
        if coarse_coords == cell_coords
          pixel_number::Int64 = pixel_numbers(fine_coords)
          if pixel_number > 0
            if pixels[pixel_number].fine_grid_coords == fine_coords &&
                pixels[pixel_number].lake_number == lake.lake_number
                potential_lake_pixel_count += 1
              if pixels[pixel_number].filled
                pixels_in_cell[pixel_number] = pixels[pixel_number]
              end
            end
          end
        end
      end
      push!(lake_cells,LakeCell(cell_coords,cell_pixel_counts(cell_coords),
                                potential_lake_pixel_count,pixels_in_cell))
    end
    lake_cell_pixel_count::Int64 = length(lake.lake_pixel_coords_list)
    push!(lake_properties,LakeProperties(lake.lake_number,
                                         lake_cells,
                                         lake_cell_pixel_count))
  end
  for_all(lake_grid; use_cartesian_index=true) do fine_coords::CartesianIndex
    if all_lake_potential_pixel_mask(fine_coords)
      coarse_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(fine_coords,
                                                  grid_specific_lake_model_parameters)
      set!(all_lake_potential_pixel_counts,coarse_coords,
           all_lake_potential_pixel_counts(coarse_coords) + 1)
    end
  end
  for lake::LakeProperties in lake_properties::Vector{LakeProperties}
    for cell in lake.cell_list
      cell.all_lake_potential_pixel_count =
        all_lake_potential_pixel_counts(cell.coarse_grid_coords)
    end
  end
  return all_lake_total_pixels
end

function set_max_pixels_from_lake_for_cell(cell::LakeCell,
                                           non_lake_filled_pixel_count_field::Field{Int64})
  other_lake_potential_pixels::Int64 = cell.all_lake_potential_pixel_count -
                                       cell.lake_potential_pixel_count
   non_lake_filled_pixel_count::Int64 = cell.lake_pixel_count -
                                       cell.lake_potential_pixel_count
  if  non_lake_filled_pixel_count < 0
     non_lake_filled_pixel_count = 0
  end
  other_lake_filled_non_lake_pixels::Int64 =
    non_lake_filled_pixel_count_field(cell.coarse_grid_coords) -
    non_lake_filled_pixel_count
  if other_lake_filled_non_lake_pixels < 0
    error("Non lake filled pixel count logic error")
  end
  cell.max_pixels_from_lake =  cell.pixel_count - other_lake_potential_pixels - other_lake_filled_non_lake_pixels
end

function calculate_lake_fractions(lakes::Vector{LakeInput},
                                  cell_pixel_counts::Field{Int64},
                                  grid_specific_lake_model_parameters::GridSpecificLakeModelParameters,
                                  lake_grid::Grid,
                                  surface_grid::Grid)
  all_lake_potential_pixel_mask::Field{Bool} =  Field{Bool}(lake_grid,false)
  lake_properties::Vector{LakeProperties} = LakeProperties[]
  pixel_numbers::Field{Int64} = Field{Int64}(lake_grid,0)
  pixels::Vector{Pixel} = Pixel[]
  all_lake_potential_pixel_counts::Field{Int64} = Field{Int64}(surface_grid,0)
  lake_pixel_counts_field::Field{Int64} =  Field{Int64}(surface_grid,0)
  non_lake_filled_pixel_count_field::Field{Int64} =  Field{Int64}(surface_grid,0)
  all_lake_total_pixels::Int64 =
    setup_cells_lakes_and_pixels(lakes,cell_pixel_counts,
                                 all_lake_potential_pixel_mask,
                                 lake_properties,
                                 pixel_numbers,pixels,
                                 all_lake_potential_pixel_counts,
                                 grid_specific_lake_model_parameters,
                                 lake_grid)
  sort!(lake_properties,by=x->x.lake_pixel_count,rev=true)
  for lake::LakeProperties in lake_properties::Vector{LakeProperties}
    for cell in lake.cell_list
      set_max_pixels_from_lake_for_cell(cell,non_lake_filled_pixel_count_field)
    end
    sort!(lake.cell_list,by=x->x.lake_pixel_count/x.pixel_count,rev=true)
    j = length(lake.cell_list)
    unprocessed_cells_total_pixel_count = 0
    for cell in lake.cell_list
      unprocessed_cells_total_pixel_count += cell.lake_pixel_count
    end
    if unprocessed_cells_total_pixel_count <
        0.5*minimum(map(cell->cell.pixel_count,lake.cell_list))
      for cell in lake.cell_list
        set!(lake_pixel_counts_field,cell.coarse_grid_coords,
        lake_pixel_counts_field(cell.coarse_grid_coords)+cell.lake_pixel_count)
      end
      continue
    end
    for i = 1:length(lake.cell_list)
      if i == j
        break
      end
      cell = lake.cell_list[i]
      unprocessed_cells_total_pixel_count -= cell.lake_pixel_count
      if unprocessed_cells_total_pixel_count + cell.lake_pixel_count < 0.5*cell.pixel_count
        break
      end
      if cell.max_pixels_from_lake == cell.lake_pixel_count
        continue
      end
      if cell.max_pixels_from_lake < 0.5*cell.pixel_count
        continue
      end
      while true
        least_filled_cell = lake.cell_list[j]
        if cell.lake_pixel_count + least_filled_cell.lake_pixel_count  <= cell.max_pixels_from_lake
          for pixel in values(least_filled_cell.lake_pixels)
            move_pixel(pixel,least_filled_cell,cell)
            unprocessed_cells_total_pixel_count -= 1
          end
          j -= 1
          if cell.lake_pixel_count == cell.max_pixels_from_lake
            break
          end
          if i == j
            break
          end
        else
          pixels_to_transfer = cell.max_pixels_from_lake - cell.lake_pixel_count
          for (k,pixel_id) in enumerate(keys(least_filled_cell.lake_pixels))
            if k > pixels_to_transfer
              break
            end
            pixel = least_filled_cell.lake_pixels[pixel_id]
            move_pixel(pixel,least_filled_cell,cell)
            unprocessed_cells_total_pixel_count -= 1
          end
          break
        end
      end
    end
    for cell in lake.cell_list
      set!(lake_pixel_counts_field,cell.coarse_grid_coords,
           lake_pixel_counts_field(cell.coarse_grid_coords)+cell.lake_pixel_count)
      overflow_count::Int64 = cell.lake_pixel_count - cell.lake_potential_pixel_count
      if overflow_count > 0
        set!(non_lake_filled_pixel_count_field,cell.coarse_grid_coords,
             non_lake_filled_pixel_count_field(cell.coarse_grid_coords)+overflow_count)
      end
    end
  end
  lake_fractions_field::Field{Float64} =
    elementwise_divide(lake_pixel_counts_field,
                       cell_pixel_counts)
  binary_lake_mask::Field{Bool} =
    lake_fractions_field >= 0.5
  return lake_pixel_counts_field,lake_fractions_field,binary_lake_mask
end

function setup_lake_for_fraction_calculation(lakes::Vector{LakeInput},
                                             cell_pixel_counts::Field{Int64},
                                             binary_lake_mask::Field{Bool},
                                             primary_lake_numbers::Field{Int64},
                                             grid_specific_lake_model_parameters::
                                             GridSpecificLakeModelParameters,
                                             lake_grid::Grid,
                                             surface_grid::Grid)
  all_lake_potential_pixel_mask::Field{Bool} =  Field{Bool}(lake_grid,false)
  lake_properties::Vector{LakeProperties} = LakeProperties[]
  pixel_numbers::Field{Int64} = Field{Int64}(lake_grid,0)
  pixels::Vector{Pixel} = Pixel[]
  all_lake_total_pixels::Int64 = 0
  all_lake_potential_pixel_counts::Field{Int64} = Field{Int64}(surface_grid,0)
  non_lake_filled_pixel_count_field::Field{Int64} =  Field{Int64}(surface_grid,0)
  lake_pixel_counts_field::Field{Int64} =  Field{Int64}(surface_grid,0)
  setup_cells_lakes_and_pixels(lakes,cell_pixel_counts,
                               all_lake_potential_pixel_mask,
                               lake_properties,
                               pixel_numbers,pixels,
                               all_lake_potential_pixel_counts,
                               grid_specific_lake_model_parameters,
                               lake_grid)
  for lake in lake_properties
    for cell in lake.cell_list
      cell.in_binary_mask = binary_lake_mask(cell.coarse_grid_coords)
      if cell.in_binary_mask
        lake.has_cells_in_binary_mask = true
      end
    end
  end
  return LakeFractionCalculationPrognostics(lake_properties,
                                            pixel_numbers,
                                            pixels,
                                            primary_lake_numbers,
                                            grid_specific_lake_model_parameters,
                                            lake_grid)
end

function add_pixel(lake::LakeProperties,pixel::Pixel,
                   lake_pixel_counts_field::Field{Int64},
                   non_lake_filled_pixel_count_field::Field{Int64})
  local cell::LakeCell
  local overflow_count::Int64
  for entry in lake.cell_list
    if entry.coarse_grid_coords == pixel.original_coarse_grid_coords
      cell = entry
    end
  end
  set_max_pixels_from_lake_for_cell(cell,non_lake_filled_pixel_count_field)
  if cell.max_pixels_from_lake == cell.lake_pixel_count
    local other_pixel::Pixel
    local other_pixel_origin_cell::LakeCell
    max_other_pixel_origin_cell_lake_fraction::Int64 = -1
    for working_pixel in values(cell.lake_pixels_added)
      working_pixel_origin_cell = get_lake_cell_from_coords(lake,working_pixel.original_coarse_grid_coords)
      working_pixel_origin_cell_lake_fraction::Float64 =
        working_pixel_origin_cell.lake_pixel_count/
        working_pixel_origin_cell.pixel_count
      if working_pixel_origin_cell_lake_fraction > max_other_pixel_origin_cell_lake_fraction
        other_pixel = working_pixel
        other_pixel_origin_cell =  working_pixel_origin_cell
        max_other_pixel_origin_cell_lake_fraction = working_pixel_origin_cell.lake_pixel_count
      end
    end
    if max_other_pixel_origin_cell_lake_fraction == -1
      error("No added pixel to return - logic error")
    end
    extract_pixel(cell,other_pixel)
    #Order of next two statements is critical to prevent loops
    insert_pixel(cell,pixel)
    add_pixel(lake,other_pixel,lake_pixel_counts_field,
              non_lake_filled_pixel_count_field)
  elseif cell.max_pixels_from_lake < cell.lake_pixel_count
    error("Cell has more pixel than possible - logic error")
  elseif cell.in_binary_mask
    insert_pixel(cell,pixel)
    set!(lake_pixel_counts_field,cell.coarse_grid_coords,
         lake_pixel_counts_field(cell.coarse_grid_coords)+1)
    overflow_count = cell.lake_pixel_count - cell.lake_potential_pixel_count
    if overflow_count > 0
      set!(non_lake_filled_pixel_count_field,cell.coarse_grid_coords,
           non_lake_filled_pixel_count_field(cell.coarse_grid_coords)+1)
    end
  else
    most_filled_cell = cell
    most_filled_cell_lake_fraction = -1.0
    for other_cell in lake.cell_list
      set_max_pixels_from_lake_for_cell(other_cell,non_lake_filled_pixel_count_field)
      other_cell_lake_fraction = other_cell.lake_pixel_count/
                                 other_cell.pixel_count
      if other_cell_lake_fraction > most_filled_cell_lake_fraction &&
         other_cell.lake_pixel_count < other_cell.max_pixels_from_lake &&
         other_cell.in_binary_mask
        most_filled_cell = other_cell
        most_filled_cell_lake_fraction = other_cell_lake_fraction
      end
    end
    if most_filled_cell_lake_fraction >= 0.0
      insert_pixel(most_filled_cell,pixel)
      set!(lake_pixel_counts_field,most_filled_cell.coarse_grid_coords,
           lake_pixel_counts_field(most_filled_cell.coarse_grid_coords)+1)
      overflow_count = most_filled_cell.lake_pixel_count -
                       most_filled_cell.lake_potential_pixel_count
      if overflow_count > 0
        set!(non_lake_filled_pixel_count_field,most_filled_cell.coarse_grid_coords,
             non_lake_filled_pixel_count_field(most_filled_cell.coarse_grid_coords)+1)
      end
    else
      insert_pixel(cell,pixel)
      set!(lake_pixel_counts_field,cell.coarse_grid_coords,
           lake_pixel_counts_field(cell.coarse_grid_coords)+1)
      overflow_count = cell.lake_pixel_count - cell.lake_potential_pixel_count
      if overflow_count > 0
        set!(non_lake_filled_pixel_count_field,cell.coarse_grid_coords,
             non_lake_filled_pixel_count_field(cell.coarse_grid_coords)+1)
      end
    end
  end
end

function remove_pixel(lake::LakeProperties,pixel::Pixel,
                      lake_pixel_counts_field::Field{Int64},
                      non_lake_filled_pixel_count_field::Field{Int64})
  local cell::LakeCell
  for entry in lake.cell_list
    if entry.coarse_grid_coords == pixel.assigned_coarse_grid_coords
      cell = entry
    end
  end
  overflow_count::Int64 = cell.lake_pixel_count - cell.lake_potential_pixel_count
  if overflow_count > 0
    set!(non_lake_filled_pixel_count_field,cell.coarse_grid_coords,
         non_lake_filled_pixel_count_field(cell.coarse_grid_coords)-1)
  end
  extract_pixel(cell,pixel)
  set!(lake_pixel_counts_field,cell.coarse_grid_coords,
       lake_pixel_counts_field(cell.coarse_grid_coords)-1)
  if cell.in_binary_mask
    local least_filled_cell_lake_fraction::Float64 = 999.0
    local least_filled_cell::LakeCell
    for other_cell in lake.cell_list
      other_cell_lake_fraction = other_cell.lake_pixel_count/
                                 other_cell.pixel_count
      if ! other_cell.in_binary_mask && other_cell.lake_pixel_count > 0 &&
          other_cell_lake_fraction < least_filled_cell_lake_fraction
         least_filled_cell_lake_fraction = other_cell_lake_fraction
         least_filled_cell = other_cell
      end
    end
    if least_filled_cell_lake_fraction < 999.0
      overflow_count = least_filled_cell.lake_pixel_count -
                       least_filled_cell.lake_potential_pixel_count
      if overflow_count > 0
        set!(non_lake_filled_pixel_count_field,least_filled_cell.coarse_grid_coords,
             non_lake_filled_pixel_count_field(
                least_filled_cell.coarse_grid_coords)-1)
      end
      other_pixel = extract_any_pixel(least_filled_cell)
      set!(lake_pixel_counts_field,least_filled_cell.coarse_grid_coords,
       lake_pixel_counts_field(least_filled_cell.coarse_grid_coords)-1)
      insert_pixel(cell,other_pixel)
      set!(lake_pixel_counts_field,cell.coarse_grid_coords,
       lake_pixel_counts_field(cell.coarse_grid_coords)+1)
      overflow_count = cell.lake_pixel_count - cell.lake_potential_pixel_count
      if overflow_count > 0
        set!(non_lake_filled_pixel_count_field,cell.coarse_grid_coords,
             non_lake_filled_pixel_count_field(cell.coarse_grid_coords)+1)
      end
    end
  end
end

function add_pixel_by_coords(pixel_coords::CartesianIndex,
                             lake_pixel_counts_field::Field{Int64},
                             prognostics::
                             LakeFractionCalculationPrognostics)
  local lake_in::LakeProperties
  lake_index::Int64 = prognostics.lake_index(pixel_coords)
  if  lake_index == 0
    lake_number = prognostics.primary_lake_numbers(pixel_coords)
    for (i,working_lake) in pairs(IndexLinear(),prognostics.lakes)
      if working_lake.lake_number == lake_number
        lake_in = working_lake
        set!(prognostics.lake_index,pixel_coords,i)
      end
    end
  else
    lake_in = prognostics.lakes[lake_index]
  end
  if lake_in.has_cells_in_binary_mask
    pixel_in::Pixel = prognostics.pixels[prognostics.pixel_numbers(pixel_coords)]
    add_pixel(lake_in,pixel_in,
              lake_pixel_counts_field,
              prognostics.non_lake_filled_pixel_count_field)
  else
    surface_model_coords::CartesianIndex =
      get_corresponding_surface_model_grid_cell(pixel_coords,
        prognostics.grid_specific_lake_model_parameters)
      set!(lake_pixel_counts_field,surface_model_coords,
           lake_pixel_counts_field(surface_model_coords) + 1)
  end

end

function remove_pixel_by_coords(pixel_coords::CartesianIndex,
                                lake_pixel_counts_field::Field{Int64},
                                prognostics::
                                LakeFractionCalculationPrognostics)
  local lake_in::LakeProperties
  lake_index::Int64 = prognostics.lake_index(pixel_coords)
  if  lake_index == 0
    lake_number = prognostics.primary_lake_numbers(pixel_coords)
    for (i,working_lake) in pairs(IndexLinear(),prognostics.lakes)
      if working_lake.lake_number == lake_number
        lake_in = working_lake
        set!(prognostics.lake_index,pixel_coords,i)
      end
    end
  else
    lake_in = prognostics.lakes[lake_index]
  end
  if lake_in.has_cells_in_binary_mask
    pixel_in::Pixel = prognostics.pixels[prognostics.pixel_numbers(pixel_coords)]
    remove_pixel(lake_in,pixel_in,
                 lake_pixel_counts_field,
                 prognostics.non_lake_filled_pixel_count_field)
  else
    surface_model_coords::CartesianIndex =
      get_corresponding_surface_model_grid_cell(pixel_coords,
                                                prognostics.grid_specific_lake_model_parameters)
      set!(lake_pixel_counts_field,surface_model_coords,
           lake_pixel_counts_field(surface_model_coords) - 1)
  end
end

end
