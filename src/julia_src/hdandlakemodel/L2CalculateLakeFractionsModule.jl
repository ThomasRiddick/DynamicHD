module L2CalculateLakeFractionsModule

using GridModule: Grid, for_all,for_all_fine_cells_in_coarse_cell,get
using GridModule: find_coarse_cell_containing_fine_cell
using FieldModule: Field,set!

#pass in total all lake cell pixel count from outside then skip
#lakes that are too small to change

#look into breaking when not more elgible cells are left ...
#is this okay or should it be continue?

mutable struct Pixel
  id::Int64
  lake_number::Int64
  fine_grid_coords::CartesianIndex
  original_coarse_grid_coords::CartesianIndex
  assigned_coarse_grid_coords::CartesianIndex
  transferred::Bool
end

mutable struct LakeCell
  coarse_grid_coords::CartesianIndex
  total_all_lake_pixel_count::Int64
  lake_pixels::Dict{Int64,Pixel}
  lake_pixels_added::Dict{Int64,Pixel}
  pixel_count::Int64
  original_lake_pixel_count::Int64
  lake_pixel_count::Int64
  lake_pixels_added_count::Int64
  max_pixels_from_lake::Int64
  contains_lakes_in_slake_mask::Bool
  function LakeCell(coarse_grid_coords::CartesianIndex,
                    pixel_count::Int64,
                    lake_pixels_original::Dict{Int64,Pixel})
    return new(coarse_grid_coords,-1,
               lake_pixels_original,
               Dict{Int64,Pixel}(),
               pixel_count,
               length(lake_pixels_original),
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

struct LakeProperties
  lake_number::Int64
  cell_list::Vector{LakeCell}
  lake_pixel_count::Int64
  function LakeProperties(lake_number::Int64,
                          cell_list::Vector{LakeCell},
                          lake_pixel_count::Int64)
    return new(lake_number,cell_list,lake_pixel_count)
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
  println("inserting pixel")
end

function extract_pixel(cell::LakeCell,pixel::Pixel)
  if pixel.original_coarse_grid_coords != cell.coarse_grid_coords
    delete!(cell.lake_pixels_added,pixel.id)
    cell.lake_pixels_added_count -= 1
    pixel.transferred = false
  end
  delete!(cell.lake_pixels,pixel.id)
  cell.lake_pixel_count -= 1
  println("extracting pixel")
end

function extract_any_pixel(cell::LakeCell)
  if cell.lake_pixel_count <= 0
    error("No pixels to extract")
  end
  pixel = keys(cell.lake_pixels)[1]
  extract_pixel(cell,pixel)
  return pixel
end

function move_pixel(pixel,source_cell,target_cell)
  extract_pixel(source_cell,pixel)
  insert_pixel(target_cell,pixel)
end


function calculate_lake_fractions(lakes::Vector{LakeInput},
                                  cell_pixel_counts::Field{Int64},
                                  lake_grid::Grid,
                                  surface_grid::Grid)
  all_lake_pixel_mask::Field{Bool} =  Field{Bool}(lake_grid,false)
  binary_lake_mask::Field{Bool} = Field{Bool}(surface_grid,false)
  lake_properties::Vector{LakeProperties} = LakeProperties[]
  pixel_numbers::Field{Int64} = Field{Int64}(lake_grid,0)
  pixels::Vector{Pixel} = Pixel[]
  all_lake_total_pixels::Int64 = 0
  for lake::LakeInput in lakes::Vector{LakeInput}
    cell_coords_list = lake.cell_coords_list
    lake_cells::Vector{LakeCell} = LakeCell[]
    for pixel_coords in lake.potential_lake_pixel_coords_list
      all_lake_total_pixels += 1
      set!(pixel_numbers,pixel_coords,all_lake_total_pixels)
      in_cell::CartesianIndex =
        find_coarse_cell_containing_fine_cell(lake_grid,surface_grid,
                                              pixel_coords)
      pixel::Pixel = Pixel(all_lake_total_pixels,lake.lake_number,
                           pixel_coords,in_cell,in_cell,false)
      push!(pixels,pixel)
      set!(all_lake_pixel_mask,pixel.fine_grid_coords,true)
    end
    for cell_coords in cell_coords_list
      pixels_in_cell::Dict{Int64,Pixel} = Dict{Int64,Pixel}()
      for_all_fine_cells_in_coarse_cell(lake_grid,surface_grid,
                                        cell_coords) do fine_coords::CartesianIndex
        pixel_number::Int64 = pixel_numbers(fine_coords)
        if pixel_number > 0
          if pixels[pixel_number].fine_grid_coords == fine_coords &&
              pixels[pixel_number].lake_number == lake.lake_number
            pixels_in_cell[pixel_number] = pixels[pixel_number]
          end
        end
      end
      push!(lake_cells,LakeCell(cell_coords,cell_pixel_counts(cell_coords),pixels_in_cell))
    end
    lake_cell_pixel_count::Int64 = 0
    for pixel_coords in lake.lake_pixel_coords_list
      lake_cell_pixel_count += 1
    end
    push!(lake_properties,LakeProperties(lake.lake_number,
                                         lake_cells,
                                         lake_cell_pixel_count))
  end
  sort!(lake_properties,by=x->x.lake_pixel_count,rev=true)
  all_lake_pixel_counts::Field{Int64} = Field{Int64}(surface_grid,0)
  for_all(surface_grid; use_cartesian_index=true) do coords::CartesianIndex
    cell_all_lake_pixel_count = 0
    for_all_fine_cells_in_coarse_cell(lake_grid,surface_grid,
                                      coords) do fine_coords::CartesianIndex
      if all_lake_pixel_mask(fine_coords)
        cell_all_lake_pixel_count += 1
      end
    end
    set!(all_lake_pixel_counts,coords,cell_all_lake_pixel_count)
  end

  for lake::LakeProperties in lake_properties::Vector{LakeProperties}
    for cell in lake.cell_list
      cell.total_all_lake_pixel_count =
        all_lake_pixel_counts(cell.coarse_grid_coords)
      cell.max_pixels_from_lake = cell.original_lake_pixel_count + cell.pixel_count -
                                  cell.total_all_lake_pixel_count
    end
    sort!(lake.cell_list,by=x->x.lake_pixel_count/x.pixel_count,rev=true)
    j = length(lake.cell_list)
    unprocessed_cells_total_pixel_count = 0
    for cell in lake.cell_list
      unprocessed_cells_total_pixel_count += cell.lake_pixel_count
    end
    if unprocessed_cells_total_pixel_count <
        0.5*minimum(map(cell->cell.pixel_count,lake.cell_list))
      continue
    end
    for i = 1:length(lake.cell_list)
      println("----")
      if i == j
        break
      end
      cell = lake.cell_list[i]
      unprocessed_cells_total_pixel_count -= cell.lake_pixel_count
      println(cell.lake_pixel_count)
      println(unprocessed_cells_total_pixel_count)
      if unprocessed_cells_total_pixel_count + cell.lake_pixel_count < 0.5*cell.pixel_count
        break
      end
      if cell.max_pixels_from_lake == cell.lake_pixel_count
        continue
      end
      if cell.max_pixels_from_lake < 0.5*cell.pixel_count
        continue
      end
      println(cell.lake_pixel_count)
      println(unprocessed_cells_total_pixel_count)
      while true
        least_filled_cell = lake.cell_list[j]
        if cell.lake_pixel_count + least_filled_cell.lake_pixel_count  <= cell.max_pixels_from_lake
          for pixel in values(least_filled_cell.lake_pixels)
            move_pixel(pixel,least_filled_cell,cell)
            unprocessed_cells_total_pixel_count -= 1
          end
          j -= 1
          println(unprocessed_cells_total_pixel_count)
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
          println(unprocessed_cells_total_pixel_count)
          break
        end
      end
      println(unprocessed_cells_total_pixel_count)
    end
  end
  println("----")
  pixel_counts_field::Field{Int64} =  Field{Int64}(surface_grid,0)
  for lake in lake_properties
    for cell in lake.cell_list
      set!(pixel_counts_field,cell.coarse_grid_coords,
           pixel_counts_field(cell.coarse_grid_coords)+cell.lake_pixel_count)
    end
  end
  println(pixel_counts_field)
end

function add_pixel(cell,pixel)
  if contains_lakes_in_slake_mask
    if cell.max_pixels_from_lake == cell.lake_pixel_count
      other_pixel = nothing
      other_pixel_origin_cell = nothing
      max_other_pixel_origin_cell_lake_fraction = -1
      for pixel in values(cell.lake_pixels_added)
        working_pixel_origin_cell = get_lake_cell_from_coords(lake,pixel.original_coarse_grid_coords)
        working_pixel_origin_cell_lake_fraction::Float64 =
          working_pixel_origin_cell.lake_cell_count/
          working_pixel_origin_cell.cell_count
        if working_pixel_origin_cell_lake_fraction > max_other_pixel_origin_cell_lake_fraction
          other_pixel = pixel
          other_pixel_origin_cell =  working_pixel_origin_cell
          max_other_pixel_origin_cell_lake_fraction = working_pixel_origin_cell.lake_cell_count
        end
      end
      if isnothing(other_pixel)
        error("No added pixel to return - logic error")
      end
      add_pixel(other_pixel_origin_cell,other_pixel)
      insert_pixel(cell,pixel)
    elseif in_slake_mask(cell)
      insert_pixel(cell,pixel)
    else
      most_filled_cell = nothing
      most_filled_cell_pixel_count = 0
      for other_cell in lake.cells_list
        if other_cell.lake_pixel_count > most_filled_cell_pixel_count &&
           other_cell.lake_pixel_count < other_cell.max_pixels_from_lake
          most_filled_cell = other_cell
          most_filled_cell_pixel_count = other_cell.lake_pixel_count
        end
      end
      if isnothing(most_filled_cell)
        insert_pixel(cell,pixel)
      elseif in_slake_mask(most_filled_cell)
        insert_pixel(most_filled_cell,pixel)
      else
        insert_pixel(cell,pixel)
      end
    end
  else
    insert_pixel(cell,pixel)
  end
end

function remove_pixel(cell,pixel)
  extract_pixel(cell,pixel)
  if in_slake_mask(cell)
    for other_cell in lake.cell_list
      least_filled_cell_lake_fraction = 999.0
      least_filled_cell = nothing
      other_cell_lake_fraction = other_cell.lake_pixel_count/
                                 other_cell.pixel_count
      if other_cell_lake_fraction < least_filled_cell_lake_fraction &&
         ! in_slake_mask(other_cell) && other_cell.lake_pixel_count > 0
         least_filled_cell_lake_fraction = other_cell_lake_fraction
         least_filled_cell = other_cell
      end
    end
    if ! isnothing(least_filled_cell)
      pixel = extract_any_pixel(least_filled_cell)
      insert_pixel(cell,pixel)
    end
  end
end

end
