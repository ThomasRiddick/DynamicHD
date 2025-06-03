module TransferTrueSinksToIconGrid

function transfer_true_sinks_from_latlon_to_icon_grid(latlon_true_sinks_field,
                                                      latlon_grid_clats,
                                                      latlon_grid_clons,
                                                      icon_cell_indices,
                                                      icon_grid_clats,
                                                      icon_grid_clons)
  search_box_width::Float64 = deg2rad(0.5)
  true_sinks_list::Vector{CartesianIndex} = findall(latlon_true_sinks_field)
  icon_true_sinks_field::Vector{Bool} = zeros(Bool,size(icon_cell_indices))
  high_latitude_limit::Float64 = deg2rad(89)
  for indices_latlon::CartesianIndex in true_sinks_list
    latlon_grid_clat = deg2rad(latlon_grid_clats[indices_latlon[1]])
    latlon_grid_clon = deg2rad(latlon_grid_clons[indices_latlon[2]])
    if latlon_grid_clat > high_latitude_limit || latlon_grid_clat < -high_latitude_limit
      continue
    end
    indices_for_icon_cells_in_search_box =
      filter(x->((abs(icon_grid_clats[x] - latlon_grid_clat) < search_box_width) &&
                 (abs(icon_grid_clons[x] - latlon_grid_clon) < search_box_width)),
                 icon_cell_indices)
    min_central_angle_over_two_squared = Inf
    closest_icon_cell_center_index::Int64 = -1
    for index_icon in indices_for_icon_cells_in_search_box
      delta_lat = abs(icon_grid_clats[index_icon] - latlon_grid_clat)
      delta_lon = abs(icon_grid_clons[index_icon] - latlon_grid_clon)
      average_lat = (icon_grid_clats[index_icon] + latlon_grid_clat)/2.0
      central_angle_over_two_squared  = (sin(delta_lon/2)*cos(average_lat))^2 +
                                        (cos(delta_lon/2)*sin(delta_lat/2))^2
      if central_angle_over_two_squared  < min_central_angle_over_two_squared
        min_central_angle_over_two_squared  = central_angle_over_two_squared
        closest_icon_cell_center_index = index_icon
      end
    end
    icon_true_sinks_field[closest_icon_cell_center_index] = true
  end
  return icon_true_sinks_field
end

end
