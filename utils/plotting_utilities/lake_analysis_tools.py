import numpy as np
import scipy.ndimage as ndimage
from collections import deque
from copy import deepcopy
import matplotlib.pyplot as plt
import sys
import warnings

warnings.warn("What does the input area bounds variable do???? Ditto dates")

#Not wrapping - no known paleo-lakes that
#would require this
def get_neighbors(coords):
    nbr_coords = []
    if len(coords) == 2:
        (i,j) = coords
        for k in range(-1,2):
            for l in range(-1,2):
                if k == 0 and l == 0:
                    continue
                nbr_coords.append((k+i,j+l))
    return nbr_coords

class LakeTracker:

    def __init__(self,initial_lake_mask,initial_lake_center):
        initial_lake_center = tuple(initial_lake_center)
        initial_lake_numbers = self.number_lakes(initial_lake_mask)
        initial_lake_number = initial_lake_numbers[initial_lake_center]
        self.lake = (initial_lake_numbers == initial_lake_number)

    def get_current_lake_mask(self):
        return self.lake

    @staticmethod
    def number_lakes(lake_mask):
        lake_cell_coords_list = np.argwhere(lake_mask)
        completed_cells = np.zeros(lake_mask.shape,
                                       dtype=np.bool_)
        lake_numbers = np.zeros(lake_mask.shape,dtype=np.int64)
        q = deque()
        lake_number = 1
        for lake_cell_coords in lake_cell_coords_list:
            if not completed_cells[tuple(lake_cell_coords)]:
                q.appendleft(tuple(lake_cell_coords))
                completed_cells[tuple(lake_cell_coords)] = True
                while len(q) > 0:
                    working_coords = q.pop()
                    lake_numbers[working_coords] = lake_number
                    for nbr_coords in get_neighbors(working_coords):
                        if (lake_mask[nbr_coords] and
                            not (completed_cells[nbr_coords])):
                            completed_cells[nbr_coords] = True
                            q.appendleft(nbr_coords)
                lake_number += 1
        return lake_numbers


    def track_lake(self,lake_mask):
        new_lake_numbers = self.number_lakes(lake_mask)
        intersecting_lakes = new_lake_numbers[self.lake]
        (intersecting_lake_numbers,
         intersecting_lake_cell_counts) = np.unique(intersecting_lakes,return_counts=True)
        zero_mask = (intersecting_lake_numbers == 0)
        intersecting_lake_numbers = np.delete(intersecting_lake_numbers,zero_mask)
        intersecting_lake_cell_counts = np.delete(intersecting_lake_cell_counts,zero_mask)
        max_index = np.argmax(intersecting_lake_cell_counts)
        new_lake_number = intersecting_lake_numbers[max_index]
        self.lake = (new_lake_number == new_lake_numbers)
        return self.lake

class LakePointExtractor:

    def __init__(self):
        self.lake_tracker = None

    def extract_lake_point_sequence(self,
                                   initial_lake_center,
                                   dates,
                                   input_area_bounds,
                                   lake_basin_numbers_sequence):
        self.lake_tracker = None
        self.initial_lake_center = initial_lake_center
        lake_point_sequence = []
        for date,lake_basin_numbers in \
              zip(dates,
                  lake_basin_numbers_sequence):
              lake_point_sequence.append(self.extract_lake(lake_basin_numbers))
        return lake_point_sequence

    def extract_lake(self,lake_basin_numbers):
        all_lake_mask = np.logical_not(lake_basin_numbers <= 0)
        if self.lake_tracker is None:
           self.lake_tracker = LakeTracker(all_lake_mask,self.initial_lake_center)
           this_lake_mask = self.lake_tracker.get_current_lake_mask()
        else:
           this_lake_mask = self.lake_tracker.track_lake(all_lake_mask)
        return tuple(np.argwhere(this_lake_mask)[0])

class LakeHeightAndVolumeExtractor:

    def extract_lake_height_and_volume_sequence(self,
                                                lake_point_sequence,
                                                filled_orography_sequence,
                                                lake_volumes_sequence):
        lake_heights = []
        lake_volumes = []
        for lake_point,filled_orography,lake_volumes_array in \
              zip(lake_point_sequence,
                  filled_orography_sequence,
                  lake_volumes_sequence):
              lake_heights.append(filled_orography[tuple(lake_point)])
              lake_volumes.append(lake_volumes_array[tuple(lake_point)])
        return lake_heights,lake_volumes

class CoastlineIdentifier:

    def __init__(self,first_transect_endpoints,
                 second_transect_endpoints,
                 shape):
        self.shape = shape
        self.first_transect_endpoints = first_transect_endpoints
        self.second_transect_endpoints = second_transect_endpoints
        first_transect_mask = self.generate_transect(first_transect_endpoints)
        second_transect_mask = self.generate_transect(second_transect_endpoints)
        self.limiting_transect_mask = np.logical_or(first_transect_mask,
                                                    second_transect_mask)
        self.midpoint_transect_mask = \
            self.generate_transect(self.get_midpoints(first_transect_endpoints,
                                                      second_transect_endpoints))

    #See above comment on not wrapping
    @staticmethod
    def get_midpoints(first_transect_endpoints,
                      second_transect_endpoints):
        return [[(first_transect_endpoints[0][0] + second_transect_endpoints[0][0])//2,
                 (first_transect_endpoints[0][1] + second_transect_endpoints[0][1])//2],
                [(first_transect_endpoints[1][0] + second_transect_endpoints[1][0])//2,
                 (first_transect_endpoints[1][1] + second_transect_endpoints[1][1])//2]]

    #This is fairly ineffecient - a distance from
    #end pointed sorted priority queue might be better
    def generate_transect(self,endpoints):
        starting_point_coords = tuple(endpoints[0])
        end_point_coords = tuple(endpoints[1])
        cells_in_transect = np.zeros(self.shape,
                                   dtype=np.bool_)
        working_coords = starting_point_coords
        while True:
            if working_coords == end_point_coords:
                break
            closest_nbr_coords = None
            working_distance = sys.float_info.max
            for nbr_coords in get_neighbors(working_coords):
                nbr_to_endpoint_distance = pow((end_point_coords[0] - nbr_coords[0])**2 +
                                               (end_point_coords[1] - nbr_coords[1])**2,0.5)
                if (closest_nbr_coords is None or
                    nbr_to_endpoint_distance < working_distance):
                    closest_nbr_coords = nbr_coords
                    working_distance = nbr_to_endpoint_distance
            cells_in_transect[closest_nbr_coords] = True
            working_coords = closest_nbr_coords
        cells_in_transect = \
            ndimage.binary_dilation(cells_in_transect,
                                    structure=
                                    ndimage.generate_binary_structure(2,2))
        return cells_in_transect

    def identify_coastline(self,landsea_mask):
        midpoint_transect_ocean_cell_coords = \
            np.argwhere(np.logical_and(self.midpoint_transect_mask,landsea_mask))
        starting_point_coords = None
        #Islands???
        for cell_coords in midpoint_transect_ocean_cell_coords:
            for nbr_coords in get_neighbors(cell_coords):
                if not landsea_mask[nbr_coords]:
                    starting_point_coords = nbr_coords;
                    break
        completed_cells = np.zeros(self.shape,
                                   dtype=np.bool_)
        coastal_land_cells = np.zeros(self.shape,
                                      dtype=np.bool_)
        coastal_ocean_cells = np.zeros(self.shape,
                                      dtype=np.bool_)
        q = deque()
        q.appendleft(starting_point_coords)
        completed_cells[starting_point_coords] = True
        while len(q) > 0:
            working_coords = q.pop()
            for nbr_coords in get_neighbors(working_coords):
                if not (completed_cells[nbr_coords] or
                        landsea_mask[nbr_coords] or
                        self.limiting_transect_mask[nbr_coords]):
                    completed_cells[nbr_coords] = True
                    next_to_ocean = False
                    for secondary_nbr_coords in \
                        get_neighbors(nbr_coords):
                        if landsea_mask[secondary_nbr_coords]:
                            next_to_ocean = True
                    if next_to_ocean:
                        coastal_land_cells[nbr_coords] = True
                        q.appendleft(nbr_coords)
                        for secondary_nbr_coords in \
                            get_neighbors(nbr_coords):
                            if landsea_mask[secondary_nbr_coords]:
                                coastal_ocean_cells[secondary_nbr_coords] = True
        return coastal_land_cells,coastal_ocean_cells

class OutflowBasinIdentifier:

    ocean_basins_30min_latlon = {"Artic":[[[30,88],[60,88]],[[12,190],[45,190]]],
                                 "Carib":[[[133,167],[122,167]],[[112,221],[106,202]]],
                                 "St Lawrence":[[[112,221],[106,202]],[[80,262],[80,230]]]}
    ocean_basins_definitions = {"30minLatLong":ocean_basins_30min_latlon}
    field_shapes = {"30minLatLong":(360,720)}

    def __init__(self,grid_type):
        ocean_basins_definitions = self.ocean_basins_definitions[grid_type]
        field_shape = self.field_shapes[grid_type]
        self.ocean_basin_numbers = -1*np.ones(field_shape,dtype=np.int32)
        self.ocean_basin_names = []
        self.coastline_identifiers = []
        for key,value in ocean_basins_definitions.items():
            self.coastline_identifiers.append(CoastlineIdentifier(*value,field_shape))
            self.ocean_basin_names.append(key)

    def set_lsmask(self,lsmask):
        for i,coastline_identifier in enumerate(self.coastline_identifiers):
            _,coastal_ocean_cells = coastline_identifier.identify_coastline(lsmask)
            self.ocean_basin_numbers[coastal_ocean_cells] = i

    def identify_ocean_basin_for_lake_outflow(self,
                                              lsmask,
                                              connected_catchments,
                                              lake_point,
                                              input_area_bounds):
        self.set_lsmask(lsmask)
        catchment = connected_catchments[tuple(lake_point)]
        ocean_basin_number = self.ocean_basin_numbers[np.logical_and(connected_catchments == catchment,
                                                                     self.ocean_basin_numbers >= 0)].flatten()[0]
        return self.ocean_basin_names[ocean_basin_number]

    def extract_ocean_basin_for_lake_outflow_sequence(self,
                                                      dates,
                                                      input_area_bounds,
                                                      lsmask_sequence,
                                                      lake_point_sequence,
                                                      connected_catchments_sequence):
        lake_outflow_basins = []
        for date,lsmask,lake_point,connected_catchments in \
              zip(dates,lsmask_sequence,lake_point_sequence,
                  connected_catchments_sequence):
              basin_name = self.identify_ocean_basin_for_lake_outflow(lsmask,
                                                                      connected_catchments,
                                                                      lake_point,
                                                                      input_area_bounds)
              lake_outflow_basins.append(basin_name)
        return lake_outflow_basins

    def calculate_discharge_to_ocean_basins(self,
                                            lsmask,
                                            discharge_to_ocean,
                                            input_area_bounds):
        self.set_lsmask(lsmask)
        discharge_to_ocean_basins = []
        for ocean_basin_number in range(len(self.coastline_identifiers)):
            discharge_to_ocean_basins.append(sum(discharge_to_ocean[self.ocean_basin_numbers ==
                                                                         ocean_basin_number]))
        return discharge_to_ocean_basins




    def calculate_discharge_to_ocean_basins_sequence(self,
                                                     dates,
                                                     lsmask_sequence,
                                                     discharge_to_ocean_sequence):
        discharge_to_ocean_basin_timeseries = []
        for date,lsmask,discharge_to_ocean in \
              zip(dates,lsmask_sequence,discharge_to_ocean_sequence):
              discharge_to_ocean_basins = self.calculate_discharge_to_ocean_basins(lsmask,
                                                                                   discharge_to_ocean,
                                                                                   input_area_bounds)
              discharge_to_ocean_basin_timeseries.append(discharge_to_ocean_basins)
        return discharge_to_ocean_basin_timeseries

class SpillwayProfiler:

    #Again no wrapping as no known case where wrapping would be required
    @staticmethod
    def find_downstream_cell(working_coords,sinkless_rdirs):
        rdir = sinkless_rdirs[tuple(working_coords)]
        inc_i = 0
        inc_j = 0
        if rdir == 5 or rdir == 0:
            return False
        elif rdir == -1:
            raise RuntimeError("River reaching unphysical point")
        elif rdir < -1 or rdir > 9:
            raise RuntimeError("Unrecognised river direction: {}".format(rdir))
        if rdir == 7 or rdir == 8 or rdir == 9:
            inc_i = -1
        elif rdir == 1 or rdir == 2 or rdir == 3:
            inc_i = 1
        if rdir == 7 or rdir == 4 or rdir == 1:
            inc_j = -1
        elif rdir == 9 or rdir == 6 or rdir == 3:
            inc_j =  1
        working_coords[0] = working_coords[0] + inc_i
        working_coords[1] = working_coords[1] + inc_j
        return working_coords

    @classmethod
    def extract_spillway_profile(cls,lake_center,sinkless_rdirs,
                                 corrected_heights):
        spillway_height_profile = [corrected_heights[tuple(lake_center)]]
        working_coords = list(deepcopy(lake_center))
        while cls.find_downstream_cell(working_coords,sinkless_rdirs):
            spillway_height_profile.append(corrected_heights[tuple(working_coords)])
        return spillway_height_profile

if __name__ == '__main__':
    from Dynamic_HD_Scripts.base.iodriver import advanced_field_loader
    rdirs = advanced_field_loader(filename="/Users/thomasriddick/Documents/data/lake_analysis_runs/lake_analysis_two_26_Mar_2022/lakes/results/diag_version_35_date_14900/30min_rdirs.nc",
                                  time_slice=None,
                                  field_type="RiverDirections",
                                  fieldname="rdirs",
                                  adjust_orientation=True)
    connected_catchments = advanced_field_loader(filename="/Users/thomasriddick/Documents/data/lake_analysis_runs/lake_analysis_two_26_Mar_2022/lakes/results/diag_version_35_date_14900/30min_connected_catchments.nc",
                                                 time_slice=None,
                                                 field_type="Generic",
                                                 fieldname="catchments",
                                                 adjust_orientation=True)
    lsmask_data = rdirs.get_lsmask()
    shape = lsmask_data.shape
    lake_center = [266//3,521//3]
    basin_identifier = OutflowBasinIdentifier("30minLatLong")
    print(basin_identifier.identify_ocean_basin_for_lake_outflow(lsmask=lsmask_data,
                                                                 connected_catchments=
                                                                 connected_catchments.get_data(),
                                                                 lake_point=lake_center,
                                                                 input_area_bounds=[]))
