'''
Contains classes related to automatically matching river mouths in a dataset
under evaluation against those in a reference dataset.

Created on Apr 22, 2016

@author: thomasriddick
'''

import numpy as np
import copy
import sys
import re
from ../Dynamic_HD_Scripts import dynamic_hd

class Params(object):

    def __init__(self, paramset='default'):
        {'default':self.init_default_params,
         'testing':self.init_testing_params,
         'area':self.init_area_params,
         'extensive':self.init_extensive_params,
         'area_extensive':self.init_area_extensive_params,
         'magnitude_extensive':self.init_magnitdue_extensive}[paramset]()

    def init_default_params(self):
        self.max_complexity = 15
        self.missing_pair_penalty_factor = 2.0
        self.missing_pair_penalty_constant = 2.0
        self.ps_a = 0.0
        self.ps_b = 0.0
        self.ps_c = 1.0
        self.minflow = 200
        self.range = 10
        self.magnitude_tolerance_factor = 5

    def init_extensive_params(self):
        self.init_default_params()
        self.minflow = 135

    def init_testing_params(self):
        self.max_complexity = 15
        self.missing_pair_penalty_factor = 10.0
        self.missing_pair_penalty_constant = 5.0
        self.ps_a = 0.001
        self.ps_b = 0.2
        self.ps_c = 50.0
        self.minflow = 1000
        self.range = 10
        self.magnitude_tolerance_factor = 5

    def init_area_params(self):
        self.max_complexity = 15
        self.missing_pair_penalty_factor = 2.0
        self.missing_pair_penalty_constant = 2.0
        self.ps_a = 1.0
        self.ps_b = 0.0
        self.ps_c = 0.0
        self.minflow = 200
        self.range = 5
        self.magnitude_tolerance_factor = 5

    def init_area_extensive_params(self):
        self.init_area_params(),
        self.range = 10
        self.minflow= 80

    def init_magnitdue_extensive(self):
        self.init_default_params()
        self.minflow = 100

class RiverMouth(object):

    def __init__(self,lat,lon,outflow,idnum,params):
        self.lat = lat
        self.lon = lon
        self.outflow = outflow
        self.idnum = idnum
        self.params = params
        self.rangesqrd = self.params.range**2

    def __str__(self):
        return "lat: {0}, lon: {1}, outflow: {2}, idnum: {3} ".format(self.lat,
                                                                     self.lon,
                                                                     self.outflow,
                                                                     self.idnum)

    __repr__ = __str__

    def __eq__(self,rhs):
        return self.lat==rhs.lat and self.lon == rhs.lon and \
                self.outflow == rhs.outflow and self.idnum == rhs.idnum

    def get_coords(self):
        return (self.lat,self.lon)

    def get_lat(self):
        return self.lat

    def get_lon(self):
        return self.lon

    def get_outflow(self):
        return self.outflow

    def get_idnum(self):
        return self.idnum

    def square_of_range_difference_from(self,x,y):
        return (self.lat -x)**2 + (self.lon - y)**2

    def fraction_magnitude_difference_from(self,value):
        return 2.0*abs(value - self.outflow)/(value + self.outflow)

    def is_within_range_of(self,x,y):
        if self.square_of_range_difference_from(x,y) <= self.rangesqrd:
                return True
        else:
                return False

    def has_similar_magnitude_to(self,value):
        if (self.outflow/self.params.magnitude_tolerance_factor <= value) and \
            (value < self.outflow*self.params.magnitude_tolerance_factor):
            return True
        else:
            return False

class ConflictChecker(object):

    @classmethod
    def check_pair_sets_for_conflicts(cls,candidatepairs):
        conflicts,conflict_free_pairs = cls.check_pair_set_for_conflicts(candidatepairs,0)
        additional_conflicts,conflict_free_pairs = \
            cls.check_pair_set_for_conflicts(conflict_free_pairs,1)
        conflicts.extend(additional_conflicts)
        conflicts = cls.associate_conflicts(conflicts, 0)
        conflicts = cls.associate_conflicts(conflicts, 1)
        return conflicts, conflict_free_pairs

    @classmethod
    def check_pair_set_for_conflicts(cls,candidatepairs,pairindex):
        conflicts_list = []
        no_conflicts_list = []
        matchedcandidatesids  = np.array([cdt[pairindex].get_idnum() for cdt in candidatepairs])
        uniqueids, uniqueidcounts = np.unique(matchedcandidatesids,
                                              return_counts=True)
        uniqueidcounts = np.ma.masked_less_equal(uniqueidcounts, 1, copy=False)
        uniqueids = np.ma.array(uniqueids,copy=False,mask=np.ma.getmaskarray(uniqueidcounts),
                                keep_mask=False)
        for idnum in np.ma.compressed(uniqueids):
            conflict = []
            for candidatepair, candidateid in zip(candidatepairs,matchedcandidatesids):
                if candidateid == idnum:
                    conflict.append(candidatepair)
            conflicts_list.append(conflict)
        #It is not clear if masked_equal clears the previous mask so do it
        #manually
        uniqueidcounts.mask = np.ma.nomask
        uniqueidcounts = np.ma.masked_not_equal(uniqueidcounts,1,copy=False)
        uniqueids = np.ma.array(uniqueids,copy=False,mask=np.ma.getmaskarray(uniqueidcounts),
                                keep_mask=False)
        for idnum in np.ma.compressed(uniqueids):
            for candidatepair, candidateid in zip(candidatepairs,matchedcandidatesids):
                if candidateid == idnum:
                    no_conflicts_list.append(candidatepair)
        return conflicts_list,no_conflicts_list

    @staticmethod
    def associate_conflicts(conflicts_list,pairindex):
        conflicts_list = [{'item':entry,'skip':False} for entry in conflicts_list]
        invertedpairindex = (1,0)[pairindex]
        for conflict in conflicts_list:
            if conflict['skip']:
                continue
            repeat = True
            while repeat:
                repeat = False
                for furtherconflict in conflicts_list:
                    if furtherconflict['skip'] or (conflict is furtherconflict):
                        continue
                    for pair in conflict['item']:
                        if furtherconflict['skip']:
                            continue
                        for furtherpair in furtherconflict['item']:
                            if pair[invertedpairindex].get_idnum() == \
                                furtherpair[invertedpairindex].get_idnum() and \
                                pair is not furtherpair:
                                    conflict['item'].extend(furtherconflict['item'])
                                    furtherconflict['skip'] = True
                                    repeat = True
                                    break
        return [item['item'] for item in conflicts_list if not item['skip']]

class ConflictResolver(object):
    """Contains methods necessary to resolve a set of conflicts"""

    @classmethod
    def resolve_conflicts(cls,conflicts,params):
        """Resolve a list of conflicts"""
        pairs_from_resolved_conflicts = []
        pairs_from_unresolved_conflicts = []
        for conflict in conflicts:
            if len(conflict) > params.max_complexity:
                pairs_from_unresolved_conflicts.append(conflict)
            else:
                best_config = cls.resolve_conflict(conflict,params)
                best_config = [x for x in best_config if x is not None]
                pairs_from_resolved_conflicts.extend(best_config)
        return pairs_from_resolved_conflicts,pairs_from_unresolved_conflicts

    @classmethod
    def resolve_conflict(cls,conflict,params):
        """Resolve an individual conflict"""
        allowed_configurations = cls.generate_possible_inconflict_pairings(conflict)
        best_config_score = sys.float_info.max
        for allowed_configuration in allowed_configurations:
            if all(pair is None for pair in allowed_configuration):
                continue
            config_score = cls.evaulate_configuration(allowed_configuration,params)
            if config_score < best_config_score:
                best_config_score = config_score
                best_config = allowed_configuration
        return best_config

    @staticmethod
    def evaulate_configuration(allowed_configuration,params):
        """Evaluate a configurations of pair and give it a likelihood score"""
        missing_pair_count = 0
        total_score = 0
        for pair in allowed_configuration:
            if pair is None:
                missing_pair_count += 1
                continue
            pair_score = params.ps_a*pair[0].square_of_range_difference_from(*pair[1].get_coords()) + \
                         params.ps_b*pair[0].square_of_range_difference_from(*pair[1].get_coords())* \
                         pair[0].fraction_magnitude_difference_from(pair[1].get_outflow()) + \
                         params.ps_c*pair[0].fraction_magnitude_difference_from(pair[1].get_outflow())
            total_score += pair_score
        total_score *= 1.0*(len(allowed_configuration) - missing_pair_count)/\
                        len(allowed_configuration)
        if missing_pair_count > 0:
            total_score *= missing_pair_count*params.missing_pair_penalty_factor
            total_score += missing_pair_count*params.missing_pair_penalty_constant
        return total_score

    @classmethod
    def generate_possible_inconflict_pairings(cls,conflict):
        """Generate all possible pairings within a conflicts"""
        referencemouthidnums =  [pair[0].get_idnum() for pair in conflict]
        #Easier to test if we perserve order rather than just using a set
        seen = set()
        referencemouthidnums = [value for value in referencemouthidnums
                                    if not (value in seen or seen.add(value))]
        possiblepairings_for_all_idnums = []
        for referencemouthidnum in referencemouthidnums:
            possiblepairings_for_this_idnum= [pair for pair in conflict if
                                                 pair[0].get_idnum() == referencemouthidnum]
            possiblepairings_for_all_idnums.append(possiblepairings_for_this_idnum)
        allowed_configurations = []
        return cls.add_to_allowed_configurations(allowed_configurations, 0,
                                                 possiblepairings_for_all_idnums)

    @classmethod
    def add_to_allowed_configurations(cls,allowed_configurations,refidnumindex,possible_pairings_for_all_idnums):
        """Recursively expand a set of allowed configuration"""
        if refidnumindex >= len(possible_pairings_for_all_idnums):
            return allowed_configurations
        if len(allowed_configurations) <= 0:
            for pairing in possible_pairings_for_all_idnums[refidnumindex]:
                allowed_configurations.append([pairing])
        else:
            newconfigurations = []
            for configuration in allowed_configurations:
                for pairing in possible_pairings_for_all_idnums[refidnumindex]:
                    if cls.is_pairing_allowed_in_configuration(pairing,configuration):
                        newconfiguration = copy.deepcopy(configuration)
                        newconfiguration.append(pairing)
                        newconfigurations.append(newconfiguration)
                configuration.append(None)
            allowed_configurations.extend(newconfigurations)
        allowed_configurations = cls.add_to_allowed_configurations(allowed_configurations, refidnumindex+1,
                                                                   possible_pairings_for_all_idnums)
        return allowed_configurations

    @staticmethod
    def is_pairing_allowed_in_configuration(pairing,configuration):
        """Check if this pairing is allowed by checking if the pair has been already used"""
        if pairing[1].get_idnum() in [(pair[1].get_idnum() if pair is not None else None)
                                      for pair in configuration]:
            return False
        else:
            return True

def generate_candidate_pairs(referencelist,rmouthlist):
    candidatepairs = []
    for river_mouth in rmouthlist:
        for reference_rmouth in referencelist:
            if (reference_rmouth.is_within_range_of(*river_mouth.get_coords())) and \
                (reference_rmouth.has_similar_magnitude_to(river_mouth.get_outflow())):
                candidatepairs.append((reference_rmouth,river_mouth))
    return candidatepairs

def find_rivermouth_points(field,params):
    field[field < params.minflow] = 0
    xcoords, ycoords = np.nonzero(field)
    list_of_rmouths = []
    for idnum,(x,y) in enumerate(zip(xcoords,ycoords)):
        list_of_rmouths.append(RiverMouth(x,y,field[x][y],idnum,params))
    return list_of_rmouths

def generate_matches(reference_field,data_field,params):
    """High level river mouth matching method. Call low level routines"""
    referencelist = find_rivermouth_points(reference_field,params)
    rmouthlist  = find_rivermouth_points(data_field,params)
    candidatepairs = generate_candidate_pairs(referencelist, rmouthlist)
    conflicts, conflict_free_pairs = \
        ConflictChecker.check_pair_sets_for_conflicts(candidatepairs)
    pairs_from_resolved_conflicts,pairs_from_unresolved_conflicts =\
        ConflictResolver.resolve_conflicts(conflicts,params)
    conflict_free_pairs.extend(pairs_from_resolved_conflicts)
    return conflict_free_pairs, pairs_from_unresolved_conflicts

def load_additional_manual_matches(additional_matches_filename,reference_rmouth_outflows_filename,
                                   data_rmouth_outflows_filename,flip_data_field=False,
                                   rotate_data_field=False,grid_type='HD',**grid_kwargs):
    """Add any additional matches made by hand using details list in a text file"""
    reference_field = dynamic_hd.load_field(reference_rmouth_outflows_filename,
                                            file_type = dynamic_hd.\
                                                get_file_extension(reference_rmouth_outflows_filename),
                                            field_type='Generic',
                                            grid_type=grid_type,**grid_kwargs)
    data_field = dynamic_hd.load_field(data_rmouth_outflows_filename,
                                            file_type = dynamic_hd.\
                                                get_file_extension(data_rmouth_outflows_filename),
                                            field_type='Generic',
                                            grid_type=grid_type,**grid_kwargs)
    if flip_data_field:
        data_field.flip_data_ud()
    if rotate_data_field:
        data_field.rotate_field_by_a_hundred_and_eighty_degrees()
    first_line_pattern = re.compile(r"^ref_lat *, *ref_lon *, *data_lat *, *data_lon$")
    comment_line_pattern = re.compile(r"^ *#.*$")
    additional_matches = []
    with open(additional_matches_filename) as f:
        if not first_line_pattern.match(f.readline().strip()):
            raise RuntimeError("List of corrections being loaded has incorrect format the first line")
        for line in f:
            if comment_line_pattern.match(line):
                continue
            ref_lat,ref_lon,data_lat,data_lon = [int(coord) for coord in line.strip().split(",")]
            params = Params('default')
            ref_mouth = RiverMouth(ref_lat,ref_lon,reference_field.get_data()[ref_lat,ref_lon],0,params)
            data_mouth = RiverMouth(data_lat,data_lon,data_field.get_data()[data_lat,data_lon],0,params)
            additional_matches.append((ref_mouth,data_mouth))
    return additional_matches

def load_additional_manual_truesink_matches(additional_matches_filename,reference_rmouth_outflows_filename,
                                            data_rmouth_outflows_filename,reference_flowmap_field_filename,
                                            data_flowmap_field_filename,flip_data_rmouth_outflow_field=False,
                                            rotate_data_rmouth_outflow_field=False,
                                            flip_data_flowmap_field=False,rotate_data_flowmap_field=False,
                                            grid_type="HD",**grid_kwargs):
    """Any any additional matches involving true sinks by hand using details in list in a text file

    Includes both matches between a river and truesink, a truesink and a river and a true sink and a true sink.
    True sinks are ready from the cumulative flow to cell field and are not verified.
    """

    reference_rmouth_field = dynamic_hd.load_field(reference_rmouth_outflows_filename,
                                                   file_type = dynamic_hd.\
                                                   get_file_extension(reference_rmouth_outflows_filename),
                                                   field_type='Generic',
                                                   grid_type=grid_type,**grid_kwargs)
    data_rmouth_field = dynamic_hd.load_field(data_rmouth_outflows_filename,
                                              file_type = dynamic_hd.\
                                              get_file_extension(data_rmouth_outflows_filename),
                                              field_type='Generic',
                                              grid_type=grid_type,**grid_kwargs)
    reference_flowmap_field = dynamic_hd.load_field(reference_flowmap_field_filename,
                                                    file_type = dynamic_hd.\
                                                    get_file_extension(reference_flowmap_field_filename),
                                                    field_type='Generic',
                                                    grid_type=grid_type,**grid_kwargs)
    data_flowmap_field = dynamic_hd.load_field(data_flowmap_field_filename,
                                               file_type = dynamic_hd.\
                                               get_file_extension(data_flowmap_field_filename),
                                               field_type='Generic',
                                               grid_type=grid_type,**grid_kwargs)
    if flip_data_rmouth_outflow_field:
        data_rmouth_field.flip_data_ud()
    if rotate_data_rmouth_outflow_field:
        data_rmouth_field.rotate_field_by_a_hundred_and_eighty_degrees()
    if flip_data_flowmap_field:
        data_flowmap_field.flip_data_ud()
    if rotate_data_flowmap_field:
        data_flowmap_field.rotate_field_by_a_hundred_and_eighty_degrees()
    first_line_pattern = re.compile(r"^ref_lat *, *ref_lon *, *ref_type *, *data_lat *, *data_lon *, *data_type *$")
    comment_line_pattern = re.compile(r"^ *#.*$")
    additional_matches = []
    with open(additional_matches_filename) as f:
        if not first_line_pattern.match(f.readline().strip()):
            raise RuntimeError("List of corrections being loaded has incorrect format the first line")
        for line in f:
            if comment_line_pattern.match(line):
                continue
            ref_lat,ref_lon,ref_type,data_lat,data_lon,data_type = line.strip().split(",")
            ref_lat,ref_lon,data_lat,data_lon=[int(coords) for coords in [ref_lat,ref_lon,data_lat,data_lon]]
            params = Params('default')
            outflow_types={"RM":{"ref":reference_rmouth_field,"data":data_rmouth_field},
                           "TS":{"ref":reference_flowmap_field,"data":data_flowmap_field}}
            ref_mouth = RiverMouth(ref_lat,ref_lon,
                                   outflow_types[ref_type]["ref"].get_data()[ref_lat,ref_lon],0,params)
            data_mouth = RiverMouth(data_lat,data_lon,
                                    outflow_types[data_type]["data"].get_data()[data_lat,data_lon],0,params)
            additional_matches.append((ref_mouth,data_mouth))
    return additional_matches

def main(reference_rmouth_outflows_filename, data_rmouth_outflows_filename,
         grid_type='HD',param_set='default',flip_data_field=False,
         rotate_data_field=False,flip_ref_field=False,rotate_ref_field=False,**grid_kwargs):
    """Top level river mouth matching routine. Deals with file handling"""
    reference_field = dynamic_hd.load_field(reference_rmouth_outflows_filename,
                                            file_type = dynamic_hd.\
                                                get_file_extension(reference_rmouth_outflows_filename),
                                            field_type='Generic',
                                            grid_type=grid_type,**grid_kwargs)
    data_field = dynamic_hd.load_field(data_rmouth_outflows_filename,
                                            file_type = dynamic_hd.\
                                                get_file_extension(data_rmouth_outflows_filename),
                                            field_type='Generic',
                                            grid_type=grid_type,**grid_kwargs)
    if flip_data_field:
        data_field.flip_data_ud()
    if rotate_data_field:
        data_field.rotate_field_by_a_hundred_and_eighty_degrees()
    if flip_ref_field:
        reference_field.flip_data_ud()
    if rotate_ref_field:
        reference_field.rotate_field_by_a_hundred_and_eighty_degrees()
    params = Params(param_set)
    return generate_matches(reference_field.get_data(), data_field.get_data(),params)
