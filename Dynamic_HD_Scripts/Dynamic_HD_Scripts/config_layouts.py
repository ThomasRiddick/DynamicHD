'''
Created on Jan 16, 2018

@author: thomasriddick
'''
from abc import ABCMeta, abstractmethod

class InputField(object):

    def __init__(self,name,conditions=[]):
        self.name = name
        self.conditions = conditions

    def __str__(self):
        if len(self.conditions) == 0:
          conditions_str = ''
        else:
          conditions_str = '\n      ; Conditions: '
        for condition in self.conditions:
          if condition is check_extension_is_nc:
            conditions_str += "Must be nc file"
          else:
            conditions_str += str(condition)
        return "{0}:{1}".format(self.name,
                                conditions_str)

class ExtendedInputField(InputField):

    def __init__(self,name,section,requires_netcdf_fieldname,conditions=[]):
        super(ExtendedInputField,self).__init__(name,conditions)
        self.section = section
        self.requires_netcdf_fieldname = requires_netcdf_fieldname

    def get_section(self):
        return self.section

    def get_requires_netcdf_fieldname(self):
        return self.requires_netcdf_fieldname

def valid_option_helper(values_and_layouts):
    return lambda value : (value in values_and_layouts.keys())

def check_extension_is_nc(value):
    return value.lower().endswith('nc')

def check_if_value_is_true_false(value):
    return (value.lower() == "true" or value.lower() == "t" or
            value.lower() == "false" or value.lower() == "f")

class CheckIfOptionalNetCDFFilepathHasFieldName(object):

    def __init__(self,filepath_sectionname,fieldname_sectionname,
                 filepath_optionname,fieldname_optionname):
        self.filepath_sectionname = filepath_sectionname
        self.fieldname_sectionname = fieldname_sectionname
        self.filepath_optionname = filepath_optionname
        self.fieldname_optionname = fieldname_optionname

    def __call__(self,config):
        if config.has_section(self.filepath_sectionname):
            if config.has_option(self.filepath_sectionname,
                                 self.filepath_optionname):
                if config.has_section(self.fieldname_sectionname):
                    if config.has_option(self.fieldname_sectionname,
                                         self.fieldname_optionname):
                        return True
                    else:
                        return False
                else:
                    return False
        return True

def add_new_fields(old_fields_and_sects,new_fields_and_sects):
    for section,fields in new_fields_and_sects.items():
        if section in old_fields_and_sects.keys():
            old_fields_and_sects[section].extend(fields)
        else:
            old_fields_and_sects[section] = fields

class Config(object):
    '''
    classdocs
    '''

    __metaclass__ = ABCMeta
    terminal_node = False
    driver_to_use = None
    required_input_fields = {}
    optional_input_fields = {}
    field_value_relationships = []

    def __init__(self):
        '''
        Constructor
        '''
    @classmethod
    def is_terminal_node(cls):
        return cls.terminal_node

    @classmethod
    def create_instance(cls):
        return cls()

    @abstractmethod
    def get_next_layout_name(self,config):
        pass

    def get_field_value_relationships(self):
        return self.field_value_relationships

    def get_required_input_fields(self):
        return self.required_input_fields

    def get_optional_input_fields(self):
        return self.optional_input_fields

    def get_driver_to_use(self):
        return self.driver_to_use

class GenericConfig(Config):

    terminal_node = False
    valid_operations_and_associated_layouts = {"cotat_plus":"CotatPlusConfig",
                                               "upscale_orography":"OrographyUpscalingConfig",
                                               "fill_sinks":"GeneralSinkFillingConfig",
                                               "create_connected_ls_mask":"CreateConnectedLSMaskConfig",
                                               "rebase_orography":
                                               "OrographyRebasingConfig",
                                               "apply_orog_correction_field":
                                               "OrographyCorrApplicationConfig",
                                               "generate_orog_correction_field":
                                               "OrographyCorrGenerationConfig",
                                               "merge_upscaled_and_corrected_orographies":
                                               "UpscaledOrographyMergerConfig",
                                               "merge_glaciers_and_orog_corrections":
                                               "GlacierAndOrographyCorrsMergerConfig",
                                               "create_orography":
                                               "OrographyCreationConfig",
                                               "determine_river_directions":
                                               "RiverDirectionDeterminationConfig",
                                               "compute_cumulative_flow":
                                               "CumulativeFlowComputationConfig",
                                               "compute_catchments":
                                               "CatchmentComputationConfig"}
    common_additional_fields_objects = [ExtendedInputField("rdirs","input_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("fine_rdirs","input_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("fine_cumulative_flow","input_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("fine_orography","input_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("fine_landsea","input_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("fine_truesinks","input_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("orography","input_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("landsea","input_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("truesinks","input_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("coarse_rdirs","output_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("coarse_orography","output_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("rdirs_out","output_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("orography_out","output_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("catchments_out","output_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("cumulative_flow_out",
                                                           "output_filepaths",True,
                                                           [check_extension_is_nc]),
                                        ExtendedInputField("landsea_out","output_filepaths",
                                                           True,[check_extension_is_nc]),
                                        ExtendedInputField("upscaling_factor","general",
                                                           False,[lambda value:
                                                                  float(value).is_integer])]

    def __init__(self):
        self.common_additional_fields = {common_additional_fields_object.name : common_additional_fields_object
                                         for common_additional_fields_object in self.common_additional_fields_objects}
        self.required_input_fields = \
        {"general":
        [InputField("operation",
                    [valid_option_helper(self.valid_operations_and_associated_layouts)])]}
        self.optional_input_fields = {}
        self.field_value_relationships = []

    def get_next_layout_name(self,config):
        return self.valid_operations_and_associated_layouts[config.get("general","operation")]

    def get_operation_name(self):
        if not self.terminal_node:
          raise UserWarning("Non terminal nodes don't have operation names")
        else:
          for key,value in self.valid_operations_and_associated_layouts.items():
            if str(type(self).__name__) == value:
              return key
          raise UserWarning("Operation name for {0} not found".format(type(self).__name__))

    def add_additional_existing_required_fields(self,field_names):
        for field_name in field_names:
            field_object = self.common_additional_fields[field_name]
            add_new_fields(self.required_input_fields,{field_object.get_section():[field_object]})
            if field_object.get_requires_netcdf_fieldname():
                add_new_fields(self.required_input_fields,
                               {field_object.get_section().replace("_filepaths",
                                                                   "_fieldnames"):
                                [InputField(field_object.name,[])]})

    def add_additional_existing_optional_fields(self,field_names):
        for field_name in field_names:
            field_object = self.common_additional_fields[field_name]
            add_new_fields(self.optional_input_fields,{field_object.get_section():[field_object]})
            if field_object.get_requires_netcdf_fieldname():
                add_new_fields(self.optional_input_fields,
                               {field_object.get_section().replace("_filepaths",
                                                                   "_fieldnames"):
                                [InputField(field_object.name,[])]})
                condition = CheckIfOptionalNetCDFFilepathHasFieldName(field_object.get_section(),
                                                                      field_object.get_section().
                                                                      replace("_filepaths",
                                                                              "_fieldnames"),
                                                                      field_object.name,
                                                                      field_object.name)
                self.field_value_relationships.append(condition)

class RiverDirUpscalingConfig(GenericConfig):

    terminal_node = False
    upscaling_algorithms_and_associated_layouts = {"modified_cotat_plus":"CotatPlusConfig"}

    def __init__(self):
        super(RiverDirUpscalingConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"river_direction_upscaling":
                        [InputField("algorithm",
                                    [valid_option_helper(self.\
                                                         upscaling_algorithms_and_associated_layouts)]),
                         InputField("parameters_filepath",[])]})
        self.add_additional_existing_required_fields(["fine_rdirs","fine_cumulative_flow",
                                                      "upscaling_factor"])

    def get_next_layout_name(self, config):
        return self.upscaling_algorithms_and_associated_layouts[config.get("river_direction_upscaling",
                                                                           "algorithm")]

class CotatPlusConfig(RiverDirUpscalingConfig):

    terminal_node = True

    def __init__(self):
        super(CotatPlusConfig,self).__init__()
        self.add_additional_existing_required_fields(["coarse_rdirs"])

class OrographyUpscalingConfig(GenericConfig):

    terminal_node = True

    def __init__(self):
        super(OrographyUpscalingConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"orography_upscaling":
                        [InputField("parameter_filepath",[])]})
        self.add_additional_existing_required_fields(["fine_orography",
                                                      "upscaling_factor",
                                                      "coarse_orography"])
        self.add_additional_existing_optional_fields(["fine_landsea","fine_truesinks"])
        self.driver_to_use = "orography_upscaling_driver"

class GeneralSinkFillingConfig(GenericConfig):

    terminal_node = False
    sink_filling_algorithms_and_associated_layouts = {"direct_sink_filling":
                                                      "DirectSinkFillingConfig",
                                                      "river_carving":
                                                      "RiverCarvingConfig"}

    def __init__(self):
        super(GeneralSinkFillingConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"sink_filling":
                        [InputField("algorithm",
                                    [valid_option_helper(self.\
                                                        sink_filling_algorithms_and_associated_layouts)])]})
        self.add_additional_existing_required_fields(["orography"])
        self.add_additional_existing_optional_fields(["landsea","truesinks"])

    def get_next_layout_name(self, config):
        return self.sink_filling_algorithms_and_associated_layouts[config.get("sink_filling",
                                                                              "algorithm")]

class DirectSinkFillingConfig(GeneralSinkFillingConfig):

    terminal_node = True

    def __init__(self):
        super(DirectSinkFillingConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"sink_filling":
                        [InputField("add_slight_slope_when_filling_sinks",
                                    [check_if_value_is_true_false]),
                         InputField("slope_param",[lambda value: float(value) >= 0.0])]})
        self.add_additional_existing_required_fields(["orography_out"])
        self.driver_to_use = "sink_filling_driver"

class RiverCarvingConfig(GeneralSinkFillingConfig):

    terminal_node = True

    def __init__(self):
        super(RiverCarvingConfig,self).__init__()
        self.add_additional_existing_required_fields(["rdirs_out"])
        self.add_additional_existing_optional_fields(["catchments_out"])

class RiverDirectionPostProcessingConfig(GenericConfig):

    terminal_node = False

    def __init__(self):
        super(RiverDirectionPostProcessingConfig,self).__init__()
        self.add_additional_existing_required_fields(["rdirs"])

class CatchmentComputationConfig(RiverDirectionPostProcessingConfig):

    terminal_node = True

    def __init__(self):
        super(CatchmentComputationConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"output_filepaths":
                        [InputField("loop_logfile",
                                    [lambda value : value.lower().endswith("txt")])]})
        self.add_additional_existing_required_fields(["catchments_out"])
        self.driver_to_use = "compute_catchment_driver"

class CumulativeFlowComputationConfig(RiverDirectionPostProcessingConfig):

    terminal_node = True

    def __init__(self):
        super(CumulativeFlowComputationConfig,self).__init__()
        self.add_additional_existing_required_fields(["cumulative_flow_out"])
        self.driver_to_use = "compute_cumulative_flow_driver"

class CreateConnectedLSMaskConfig(GenericConfig):

    terminal_node = True

    def __init__(self):
        super(CreateConnectedLSMaskConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"connected_lsmask_generation":
                        [InputField("use_diagonals",
                                    [check_if_value_is_true_false]),
                         InputField("rotate_seeds_about_polar_axis",
                                    [check_if_value_is_true_false]),
                         InputField("flip_seeds_upside_down",
                                    [check_if_value_is_true_false])]})
        add_new_fields(self.optional_input_fields,
                       {"input_filepaths":
                        [InputField("ls_seed_points",
                                    [check_extension_is_nc]),
                         InputField("ls_seed_points_list",[])]})
        self.add_additional_existing_required_fields(["landsea","landsea_out"])
        condition = lambda config: (config.has_option("input_filepaths",
                                                      "ls_seed_points") or
                                    config.has_option("input_filepaths",
                                                      "ls_seed_points_list"))
        self.field_value_relationships.append(condition)

class OrographyOperationsConfig(GenericConfig):

    terminal_node = False
    orography_operations_and_associated_layouts = \
    {"rebase_orography" : "OrographyRebasingConfig",
     "apply_orography_corrs" : "OrographyCorrApplicationConfig",
     "generate_orography_corrs" : "OrographyCorrGenerationConfig",
     "merge_in_upscaled_orography" : "UpscaledOrographyMergerConfig",
     "merge_glacier_with_orography_corrs" : "GlacierAndOrographyCorrsMergerConfig"}

    def __init__(self):
        super(OrographyOperationsConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"orography_operations":
                        [InputField("operation",
                                    [valid_option_helper(self.\
                                                         orography_operations_and_associated_layouts)])]})

class OrographyRebasingConfig(GenericConfig):

    terminal_node = True

    def __init__(self):
        super(OrographyRebasingConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"input_filepaths":
                        [InputField("present_day_base_orography",
                                    [check_extension_is_nc]),
                         InputField("present_day_reference_orography",
                                    [check_extension_is_nc])]})
        self.add_additional_existing_required_fields(["orography",
                                                      "orography_out"])

class OrographyCorrApplicationConfig(GenericConfig):

    terminal_node = True

    def __init__(self):
        super(OrographyCorrApplicationConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"input_filepaths":
                        [InputField("orography_corrections",
                                    [check_extension_is_nc])]})
        self.add_additional_existing_required_fields(["orography",
                                                      "orography_out"])

class OrographyCorrGenerationConfig(GenericConfig):

    terminal_node = True

    def __init__(self):
        super(OrographyCorrGenerationConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"input_filepaths":
                        [InputField("original_orography",
                                    [check_extension_is_nc]),
                         InputField("corrected_orography",
                                    [check_extension_is_nc])],
                        "output_filepaths":
                        [InputField("orography_corrections",
                                    [check_extension_is_nc])]})

class UpscaledOrographyMergerConfig(GenericConfig):

    terminal_node = True

    def __init__(self):
        super(UpscaledOrographyMergerConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"input_filepaths":
                        [InputField("upscaled_orography",
                                    [check_extension_is_nc]),
                         InputField("corrected_orography",
                                    [check_extension_is_nc])]})
        add_new_fields(self.optional_input_fields,
                       {"upscaled_orography_merging":
                        [InputField("use_upscaled_orography_only_in_region",[])]})
        self.add_additional_existing_required_fields(["orography_out"])

class GlacierAndOrographyCorrsMergerConfig(GenericConfig):

    terminal_node = True

    def __init__(self):
        super(GlacierAndOrographyCorrsMergerConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"input_filepaths":
                        [InputField("original_orography",
                                    [check_extension_is_nc]),
                         InputField("corrected_orography",
                                    [check_extension_is_nc]),
                         InputField("glacier_mask",
                                    [check_extension_is_nc])]})
        self.add_additional_existing_required_fields(["orography_out"])

class OrographyCreationConfig(GenericConfig):

    terminal_node = True

    def __init__(self):
        super(OrographyCreationConfig,self).__init__()
        add_new_fields(self.required_input_fields,
                       {"input_filepaths":
                        [InputField("inclines",
                                    [check_extension_is_nc])],
                        "input_fieldnames":
                        [InputField("inclines",[])]})
        self.add_additional_existing_required_fields(["landsea","orography_out"])
        self.driver_to_use = "orography_creation_driver"

class RiverDirectionDeterminationConfig(GenericConfig):

    terminal_node = True

    def __init__(self):
        super(RiverDirectionDeterminationConfig,self).__init__()
        add_new_fields(self.optional_input_fields,
                       {"river_direction_determination":
                        [InputField("always_flow_to_sea",
                                    [check_if_value_is_true_false]),
                         InputField("use_diagonal_nbrs",
                                    [check_if_value_is_true_false]),
                         InputField("mark_pits_as_true_sinks",
                                    [check_if_value_is_true_false])]})
        self.add_additional_existing_required_fields(["rdirs_out","landsea",
                                                      "orography"])
        self.add_additional_existing_optional_fields(["truesinks"])
        self.driver_to_use = "river_direction_determination_driver"
