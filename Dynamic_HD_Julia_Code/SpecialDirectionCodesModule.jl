module SpecialDirectionCodesModule

export coord_base_indicator_ocean,coord_base_indicator_outflow
export coord_base_indicator_truesink,coord_base_indicator_lake
export dir_based_indicator_ocean,dir_based_indicator_outflow
export dir_based_indicator_truesink,dir_based_indicator_lake

const coord_base_indicator_ocean    = -1
const coord_base_indicator_outflow  =  0
const coord_base_indicator_truesink = -5
const coord_base_indicator_lake     = -2

const dir_based_indicator_ocean =   -1
const dir_based_indicator_outflow =  0
# const dir_based_indicator_truesink = 5
# const dir_based_indicator_lake =    -2
const dir_based_indicator_truesink = -2
const dir_based_indicator_lake =      5

end
