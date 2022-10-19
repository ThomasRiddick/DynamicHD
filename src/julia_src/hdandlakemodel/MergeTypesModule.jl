module MergeTypesModule

using CoordsModule: Coords
using GridModule: Grid,for_all
import Base.zero

export MergeTypes
export SimpleMergeTypes
export no_merge_mtype
export connection_merge_as_primary_flood_merge_as_primary
export connection_merge_as_primary_flood_merge_as_secondary
export connection_merge_as_primary_flood_merge_not_set
export connection_merge_as_primary_flood_merge_as_both
export connection_merge_as_secondary_flood_merge_as_primary
export connection_merge_as_secondary_flood_merge_as_secondary
export connection_merge_as_secondary_flood_merge_not_set
export connection_merge_as_secondary_flood_merge_as_both
export connection_merge_not_set_flood_merge_as_primary
export connection_merge_not_set_flood_merge_as_secondary
export connection_merge_not_set_flood_merge_as_both
export connection_merge_as_both_flood_merge_as_primary
export connection_merge_as_both_flood_merge_as_secondary
export connection_merge_as_both_flood_merge_not_set
export connection_merge_as_both_flood_merge_as_both
export convert_to_simple_merge_type_connect
export convert_to_simple_merge_type_flood
export no_merge
export null_mtype
export primary_merge
export secondary_merge
export double_merge
export null_simple_mtype
export zero

@enum MergeTypes begin
no_merge_mtype = 0
connection_merge_as_primary_flood_merge_as_primary = 1
connection_merge_as_primary_flood_merge_as_secondary = 2
connection_merge_as_primary_flood_merge_not_set = 3
connection_merge_as_primary_flood_merge_as_both = 4
connection_merge_as_secondary_flood_merge_as_primary = 5
connection_merge_as_secondary_flood_merge_as_secondary = 6
connection_merge_as_secondary_flood_merge_not_set = 7
connection_merge_as_secondary_flood_merge_as_both = 8
connection_merge_not_set_flood_merge_as_primary = 9
connection_merge_not_set_flood_merge_as_secondary = 10
connection_merge_not_set_flood_merge_as_both = 11
connection_merge_as_both_flood_merge_as_primary = 12
connection_merge_as_both_flood_merge_as_secondary = 13
connection_merge_as_both_flood_merge_not_set = 14
connection_merge_as_both_flood_merge_as_both = 15
null_mtype = 16
end

function zero(::Type{MergeTypes})
  return no_merge_mtype::MergeTypes
end

@enum SimpleMergeTypes begin
no_merge = 0
primary_merge = 1
secondary_merge = 2
double_merge = 3
null_simple_mtype = 4
end

const convert_to_simple_merge_type_connect =
  SimpleMergeTypes[no_merge primary_merge primary_merge #=
                =# primary_merge primary_merge secondary_merge #=
                =# secondary_merge secondary_merge secondary_merge #=
                =# no_merge no_merge no_merge double_merge #=
                =# double_merge double_merge double_merge #=
                =# null_simple_mtype]
const convert_to_simple_merge_type_flood =
  SimpleMergeTypes[no_merge primary_merge secondary_merge #=
                =# no_merge double_merge primary_merge #=
                =# secondary_merge no_merge double_merge #=
                =# primary_merge secondary_merge double_merge #=
                =# primary_merge secondary_merge no_merge #=
                =# double_merge null_simple_mtype]

end
