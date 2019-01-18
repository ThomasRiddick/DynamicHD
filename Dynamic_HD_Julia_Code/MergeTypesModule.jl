module MergeTypesModule

export MergeTypes
export SimpleMergeTypes
export no_merge_mtype
export connection_merge_as_primary_flood_merge_as_primary
export connection_merge_as_primary_flood_merge_as_secondary
export connection_merge_as_primary_flood_merge_not_set
export connection_merge_as_secondary_flood_merge_as_primary
export connection_merge_as_secondary_flood_merge_as_secondary
export connection_merge_as_secondary_flood_merge_not_set
export connection_merge_not_set_flood_merge_as_primary
export connection_merge_not_set_flood_merge_as_secondary
export convert_to_simple_merge_type_connect
export convert_to_simple_merge_type_flood
export no_merge
export primary_merge
export secondary_merge
export null_simple_mtype

@enum MergeTypes begin
no_merge_mtype = 0
connection_merge_as_primary_flood_merge_as_primary = 1
connection_merge_as_primary_flood_merge_as_secondary = 2
connection_merge_as_primary_flood_merge_not_set = 3
connection_merge_as_secondary_flood_merge_as_primary = 4
connection_merge_as_secondary_flood_merge_as_secondary = 5
connection_merge_as_secondary_flood_merge_not_set = 6
connection_merge_not_set_flood_merge_as_primary = 7
connection_merge_not_set_flood_merge_as_secondary = 8
null_mtype = 9
end

@enum SimpleMergeTypes begin
no_merge = 0
primary_merge = 1
secondary_merge = 2
null_simple_mtype = 3
end

const convert_to_simple_merge_type_connect =
  SimpleMergeTypes[no_merge primary_merge primary_merge #=
                =# primary_merge secondary_merge secondary_merge #=
                =# secondary_merge no_merge no_merge null_simple_mtype]
const convert_to_simple_merge_type_flood =
  SimpleMergeTypes[no_merge primary_merge secondary_merge #=
                   =# no_merge primary_merge secondary_merge #=
                   =# no_merge primary_merge secondary_merge null_simple_mtype]

end
