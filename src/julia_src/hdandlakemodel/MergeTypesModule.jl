module MergeTypesModule

export MergeTypes
export no_merge
export primary_merge
export secondary_merge
export null_simple_mtype

@enum MergeTypes begin
no_merge = 0
primary_merge = 1
secondary_merge = 2
null_simple_mtype = 3
end

end
