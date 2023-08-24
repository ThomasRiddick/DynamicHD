module LakeModelParametersModule

struct LakeModelParameters
  instant_throughflow::Bool
  lake_retention_constant::Float64
  minimum_lake_volume_threshold::Float64
  function LakeModelParameters()
    instant_throughflow::Bool = true
    lake_retention_constant::Float64 = 0.1
    minimum_lake_volume_threshold::Float64 = 0.0000001
    return new(instant_throughflow,lake_retention_constant,
               minimum_lake_volume_threshold)
  end
end

end
