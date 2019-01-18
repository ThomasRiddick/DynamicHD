module LakeModelParametersModule

struct LakeModelParameters
  instant_throughflow::Bool
  lake_retention_constant::Float64
  function LakeModelParameters()
    instant_throughflow::Bool = true
    lake_retention_constant::Float64 = 0.01
    return new(instant_throughflow,lake_retention_constant)
  end
end

end
