module latlon_lake_model_merges_io_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

subroutine testLakeModelMergeIO
    use latlon_lake_model_io_mod
    use parameters_mod
    type(lakeparameters), pointer ::  lake_parameters
    type(mergeandredirectindices), pointer :: working_merge_and_redirect_indices
    character(len = max_name_length) :: lakepara_test_data_filename
        lakepara_test_data_filename = &
            "/Users/thomasriddick/Documents/data/unit_test_data/lakepara_test_data.nc"
        call set_lake_parameters_filename(lakepara_test_data_filename)
        lake_parameters => read_lake_parameters(.true.)
            working_merge_and_redirect_indices => &
            mergeandredirectindices(.false., &
                                    .true., &
                                    2, &
                                    3, &
                                    4, &
                                    5)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(1)&
                         &%ptr%secondary_merge_and_redirect_indices%is_equal_to(working_merge_and_redirect_indices))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(1)&
                                    &%ptr%primary_merge_and_redirect_indices(1)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(1)&
                         &%ptr%primary_merge_and_redirect_indices(2)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(1)&
                         &%ptr%primary_merge_and_redirect_indices(3)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(1)&
                         &%ptr%primary_merge_and_redirect_indices(4)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(2)&
                         &%ptr%secondary_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .false., &
                                        6, &
                                        7, &
                                        8, &
                                        9)
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(2)&
                         &%ptr%primary_merge_and_redirect_indices(1)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(2)&
                         &%ptr%primary_merge_and_redirect_indices(2)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(2)&
                         &%ptr%primary_merge_and_redirect_indices(3)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(2)&
                         &%ptr%primary_merge_and_redirect_indices(4)%ptr))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.false., &
                                        .true., &
                                        10, &
                                        11, &
                                        12, &
                                        13)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(3)&
                         &%ptr%secondary_merge_and_redirect_indices%is_equal_to(working_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .false., &
                                        14, &
                                        15, &
                                        16, &
                                        17)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(3)&
                         &%ptr%primary_merge_and_redirect_indices(1)%ptr%is_equal_to(working_merge_and_redirect_indices))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(3)&
                         &%ptr%primary_merge_and_redirect_indices(2)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(3)&
                         &%ptr%primary_merge_and_redirect_indices(3)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(3)&
                         &%ptr%primary_merge_and_redirect_indices(4)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(4)&
                         &%ptr%secondary_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .true., &
                                        18, &
                                        19, &
                                        20, &
                                        21)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(4)&
                         &%ptr%primary_merge_and_redirect_indices(1)%ptr%is_equal_to(working_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .false., &
                                        22, &
                                        23, &
                                        24, &
                                        25)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(4)&
                         &%ptr%primary_merge_and_redirect_indices(2)%ptr%is_equal_to(working_merge_and_redirect_indices))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(4)&
                         &%ptr%primary_merge_and_redirect_indices(3)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(4)&
                         &%ptr%primary_merge_and_redirect_indices(4)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(5)&
                         &%ptr%secondary_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .true., &
                                        26, &
                                        27, &
                                        28, &
                                        29)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(5)&
                         &%ptr%primary_merge_and_redirect_indices(1)%ptr%is_equal_to(working_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .false., &
                                        30, &
                                        31, &
                                        32, &
                                        33)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(5)&
                         &%ptr%primary_merge_and_redirect_indices(2)%ptr%is_equal_to(working_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .false., &
                                        34, &
                                        35, &
                                        36, &
                                        37)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(5)&
                         &%ptr%primary_merge_and_redirect_indices(3)%ptr%is_equal_to(working_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .false., &
                                        38, &
                                        39, &
                                        40, &
                                        41)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(5)&
                         &%ptr%primary_merge_and_redirect_indices(4)%ptr%is_equal_to(working_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.false., &
                                        .true., &
                                        42, &
                                        43, &
                                        44, &
                                        45)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(6)&
                         &%ptr%secondary_merge_and_redirect_indices%is_equal_to(working_merge_and_redirect_indices))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(6)&
                         &%ptr%primary_merge_and_redirect_indices(1)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(6)&
                         &%ptr%primary_merge_and_redirect_indices(2)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(6)&
                         &%ptr%primary_merge_and_redirect_indices(3)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(6)&
                         &%ptr%primary_merge_and_redirect_indices(4)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(7)&
                         &%ptr%secondary_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .false., &
                                        46, &
                                        47, &
                                        48, &
                                        49)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(7)&
                         &%ptr%primary_merge_and_redirect_indices(1)%ptr%is_equal_to(working_merge_and_redirect_indices))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(7)&
                         &%ptr%primary_merge_and_redirect_indices(2)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(7)&
                         &%ptr%primary_merge_and_redirect_indices(3)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(7)&
                         &%ptr%primary_merge_and_redirect_indices(4)%ptr))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.false., &
                                        .true., &
                                        50, &
                                        51, &
                                        52, &
                                        53)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(8)&
                         &%ptr%secondary_merge_and_redirect_indices%is_equal_to(working_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .false., &
                                        54, &
                                        55, &
                                        56, &
                                        57)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(8)&
                         &%ptr%primary_merge_and_redirect_indices(1)%ptr%is_equal_to(working_merge_and_redirect_indices))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.true., &
                                        .false., &
                                        58, &
                                        59, &
                                        60, &
                                        61)
        call assert_true(lake_parameters%flood_merge_and_redirect_indices_collections(8)&
                         &%ptr%primary_merge_and_redirect_indices(2)%ptr%is_equal_to(working_merge_and_redirect_indices))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(8)&
                         &%ptr%primary_merge_and_redirect_indices(3)%ptr))
        call assert_true(associated(lake_parameters%flood_merge_and_redirect_indices_collections(8)&
                         &%ptr%primary_merge_and_redirect_indices(4)%ptr))
        working_merge_and_redirect_indices => &
                mergeandredirectindices(.false., &
                                        .false., &
                                        62, &
                                        63, &
                                        64, &
                                        65)
        call assert_true(lake_parameters%connect_merge_and_redirect_indices_collections(1)&
                         &%ptr%secondary_merge_and_redirect_indices%is_equal_to(working_merge_and_redirect_indices))
end subroutine testLakeModelMergeIO

end module latlon_lake_model_merges_io_test_mod
