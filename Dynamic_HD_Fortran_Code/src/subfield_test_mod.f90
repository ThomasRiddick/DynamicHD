module subfield_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

        subroutine testConstructorAndGettingAndSettingReals
        use subfield_mod
        use coords_mod
        type(latlon_subfield), pointer :: field
        class(*), dimension(:,:), pointer :: data => null()
        class(*), pointer :: value => null()
        real :: result1, result2, result3, result4, result5, result6
        real :: result7, result8, result9, result10, result11, result12
        type(latlon_section_coords) :: latlon_section_coords_test
            allocate(real::data(8,8))
            select type(data)
            type is (real)
                 data = transpose(reshape((/ 11.01, 12.01, 13.01, 14.01, 15.01, 16.01, 17.01, 18.01, &
                                             21.01, 22.01, 23.01, 24.01, 25.01, 26.01, 27.01, 28.01, &
                                             31.01, 32.01, 33.01, 34.01, 35.01, 36.01, 37.01, 38.01, &
                                             41.01, 42.01, 43.01, 44.01, 45.01, 46.01, 47.01, 48.01, &
                                             51.01, 52.01, 53.01, 54.01, 55.01, 56.01, 57.01, 58.01, &
                                             61.01, 62.01, 63.01, 64.01, 65.01, 66.01, 67.01, 68.01, &
                                             71.01, 72.01, 73.01, 74.01, 75.01, 76.01, 77.01, 78.01, &
                                             81.01, 82.01, 83.01, 84.01, 85.01, 86.01, 87.01, 88.01 /),&
                                             shape(data)))
            end select
            latlon_section_coords_test = latlon_section_coords(7,11,8,8)
            field => latlon_subfield(data,latlon_section_coords_test)
            value => field%get_value(latlon_coords(7,11))
            select type (value)
            type is (real)
                result1 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,18))
            select type (value)
            type is (real)
                result2 = value
            end select
            deallocate(value)
            value  => field%get_value(latlon_coords(7,18))
            select type (value)
            type is (real)
                result3 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,11))
            select type (value)
            type is (real)
                result4 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(11,13))
            select type (value)
            type is (real)
                result5 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(9,14))
            select type (value)
            type is (real)
                result6 = value
            end select
            deallocate(value)
            call field%set_value(latlon_coords(7,11),1101.02)
            call field%set_value(latlon_coords(14,18),8801.02)
            call field%set_value(latlon_coords(7,18),1801.02)
            call field%set_value(latlon_coords(14,11),8101.02)
            call field%set_value(latlon_coords(11,13),5301.02)
            call field%set_value(latlon_coords(9,14),3401.02)
            value => field%get_value(latlon_coords(7,11))
            select type (value)
            type is (real)
                result7 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,18))
            select type (value)
            type is (real)
                result8 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(7,18))
            select type (value)
            type is (real)
                result9 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,11))
            select type (value)
            type is (real)
                result10 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(11,13))
            select type (value)
            type is (real)
                result11 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(9,14))
            select type (value)
            type is (real)
                result12 = value
            end select
            deallocate(value)
            call assert_equals(11.01,result1)
            call assert_equals(88.01,result2)
            call assert_equals(18.01,result3)
            call assert_equals(81.01,result4)
            call assert_equals(53.01,result5)
            call assert_equals(34.01,result6)
            call assert_equals(1101.02,result7)
            call assert_equals(8801.02,result8)
            call assert_equals(1801.02,result9)
            call assert_equals(8101.02,result10)
            call assert_equals(5301.02,result11)
            call assert_equals(3401.02,result12)
            deallocate(field)
            deallocate(data)
    end subroutine testConstructorAndGettingAndSettingReals

    subroutine testConstructorAndGettingAndSettingIntegers
        use subfield_mod
        use coords_mod
        type(latlon_subfield), pointer :: field
        class(*), dimension(:,:), pointer :: data => null()
        class(*), pointer :: value => null()
        integer :: result1, result2, result3, result4, result5, result6
        integer :: result7, result8, result9, result10, result11, result12
        type(latlon_section_coords) :: latlon_section_coords_test
            allocate(integer::data(8,8))
            select type(data)
            type is (integer)
                 data = transpose(reshape((/ 11, 12, 13, 14, 15, 16, 17, 18, &
                                             21, 22, 23, 24, 25, 26, 27, 28, &
                                             31, 32, 33, 34, 35, 36, 37, 38, &
                                             41, 42, 43, 44, 45, 46, 47, 48, &
                                             51, 52, 53, 54, 55, 56, 57, 58, &
                                             61, 62, 63, 64, 65, 66, 67, 68, &
                                             71, 72, 73, 74, 75, 76, 77, 78, &
                                             81, 82, 83, 84, 85, 86, 87, 88/),&
                                             shape(data)))
            end select
            latlon_section_coords_test = latlon_section_coords(7,11,8,8)
            field => latlon_subfield(data,latlon_section_coords_test)
            value => field%get_value(latlon_coords(7,11))
            select type (value)
            type is (integer)
                result1 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,18))
            select type (value)
            type is (integer)
                result2 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(7,18))
            select type (value)
            type is (integer)
                result3 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,11))
            select type (value)
            type is (integer)
                result4 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(11,13))
            select type (value)
            type is (integer)
                result5 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(9,14))
            select type (value)
            type is (integer)
                result6 = value
            end select
            deallocate(value)
            call field%set_value(latlon_coords(7,11),1101)
            call field%set_value(latlon_coords(14,18),8801)
            call field%set_value(latlon_coords(7,18),1801)
            call field%set_value(latlon_coords(14,11),8101)
            call field%set_value(latlon_coords(11,13),5301)
            call field%set_value(latlon_coords(9,14),3401)
            value => field%get_value(latlon_coords(7,11))
            select type (value)
            type is (integer)
                result7 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,18))
            select type (value)
            type is (integer)
                result8 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(7,18))
            select type (value)
            type is (integer)
                result9 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,11))
            select type (value)
            type is (integer)
                result10 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(11,13))
            select type (value)
            type is (integer)
                result11 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(9,14))
            select type (value)
            type is (integer)
                result12 = value
            end select
            deallocate(value)
            call assert_equals(11,result1)
            call assert_equals(88,result2)
            call assert_equals(18,result3)
            call assert_equals(81,result4)
            call assert_equals(53,result5)
            call assert_equals(34,result6)
            call assert_equals(1101,result7)
            call assert_equals(8801,result8)
            call assert_equals(1801,result9)
            call assert_equals(8101,result10)
            call assert_equals(5301,result11)
            call assert_equals(3401,result12)
            deallocate(field)
            deallocate(data)
    end subroutine testConstructorAndGettingAndSettingIntegers

    subroutine testConstructorAndGettingAndSettingLogicals
        use subfield_mod
        use coords_mod
        type(latlon_subfield), pointer :: field
        class(*), dimension(:,:), pointer :: data => null()
        class(*), pointer :: value => null()
        logical :: result1, result2, result3, result4, result5, result6
        logical :: result7, result8, result9, result10, result11, result12
        type(latlon_section_coords) :: latlon_section_coords_test
            allocate(logical::data(8,8))
            select type(data)
            type is (logical)
                 data = transpose(reshape((/ .True., .True., .True., .True., .True., .True., .True., .True., &
                                             .True., .True., .True., .True., .True., .True., .True., .True., &
                                             .True., .True., .True., .True., .True., .True., .True., .True., &
                                             .True., .True., .True., .True., .True., .True., .True., .True., &
                                             .True., .True., .True., .True., .True., .True., .True., .True., &
                                             .True., .True., .True., .True., .True., .True., .True., .True., &
                                             .True., .True., .True., .True., .True., .True., .True., .True., &
                                             .True., .True., .True., .True., .True., .True., .True., .True./),&
                                             shape(data)))
            end select
            latlon_section_coords_test = latlon_section_coords(7,11,8,8)
            field => latlon_subfield(data,latlon_section_coords_test)
            value => field%get_value(latlon_coords(7,11))
            select type (value)
            type is (logical)
                result1 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,18))
            select type (value)
            type is (logical)
                result2 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(7,18))
            select type (value)
            type is (logical)
                result3 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,11))
            select type (value)
            type is (logical)
                result4 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(11,13))
            select type (value)
            type is (logical)
                result5 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(9,14))
            select type (value)
            type is (logical)
                result6 = value
            end select
            deallocate(value)
            call field%set_value(latlon_coords(7,11),.False.)
            call field%set_value(latlon_coords(14,18),.False.)
            call field%set_value(latlon_coords(7,18),.False.)
            call field%set_value(latlon_coords(14,11),.False.)
            call field%set_value(latlon_coords(11,13),.False.)
            call field%set_value(latlon_coords(9,14),.False.)
            value => field%get_value(latlon_coords(7,11))
            select type (value)
            type is (logical)
                result7 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,18))
            select type (value)
            type is (logical)
                result8 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(7,18))
            select type (value)
            type is (logical)
                result9 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(14,11))
            select type (value)
            type is (logical)
                result10 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(11,13))
            select type (value)
            type is (logical)
                result11 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(9,14))
            select type (value)
            type is (logical)
                result12 = value
            end select
            deallocate(value)
            call assert_equals(.True.,result1)
            call assert_equals(.True.,result2)
            call assert_equals(.True.,result3)
            call assert_equals(.True.,result4)
            call assert_equals(.True.,result5)
            call assert_equals(.True.,result6)
            call assert_equals(.False.,result7)
            call assert_equals(.False.,result8)
            call assert_equals(.False.,result9)
            call assert_equals(.False.,result10)
            call assert_equals(.False.,result11)
            call assert_equals(.False.,result12)
            deallocate(field)
            deallocate(data)
    end subroutine testConstructorAndGettingAndSettingLogicals

end module subfield_test_mod
