module field_section_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

    subroutine testSettingAndGettingIntegers
        use field_section_mod
        use coords_mod
        type(latlon_field_section), pointer :: field
        class(*), dimension(:,:), pointer :: data => null()
        class(*), pointer :: value
        integer :: result1, result2, result3, result4, result5, result6
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
                                            81, 82, 83, 84, 85, 86, 87, 88 /), shape(data)))
            end select
            latlon_section_coords_test = latlon_section_coords(2,3,3,4)
            field => latlon_field_section(data,latlon_section_coords_test)
            value => field%get_value(latlon_coords(6,5))
            select type (value)
            type is (integer)
                result1 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(5,2))
            select type (value)
            type is (integer)
                result2 = value
            end select
            deallocate(value)
            call field%set_value(latlon_coords(5,2),5201)
            call field%set_value(latlon_coords(6,5),6501)
            value => field%get_value(latlon_coords(6,5))
            select type (value)
            type is (integer)
                result3 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(5,2))
            select type (value)
            type is (integer)
                result4 = value
            end select
            deallocate(value)
            call field%set_value(latlon_coords(1,13),1502)
            call field%set_value(latlon_coords(7,-6),7202)
            value => field%get_value(latlon_coords(1,5))
            select type (value)
            type is (integer)
                result5 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(7,2))
            select type (value)
            type is (integer)
                result6 = value
            end select
            deallocate(value)
            call assert_equals(65,result1)
            call assert_equals(52,result2)
            call assert_equals(6501,result3)
            call assert_equals(5201,result4)
            call assert_equals(1502,result5)
            call assert_equals(7202,result6)
            deallocate(data)
            deallocate(field)
    end subroutine testSettingAndGettingIntegers

    subroutine testSettingAndGettingReals
        use field_section_mod
        use coords_mod
        type(latlon_field_section), pointer :: field
        class(*), dimension(:,:), pointer :: data => null()
        class(*), pointer :: value
        real :: result1, result2, result3, result4, result5, result6
        type(latlon_section_coords) :: latlon_section_coords_test
            allocate(real::data(8,8))
            select type(data)
            type is (real)
                data = transpose(reshape((/ 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, &
                                            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, &
                                            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, &
                                            41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, &
                                            51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, &
                                            61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, &
                                            71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, &
                                            81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0 /), shape(data)))
            end select
            latlon_section_coords_test = latlon_section_coords(2,3,3,4)
            field => latlon_field_section(data,latlon_section_coords_test)
            value => field%get_value(latlon_coords(6,5))
            select type (value)
            type is (real)
                result1 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(5,2))
            select type (value)
            type is (real)
                result2 = value
            end select
            deallocate(value)
            call field%set_value(latlon_coords(5,2),5201.2)
            call field%set_value(latlon_coords(6,5),6501.5)
            value => field%get_value(latlon_coords(6,5))
            select type (value)
            type is (real)
                result3 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(5,2))
            select type (value)
            type is (real)
                result4 = value
            end select
            deallocate(value)
            call field%set_value(latlon_coords(1,13),1502.2)
            call field%set_value(latlon_coords(7,-6),7202.5)
            value => field%get_value(latlon_coords(1,5))
            select type (value)
            type is (real)
                result5 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(7,2))
            select type (value)
            type is (real)
                result6 = value
            end select
            deallocate(value)
            call assert_equals(65.0,result1)
            call assert_equals(52.0,result2)
            call assert_equals(6501.5,result3)
            call assert_equals(5201.2,result4)
            call assert_equals(1502.2,result5)
            call assert_equals(7202.5,result6)
            deallocate(data)
            deallocate(field)
    end subroutine testSettingAndGettingReals

    subroutine testSettingAndGettingLogicals
        use field_section_mod
        use coords_mod
        type(latlon_field_section), pointer :: field
        class(*), dimension(:,:), pointer :: data => null()
        class(*), pointer :: value
        logical :: result1, result2, result3, result4, result5, result6
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
                                            .True., .True., .True., .True., .True., .True., .True., .True. /),&
                                            shape(data)))
            end select
            latlon_section_coords_test = latlon_section_coords(2,3,3,4)
            field => latlon_field_section(data,latlon_section_coords_test)
            value => field%get_value(latlon_coords(6,5))
            select type (value)
            type is (logical)
                result1 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(5,2))
            select type (value)
            type is (logical)
                result2 = value
            end select
            deallocate(value)
            call field%set_value(latlon_coords(5,2),.False.)
            call field%set_value(latlon_coords(6,5),.False.)
            value => field%get_value(latlon_coords(6,5))
            select type (value)
            type is (logical)
                result3 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(5,2))
            select type (value)
            type is (logical)
                result4 = value
            end select
            deallocate(value)
            call field%set_value(latlon_coords(1,13),.False.)
            call field%set_value(latlon_coords(7,-6),.False.)
            value => field%get_value(latlon_coords(1,5))
            select type (value)
            type is (logical)
                result5 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(7,2))
            select type (value)
            type is (logical)
                result6 = value
            end select
            deallocate(value)
            call assert_equals(.True.,result1)
            call assert_equals(.True.,result2)
            call assert_equals(.False.,result3)
            call assert_equals(.False.,result4)
            call assert_equals(.False.,result5)
            call assert_equals(.False.,result6)
            deallocate(data)
            deallocate(field)
    end subroutine testSettingAndGettingLogicals

    subroutine testConstructorAndGettersUsingIntegers
        use field_section_mod
        use coords_mod
        type(latlon_field_section), pointer :: field
        class(*), dimension(:,:), pointer :: data => null()
        class(*), pointer :: value
        integer :: result1, result2, result3, result4, result5, result6
        integer :: result7, result8
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
                                            81, 82, 83, 84, 85, 86, 87, 88 /), shape(data)))
            end select
            latlon_section_coords_test = latlon_section_coords(2,3,3,4)
            field => latlon_field_section(data,latlon_section_coords_test)
            value => field%get_value(latlon_coords(6,5))
            select type (value)
            type is (integer)
                result1 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(5,2))
            select type (value)
            type is (integer)
                result2 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(6,-3))
            select type (value)
            type is (integer)
                result3 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(1,11))
            select type (value)
            type is (integer)
                result4 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(1,9))
            select type (value)
            type is (integer)
                result5 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(8,0))
            select type (value)
            type is (integer)
                result6 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(1,16))
            select type (value)
            type is (integer)
                result7 = value
            end select
            deallocate(value)
            value => field%get_value(latlon_coords(8,-7))
            select type (value)
            type is (integer)
                result8 = value
            end select
            deallocate(value)
            call assert_equals(65,result1)
            call assert_equals(52,result2)
            call assert_equals(65,result3)
            call assert_equals(13,result4)
            call assert_equals(11,result5)
            call assert_equals(88,result6)
            call assert_equals(18,result7)
            call assert_equals(81,result8)
            deallocate(data)
            deallocate(field)
    end subroutine testConstructorAndGettersUsingIntegers

end module field_section_test_mod
