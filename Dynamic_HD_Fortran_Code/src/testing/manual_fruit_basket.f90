module manual_fruit_basket
use fruit
implicit none

logical :: verbose = .True.

contains

    subroutine flow_all_tests
    use flow_test_mod
        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSmallGrid"
        call set_unit_name('test_something')
        call run_test_case(testSmallGrid,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSmallGridTwo"
        call set_unit_name('test_something')
        call run_test_case(testSmallGridTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSmallGridThree"
        call set_unit_name('test_something')
        call run_test_case(testSmallGridThree,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown
    end subroutine flow_all_tests

    subroutine cotat_plus_all_tests
    use cotat_plus_test_mod

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSmallGrid"
        call set_unit_name('test_something')
        call run_test_case(testSmallGrid,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSmallGridTwo"
        call set_unit_name('test_something')
        call run_test_case(testSmallGridTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSmallGridLatLonToIcon"
        call set_unit_name('test_something')
        call run_test_case(testSmallGridLatLonToIcon,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test_small_grid_latlon_to_icon")
        else
            call case_passed_xml("test_something","test_small_grid_latlon_to_icon")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSmallGridLatLonToIconTwo"
        call set_unit_name('test_something')
        call run_test_case(testSmallGridLatLonToIconTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test_small_grid_latlon_to_icon_two")
        else
            call case_passed_xml("test_something","test_small_grid_latlon_to_icon_two")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSmallGridLatLonToIconThree"
        call set_unit_name('test_something')
        call run_test_case(testSmallGridLatLonToIconThree,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test_small_grid_latlon_to_icon_three")
        else
            call case_passed_xml("test_something","test_small_grid_latlon_to_icon_three")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testFindingLocalizedLoops"
        call set_unit_name('test_something')
        call run_test_case(testFindingLocalizedLoops,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test_latlon_to_icon_finding_localized_loop")
        else
            call case_passed_xml("test_something","test_latlon_to_icon_finding_localized_loop")
        end if
        call teardown

    end subroutine cotat_plus_all_tests

    subroutine area_all_tests
    use area_test_mod

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCheckFieldForLocalizedLoops"
        call set_unit_name('test_something')
        call run_test_case(testCheckFieldForLocalizedLoops,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCellCumulativeFlowGeneration"
        call set_unit_name('test_something')
        call run_test_case(testCellCumulativeFlowGeneration,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCellCumulativeFlowGenerationTwo"
        call set_unit_name('test_something')
        call run_test_case(testCellCumulativeFlowGenerationTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupOne"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupOne,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwo"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThree"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThree,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupFour"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupFour,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupFive"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupFive,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupSix"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupSix,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupSeven"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupSeven,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupEight"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupEight,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupNine"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupNine,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTen"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupEleven"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupEleven,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwelve"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwelve,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirteen"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupFourteen"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupFourteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupFifthteen"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupFifteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSixteen"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupSixteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupSeventeen"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupSeventeen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellEighteen"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupEighteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupNineteen"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupNineteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwenty"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwenty,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwentyOne"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwentyOne,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwentyTwo"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwentyTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwentyThree"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwentyThree,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwentyFour"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwentyFour,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwentyFive"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwentyFive,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwentySix"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwentySix,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwentySeven"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwentySeven,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwentyEight"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwentyEight,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupTwentyNine"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupTwentyNine,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirty"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirty,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirtyOne"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirtyOne,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirtyTwo"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirtyTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirtyThree"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirtyThree,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirtyFour"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirtyFour,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirtyFive"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirtyFive,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirtySix"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirtySix,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirtySeven"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirtySeven,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirtyEight"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirtyEight,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupThirtyNine"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupThirtyNine,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupForty"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupForty,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupFortyOne"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupFortyOne,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupFortyTwo"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupFortyTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupFortyThree"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupFortyThree,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupFortyFour"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupFortyFour,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupFortyFive"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupFortyFive,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testOneOutflowCellSetupFortySix"
        call set_unit_name('test_something')
        call run_test_case(testOneOutflowCellSetupFortySix,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellOne"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellOne,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellTwo"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellThree"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellThree,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellFour"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellFour,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellFive"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellFive,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellSix"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellSix,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellSeven"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellSeven,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellEight"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellEight,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellNine"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellNine,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellTen"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellTen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellEleven"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellEleven,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellTwelve"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellTwelve,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellThirteen"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellThirteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellFourteen"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellFourteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellFifteen"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellFifteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellSixteen"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellSixteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellSeventeen"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellSeventeen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellEighteen"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellEighteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellNineteen"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellNineteen,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellTwenty"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellTwenty,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellTwentyOne"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellTwentyOne,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellTwentyTwo"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellTwentyTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiFindDownstreamCellTwentyThree"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiFindDownstreamCellTwentyThree,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_find_downstream_cell_test")
        else
            call case_passed_xml("test_something","yamazaki_find_downstream_cell_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiLatLonCalculateRiverDirectionsAsIndicesOne"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiLatLonCalculateRiverDirectionsAsIndicesOne,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_calculate_river_directions_as_indices_test")
        else
            call case_passed_xml("test_something","yamazaki_calculate_river_directions_as_indices_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiLatLonCalculateRiverDirectionsAsIndicesTwo"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiLatLonCalculateRiverDirectionsAsIndicesTwo,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_calculate_river_directions_as_indices_test")
        else
            call case_passed_xml("test_something","yamazaki_calculate_river_directions_as_indices_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiLatLonCalculateRiverDirectionsAsIndicesThree"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiLatLonCalculateRiverDirectionsAsIndicesThree,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_calculate_river_directions_as_indices_test")
        else
            call case_passed_xml("test_something","yamazaki_calculate_river_directions_as_indices_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testYamazakiLatLonCalculateRiverDirectionsAsIndicesFour"
        call set_unit_name('test_something')
        call run_test_case(testYamazakiLatLonCalculateRiverDirectionsAsIndicesFour,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","yamazaki_calculate_river_directions_as_indices_test")
        else
            call case_passed_xml("test_something","yamazaki_calculate_river_directions_as_indices_test")
        end if
        call teardown

    end subroutine area_all_tests

    subroutine subfield_all_tests
    use subfield_test_mod

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testConstructorAndGettingAndSettingIntegers"
        call set_unit_name('test_something')
        call run_test_case(testConstructorAndGettingAndSettingIntegers,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testConstructorAndGettingAndSettingReals"
        call set_unit_name('test_something')
        call run_test_case(testConstructorAndGettingAndSettingReals,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testConstructorAndGettingAndSettingLogicals"
        call set_unit_name('test_something')
        call run_test_case(testConstructorAndGettingAndSettingLogicals,&
                           'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

    end subroutine subfield_all_tests

    subroutine doubly_linked_list_all_tests
    use doubly_linked_list_test_module
        call setup
        if (verbose) write (*,('(/A)')) "..running test: testAddingIntegersAndIteration"
        call set_unit_name('test_something')
        call run_test_case(testAddingIntegersAndIteration,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testRemoveElementAtIteratorPosition"
        call set_unit_name('test_something')
        call run_test_case(testRemoveElementAtIteratorPosition,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testAddingReals"
        call set_unit_name('test_something')
        call run_test_case(testAddingReals,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testAddingLogicals"
        call set_unit_name('test_something')
        call run_test_case(testAddingLogicals,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testAddingCoords"
        call set_unit_name('test_something')
        call run_test_case(testAddingCoords,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

    end subroutine doubly_linked_list_all_tests

    subroutine field_section_all_tests
        use field_section_test_mod

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testConstructorAndGettersUsingIntegers"
        call set_unit_name('test_something')
        call run_test_case(testConstructorAndGettersUsingIntegers,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSettingAndGettingIntegers"
        call set_unit_name('test_something')
        call run_test_case(testSettingAndGettingIntegers,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSettingAndGettinReals"
        call set_unit_name('test_something')
        call run_test_case(testSettingAndGettingReals,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testSettingAndGettingLogicals"
        call set_unit_name('test_something')
        call run_test_case(testSettingAndGettingLogicals,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

    end subroutine field_section_all_tests

    subroutine loop_breaker_all_tests
    use loop_breaker_test_mod

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testLoopBreaker"
        call set_unit_name('test_something')
        call run_test_case(testLoopBreaker,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: TestLoopBreakerTwo"
        call set_unit_name('test_something')
        call run_test_case(testLoopBreakerTwo,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testLoopBreakerThree"
        call set_unit_name('test_something')
        call run_test_case(testLoopBreakerThree,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testLoopBreakerFour"
        call set_unit_name('test_something')
        call run_test_case(testLoopBreakerFour,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testLoopBreakerFive"
        call set_unit_name('test_something')
        call run_test_case(testLoopBreakerFive,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testLoopBreakerSix"
        call set_unit_name('test_something')
        call run_test_case(testLoopBreakerSix,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testLoopBreakerSeven"
        call set_unit_name('test_something')
        call run_test_case(testLoopBreakerSeven,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","wrapped_grid_section_test")
        else
            call case_passed_xml("test_something","wrapped_grid_section_test")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testLoopBreakerEight"
        call set_unit_name('test_something')
        call run_test_case(testLoopBreakerEight,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","double_loop_test")
        else
            call case_passed_xml("test_something","double_loop_test")
        end if
        call teardown

    end subroutine loop_breaker_all_tests

    subroutine latlon_hd_and_lake_model_all_tests
    use latlon_hd_model_test_mod
    use lake_model_array_decoder_test_mod
    use latlon_lake_model_test_mod
    use calculate_lake_fractions_test_mod
    use lake_model_input_test_mod


        if (verbose) write (*,('(/A)')) "..running test: testArrayDecoderOneLake"
        call set_unit_name('test_something')
        call run_test_case(testArrayDecoderOneLake,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Array decoder test one lake")
        else
            call case_passed_xml("test_something","Array decoder test one lake")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testArrayDecoderTwoLakes"
        call set_unit_name('test_something')
        call run_test_case(testArrayDecoderTwoLakes,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Array decoder test two lake")
        else
            call case_passed_xml("test_something","Array decoder test two lakes")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest1"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest1,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 1")
        else
            call case_passed_xml("test_something","Lake fraction calc 1")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest2"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest2,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 2")
        else
            call case_passed_xml("test_something","Lake fraction calc 2")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest3"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest3,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 3")
        else
            call case_passed_xml("test_something","Lake fraction calc 3")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest4"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest4,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 4")
        else
            call case_passed_xml("test_something","Lake fraction calc 4")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest5"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest5,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 5")
        else
            call case_passed_xml("test_something","Lake fraction calc 5")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest6"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest6,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 6")
        else
            call case_passed_xml("test_something","Lake fraction calc 6")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest7"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest7,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 7")
        else
            call case_passed_xml("test_something","Lake fraction calc 7")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest8"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest8,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 8")
        else
            call case_passed_xml("test_something","Lake fraction calc 8")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest9"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest9,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 9")
        else
            call case_passed_xml("test_something","Lake fraction calc 9")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest10"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest10,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 10")
        else
            call case_passed_xml("test_something","Lake fraction calc 10")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest11"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest11,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 11")
        else
            call case_passed_xml("test_something","Lake fraction calc 11")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest12"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest12,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 12")
        else
            call case_passed_xml("test_something","Lake fraction calc 12")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest13"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest13,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 13")
        else
            call case_passed_xml("test_something","Lake fraction calc 13")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest14"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest14,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 14")
        else
            call case_passed_xml("test_something","Lake fraction calc 14")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest15"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest15,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 15")
        else
            call case_passed_xml("test_something","Lake fraction calc 15")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest16"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest16,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 16")
        else
            call case_passed_xml("test_something","Lake fraction calc 16")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest17"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest17,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 17")
        else
            call case_passed_xml("test_something","Lake fraction calc 17")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest18"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest18,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 18")
        else
            call case_passed_xml("test_something","Lake fraction calc 18")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest19"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest19,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 19")
        else
            call case_passed_xml("test_something","Lake fraction calc 19")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest20"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest20,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 20")
        else
            call case_passed_xml("test_something","Lake fraction calc 20")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeFractionCalculationTest21"
        call set_unit_name('test_something')
        call run_test_case(testLakeFractionCalculationTest21,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake fraction calc 21")
        else
            call case_passed_xml("test_something","Lake fraction calc 21")
        end if

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testHdModel"
        call set_unit_name('test_something')
        call run_test_case(testHdModel,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","HD Model test")
        else
            call case_passed_xml("test_something","HD Model test")
        end if
        call teardown

        if (verbose) write (*,('(/A)')) "..running test: testRedirectDictionary1"
        call set_unit_name('test_something')
        call run_test_case(testRedirectDictionary1,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Redirect Dictionary test 1")
        else
            call case_passed_xml("test_something","Redirect Dictionary test 1")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testRedirectDictionary2"
        call set_unit_name('test_something')
        call run_test_case(testRedirectDictionary2,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Redirect Dictionary test 2")
        else
            call case_passed_xml("test_something","Redirect Dictionary test 2")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testRedirectDictionary3"
        call set_unit_name('test_something')
        call run_test_case(testRedirectDictionary3,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Redirect Dictionary test 3")
        else
            call case_passed_xml("test_something","Redirect Dictionary test 3")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testRedirectDictionary4"
        call set_unit_name('test_something')
        call run_test_case(testRedirectDictionary4,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Redirect Dictionary test 4")
        else
            call case_passed_xml("test_something","Redirect Dictionary test 4")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testRedirectDictionary5"
        call set_unit_name('test_something')
        call run_test_case(testRedirectDictionary5,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Redirect Dictionary test 5")
        else
            call case_passed_xml("test_something","Redirect Dictionary test 5")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testRedirectDictionary6"
        call set_unit_name('test_something')
        call run_test_case(testRedirectDictionary6,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Redirect Dictionary test 6")
        else
            call case_passed_xml("test_something","Redirect Dictionary test 6")
        end if

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel1"
        call set_unit_name('test_something')
        call run_test_case(testLakeModel1,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 1")
        else
            call case_passed_xml("test_something","Lake Model test 1")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel2"
        call set_unit_name('test_something')
        call run_test_case(testLakeModel2,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 2")
        else
            call case_passed_xml("test_something","Lake Model test 2")
        end if
        !call teardown


        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel3"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel3,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 3")
        else
            call case_passed_xml("test_something","Lake Model test 3")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel4"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel4,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 4")
        else
            call case_passed_xml("test_something","Lake Model test 4")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel5"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel5,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 5")
        else
            call case_passed_xml("test_something","Lake Model test 5")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel6"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel6,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 6")
        else
            call case_passed_xml("test_something","Lake Model test 6")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel7"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel7,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 7")
        else
            call case_passed_xml("test_something","Lake Model test 7")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel8"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel8,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 8")
        else
            call case_passed_xml("test_something","Lake Model test 8")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel9"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel9,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 9")
        else
            call case_passed_xml("test_something","Lake Model test 9")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel10"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel10,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 10")
        else
            call case_passed_xml("test_something","Lake Model test 10")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel11"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel11,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 11")
        else
            call case_passed_xml("test_something","Lake Model test 11")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel12"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel12,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 12")
        else
            call case_passed_xml("test_something","Lake Model test 12")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel13"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel13,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 13")
        else
            call case_passed_xml("test_something","Lake Model test 13")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel14"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel14,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 14")
        else
            call case_passed_xml("test_something","Lake Model test 14")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel15"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel15,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 15")
        else
            call case_passed_xml("test_something","Lake Model test 15")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel16"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel16,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 16")
        else
            call case_passed_xml("test_something","Lake Model test 16")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel17"
        call set_unit_name('Lake model with Evaporation')
        call run_test_case(testLakeModel17,'Lake model with evaporation')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 17")
        else
            call case_passed_xml("test_something","Lake Model test 17")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel18"
        call set_unit_name('Lake model')
        call run_test_case(testLakeModel18,'Lake model')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 18")
        else
            call case_passed_xml("test_something","Lake Model test 18")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel19"
        call set_unit_name('Lake model')
        call run_test_case(testLakeModel19,'Lake model')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 19")
        else
            call case_passed_xml("test_something","Lake Model test 19")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel20"
        call set_unit_name('Lake model')
        call run_test_case(testLakeModel20,'Lake model')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 20")
        else
            call case_passed_xml("test_something","Lake Model test 20")
        end if
        !call teardown

        !call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel21"
        call set_unit_name('Lake model')
        call run_test_case(testLakeModel21,'Lake model')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 21")
        else
            call case_passed_xml("test_something","Lake Model test 21")
        end if
        !call teardown

        if (verbose) write (*,('(/A)')) "..running test: testLakeModel22"
        call set_unit_name('Lake model')
        call run_test_case(testLakeModel22,'Lake model')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 22")
        else
            call case_passed_xml("test_something","Lake Model test 22")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeInputTests1"
        call set_unit_name('Lake model')
        call run_test_case(testLakeInputTests1,'Lake model')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model Input Test 1")
        else
            call case_passed_xml("test_something","Lake Model Input Test 1")
        end if

        if (verbose) write (*,('(/A)')) "..running test: testLakeInputTests2"
        call set_unit_name('Lake model')
        call run_test_case(testLakeInputTests2,'Lake model')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model Input Test 2")
        else
            call case_passed_xml("test_something","Lake Model Input Test 2")
        end if

    end subroutine latlon_hd_and_lake_model_all_tests

    subroutine latlon_lake_model_tree_all_tests
    use lake_model_tree_test_mod

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testRootedTrees"
        call set_unit_name('Test rooted trees')
        call run_test_case(testRootedTrees,'General tests of rooted trees')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Rooted Tree Test")
        else
            call case_passed_xml("test_something","Rooted Tree Test")
        end if
        call teardown

    end subroutine latlon_lake_model_tree_all_tests

    subroutine icosohedral_hd_model_all_tests
    use icosohedral_hd_and_lake_model_test_mod

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testHdModel - ICON grid"
        call set_unit_name('test_something')
        call run_test_case(testHdModel,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","HD Model test")
        else
            call case_passed_xml("test_something","HD Model test")
        end if
        call teardown

    end subroutine icosohedral_hd_all_tests

    subroutine map_non_coincident_grids_all_tests
        use map_non_coincident_grids_test_mod
        call setup
        if (verbose) write (*,('(/A)')) "..running test: testLatLonToIconGrids"
        call set_unit_name('test_something')
        call run_test_case(testLatLonToIconGrids,'Test LatLon to Icon Grid Mapping')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Test LatLon to Icon Grid Mapping")
        else
            call case_passed_xml("test_something","Test LatLon to Icon Grid Mapping")
        end if
        call teardown
    end subroutine map_non_coincident_grids_all_tests


    subroutine accumulate_flow_all_tests
        use accumulate_flow_test_mod

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCalculateCumlativeFlowLatLon"
        call set_unit_name('test_something')
        call run_test_case(testCalculateCumlativeFlowLatLon,&
                           'test calculating cumulative flow on an ICON grid')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test calculating cumulative flow on an ICON grid")
        else
            call case_passed_xml("test_something","test calculating cumulative flow on an ICON grid")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCalculateCumlativeFlowLatLonWrapped"
        call set_unit_name('test_something')
        call run_test_case(testCalculateCumlativeFlowLatLonWrapped,&
                           'test calculating cumulative flow on an ICON grid')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test calculating cumulative flow on an ICON grid")
        else
            call case_passed_xml("test_something","test calculating cumulative flow on an ICON grid")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCalculateCumlativeFlowLatLonWithMask"
        call set_unit_name('test_something')
        call run_test_case(testCalculateCumlativeFlowLatLonWithMask,&
                           'test calculating cumulative flow on an ICON grid')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test calculating cumulative flow on an ICON grid")
        else
            call case_passed_xml("test_something","test calculating cumulative flow on an ICON grid")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCalculateCumlativeFlowLatLonWithBasicLoop"
        call set_unit_name('test_something')
        call run_test_case(testCalculateCumlativeFlowLatLonWithBasicLoop,&
                           'test calculating cumulative flow on an ICON grid')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test calculating cumulative flow on an ICON grid")
        else
            call case_passed_xml("test_something","test calculating cumulative flow on an ICON grid")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCalculateCumlativeFlowLatLonWithLoop"
        call set_unit_name('test_something')
        call run_test_case(testCalculateCumlativeFlowLatLonWithLoop,&
                           'test calculating cumulative flow on an ICON grid')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test calculating cumulative flow on an ICON grid")
        else
            call case_passed_xml("test_something","test calculating cumulative flow on an ICON grid")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCalculateCumulativeFlow"
        call set_unit_name('test_something')
        call run_test_case(testCalculateCumulativeFlow,'test calculating cumulative flow on an ICON grid')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test calculating cumulative flow on an ICON grid")
        else
            call case_passed_xml("test_something","test calculating cumulative flow on an ICON grid")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCalculateCumulativeFlowWithZeroBifurcations"
        call set_unit_name('test_something')
        call run_test_case(testCalculateCumulativeFlowWithZeroBifurcations,&
                           'test calculating cumulative flow on an ICON grid')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test calculating cumulative flow on an ICON grid")
        else
            call case_passed_xml("test_something","test calculating cumulative flow on an ICON grid")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testCalculateCumulativeFlowWithBifurcations"
        call set_unit_name('test_something')
        call run_test_case(testCalculateCumulativeFlowWithBifurcations,&
                           'test calculating cumulative flow on an ICON grid')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","test calculating cumulative flow on an ICON grid")
        else
            call case_passed_xml("test_something","test calculating cumulative flow on an ICON grid")
        end if
        call teardown

    end subroutine accumulate_flow_all_tests

end module manual_fruit_basket
