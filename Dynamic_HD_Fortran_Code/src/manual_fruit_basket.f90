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

    end subroutine loop_breaker_all_tests

    subroutine latlon_hd_and_lake_model_all_tests
    use latlon_hd_and_lake_model_test_mod

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

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel1"
        call set_unit_name('test_something')
        call run_test_case(testLakeModel1,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 1")
        else
            call case_passed_xml("test_something","Lake Model test 1")
        end if
        call teardown

        call setup
        if (verbose) write (*,('(/A)')) "..running test: testLakeModel2"
        call set_unit_name('test_something')
        call run_test_case(testLakeModel2,'test_something')
        if (.not. is_case_passed()) then
            call case_failed_xml("test_something","Lake Model test 2")
        else
            call case_passed_xml("test_something","Lake Model test 2")
        end if
        call teardown

    end subroutine latlon_hd_and_lake_model_all_tests

end module manual_fruit_basket
