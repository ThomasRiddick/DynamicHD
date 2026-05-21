'''
Created on Mar 3, 2021

@author: thomasriddick
'''

import unittest
import numpy as np
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.tools import connect_coarse_lake_catchments as cclc
from Dynamic_HD_Scripts.tools.connect_coarse_lake_catchments \
    import Redirect,ArrayDecoder

def compare_nested_dictionaries(dictx,dicty):
    if not set(dictx.keys()) == set(dicty.keys()):
        print("Difference in keys {}".format(set(dictx.keys()) - set(dicty.keys())))
        return False
    for key,value in list(dictx.items()):
        if value:
            if not compare_nested_dictionaries(value,dicty[key]):
                return False
        else:
            if dicty[key]:
                print("Second dictionary not empty when first is")
                return False
    return True

class TestArrayDecoder(unittest.TestCase):

    def testDecodingOneLake(self):
        #One lake
        array_in = \
            [1.0,68.0,1.0,-1.0,0.0,4.0,3.0,11.0,4.0,3.0,1.0,0.0,5.0,4.0,4.0,
             1.0,0.0,5.0,3.0,4.0,1.0,0.0,5.0,3.0,3.0,1.0,0.0,5.0,2.0,5.0,
             1.0,5.0,6.0,4.0,5.0,1.0,5.0,6.0,3.0,5.0,1.0,12.0,7.0,3.0,2.0,
             1.0,12.0,7.0,4.0,2.0,1.0,21.0,8.0,2.0,3.0,1.0,21.0,8.0,2.0,4.0,
             1.0,43.0,10.0,1.0,-1.0,2.0,2.0,0.0,5.0,11.0]
        lake_parameters = cclc.get_lake_parameters_from_array(array_in,2)
        self.assertEqual(len(lake_parameters),1)
        self.assertEqual(lake_parameters[0].center_coords,(4,3))
        self.assertEqual(lake_parameters[0].center_cell_coarse_coords,(2,2))
        self.assertEqual(lake_parameters[0].lake_number,1)
        self.assertEqual(lake_parameters[0].is_primary,True)
        self.assertEqual(lake_parameters[0].is_leaf,True)
        self.assertEqual(lake_parameters[0].primary_lake,-1)
        self.assertEqual(len(lake_parameters[0].secondary_lakes),0)
        #Returns dummy value
        self.assertEqual(len(lake_parameters[0].filling_order),0)
        self.assertEqual(len(lake_parameters[0].outflow_points),1)
        self.assertEqual(lake_parameters[0].
                         outflow_points[-1],
                         Redirect(False, -1, (2, 2)))
        self.assertEqual(lake_parameters[0].
                         lake_lower_boundary_height,5.0)
        self.assertEqual(lake_parameters[0].
                         filled_lake_area,11.0)

class TestCatchmentNodes(unittest.TestCase):

    def setUp(self):
        self.catchment_node = cclc.CatchmentNode()

    def testAddSuperCatchment(self):
        self.catchment_node.add_supercatchment(99,cclc.CatchmentNode(89))
        self.assertEqual(self.catchment_node.supercatchment_num,99)
        self.assertEqual(self.catchment_node.supercatchment_obj.supercatchment_num,89)

    def testAddSubCatchment(self):
        self.catchment_node.add_subcatchment(11,cclc.CatchmentNode(111))
        self.catchment_node.add_subcatchment(21,cclc.CatchmentNode(211))
        self.catchment_node.add_subcatchment(31,cclc.CatchmentNode(311))
        self.assertEqual(set(self.catchment_node.subcatchments.keys()),set([11,21,31]))
        self.assertEqual(self.catchment_node.subcatchments[11].supercatchment_num,111)
        self.assertEqual(self.catchment_node.subcatchments[21].supercatchment_num,211)
        self.assertEqual(self.catchment_node.subcatchments[31].supercatchment_num,311)

    def testAllSubCatchmentNumbers(self):
        subcatchment_obj = cclc.CatchmentNode()
        subcatchment_obj.add_subcatchment(111,cclc.CatchmentNode())
        self.catchment_node.add_subcatchment(11,subcatchment_obj)
        subsubcatchment_obj = cclc.CatchmentNode()
        subsubcatchment_obj.add_subcatchment(4111,cclc.CatchmentNode())
        subcatchment_obj = cclc.CatchmentNode()
        subcatchment_obj.add_subcatchment(411,subsubcatchment_obj)
        subsubcatchment_obj = cclc.CatchmentNode()
        subsubcatchment_obj.add_subcatchment(4112,cclc.CatchmentNode())
        subcatchment_obj.add_subcatchment(412,subsubcatchment_obj)
        self.catchment_node.add_subcatchment(41,subcatchment_obj)
        subcatchment_obj = cclc.CatchmentNode()
        subcatchment_obj.add_subcatchment(211,cclc.CatchmentNode())
        self.catchment_node.add_subcatchment(21,subcatchment_obj)
        subsubcatchment_obj = cclc.CatchmentNode()
        subsubcatchment_obj.add_subcatchment(5111,cclc.CatchmentNode())
        subsubcatchment_obj.add_subcatchment(5112,cclc.CatchmentNode())
        subcatchment_obj = cclc.CatchmentNode()
        subcatchment_obj.add_subcatchment(511,subsubcatchment_obj)
        subcatchment_obj.add_subcatchment(512,cclc.CatchmentNode())
        self.catchment_node.add_subcatchment(51,subcatchment_obj)
        subcatchment_obj = cclc.CatchmentNode()
        subcatchment_obj.add_subcatchment(311,cclc.CatchmentNode())
        self.catchment_node.add_subcatchment(31,subcatchment_obj)
        subsubcatchment_obj = cclc.CatchmentNode()
        subsubcatchment_obj.add_subcatchment(6111,cclc.CatchmentNode())
        subcatchment_obj = cclc.CatchmentNode()
        subcatchment_obj.add_subcatchment(611,subsubcatchment_obj)
        self.catchment_node.add_subcatchment(61,subcatchment_obj)
        self.assertEqual(set(self.catchment_node.get_all_subcatchment_nums()),
                         set([11,41,21,51,31,61,411,412,111,211,511,512,311,611,
                              4111,4112,5111,5112,6111]))

class TestCatchmentTrees(unittest.TestCase):

    def setUp(self):
        self.catchment_trees = cclc.CatchmentTrees()

    def testCatchmentTreesOne(self):
        expected_subcatchments = {1:{11:{111:{1111:{},1112:{}},112:{1121:{}}},
                                     12:{121:{1211:{},1221:{}}},
                                     13:{131:{1311:{}},132:{1321:{},
                                                            1322:{13221:{}}}}},
                                  2:{21:{211:{2111:{}},212:{2121:{}}},
                                     22:{221:{2211:{}}}},
                                  3:{31:{311:{},312:{3121:{31211:{}}}},
                                     32:{321:{}},
                                     33:{331:{},332:{}},
                                     34:{341:{3411:{},3412:{}}}},
                                  4:{41:{411:{4111:{}},412:{}}}}
        self.catchment_trees.add_link(1311,131)
        self.catchment_trees.add_link(1321,132)
        self.catchment_trees.add_link(13221,1322)
        self.catchment_trees.add_link(311,31)
        self.catchment_trees.add_link(321,32)
        self.catchment_trees.add_link(4111,411)
        self.catchment_trees.add_link(111,11)
        self.catchment_trees.add_link(41,4)
        self.catchment_trees.add_link(1111,111)
        self.catchment_trees.add_link(211,21)
        self.catchment_trees.add_link(2111,211)
        self.catchment_trees.add_link(2121,212)
        self.catchment_trees.add_link(212,21)
        self.catchment_trees.add_link(112,11)
        self.catchment_trees.add_link(1121,112)
        self.catchment_trees.add_link(1112,111)
        self.catchment_trees.add_link(121,12)
        self.catchment_trees.add_link(2211,221)
        self.catchment_trees.add_link(21,2)
        self.catchment_trees.add_link(33,3)
        self.catchment_trees.add_link(412,41)
        self.catchment_trees.add_link(411,41)
        self.catchment_trees.add_link(11,1)
        self.catchment_trees.add_link(13,1)
        self.catchment_trees.add_link(1211,121)
        self.catchment_trees.add_link(1221,121)
        self.catchment_trees.add_link(12,1)
        self.catchment_trees.add_link(131,13)
        self.catchment_trees.add_link(221,22)
        self.catchment_trees.add_link(22,2)
        self.catchment_trees.add_link(132,13)
        self.catchment_trees.add_link(1322,132)
        self.catchment_trees.add_link(312,31)
        self.catchment_trees.add_link(31211,3121)
        self.catchment_trees.add_link(3121,312)
        self.catchment_trees.add_link(331,33)
        self.catchment_trees.add_link(332,33)
        self.catchment_trees.add_link(3411,341)
        self.catchment_trees.add_link(341,34)
        self.catchment_trees.add_link(3412,341)
        self.catchment_trees.add_link(31,3)
        self.catchment_trees.add_link(32,3)
        self.catchment_trees.add_link(34,3)
        subcatchments = self.catchment_trees.get_nested_dictionary_of_subcatchments()
        self.assertTrue(compare_nested_dictionaries(subcatchments,expected_subcatchments))

class TestConnectCoarseLakeCatchments(unittest.TestCase):

    def testConnectCoarseLakeCatchmentsOne(self):
        expected_connected_catchments = np.asarray([[2,2,1,1,1],
                                                    [1,1,1,1,1],
                                                    [1,1,1,1,1],
                                                    [1,1,1,1,1],
                                                    [1,1,1,1,1]],
                                         dtype=np.int32, order='C')
        coarse_catchments =  np.asarray([[6,6,3,4,4],
                                         [3,3,3,7,4],
                                         [1,1,2,2,2],
                                         [1,1,2,9,8],
                                         [5,5,5,2,2]],
                                         dtype=np.int32, order='C')
        river_directions = np.asarray([[0,4,2,6,-2],
                                       [6,6,0,-2,8],
                                       [6,-2,2,4,4],
                                       [9,8,7,-2,-2],
                                       [6,-2,4,6,-2]],
                                       dtype=np.int32, order='C')
        lake_centers = np.asarray([[0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                                   [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,1,0],
                                   [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],

                                   [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                                   [0,0,0, 0,0,0, 0,0,0, 0,1,0, 0,0,0],
                                   [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],

                                   [0,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0],
                                   [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                                   [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],

                                   [0,0,0, 0,0,0, 0,0,0, 1,0,0, 0,0,1],
                                   [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                                   [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],

                                   [0,0,0, 0,1,0, 0,0,0, 0,0,0, 0,1,0],
                                   [0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                                   [0,0,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0]],
                                   dtype=np.int32, order='C')
        basin_catchment_numbers = np.asarray([
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  7,  7, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  7,  7, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  7,  7, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  6, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1,  4, -1, -1, -1, -1,  3],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  4, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
           [ 9,  9,  9,  9,  9,  9, -1, -1, -1,  9,  9,  9,  9,  9,  9],
           [ 9,  9,  9,  9,  9,  9,  9,  9, -1,  9,  9,  9,  9,  9,  9],
           [ 9,  9,  9,  9,  9,  9, -1, -1, -1,  9,  9,  9,  9,  9,  9]],
                                              dtype=np.int32, order='C')
        lakes_as_array = np.asarray(
            [10.,28.,1.,9.,0.,15.,5.,3.,15.,5.,1.,
             0.,1.,15.,6.,1.,0.,1.,15.,4.,1.,3.,
             2.,1.,3.,-1.,-1.,1.,1.,3.,73.,2.,10.,
             0.,13.,14.,12.,13.,14.,1.,0.,1.,14.,14.,
             1.,0.,1.,14.,13.,1.,0.,1.,13.,13.,1.,
             4.,2.,15.,14.,1.,4.,2.,15.,13.,1.,4.,
             2.,15.,12.,1.,4.,2.,14.,12.,1.,4.,2.,
             13.,12.,1.,13.,3.,15.,11.,1.,13.,3.,14.,
             11.,1.,13.,3.,13.,11.,1.,25.,4.,1.,9.,
             5.,1.,0.,1.,12.,28.,3.,9.,0.,13.,5.,
             3.,13.,5.,1.,0.,1.,13.,6.,1.,0.,1.,
             13.,4.,1.,3.,2.,1.,1.,-1.,-1.,1.,1.,
             3.,18.,4.,-1.,0.,10.,15.,1.,10.,15.,1.,
             -3.,5.,1.,-1.,3.,5.,0.,8.,1.,23.,5.,
             -1.,0.,10.,10.,2.,10.,10.,1.,-6.,1.,11.,
             11.,1.,-4.,2.,1.,-1.,2.,3.,0.,7.,2.,
             18.,6.,-1.,0.,7.,4.,1.,7.,4.,1.,1.,
             5.,1.,-1.,2.,2.,0.,4.,1.,18.,7.,-1.,
             0.,5.,11.,1.,5.,11.,1.,4.,5.,1.,-1.,
             3.,4.,0.,1.,1.,43.,8.,-1.,0.,2.,14.,
             6.,2.,14.,1.,2.,3.,3.,14.,1.,2.,3.,
             1.,14.,1.,2.,3.,3.,13.,1.,2.,3.,2.,
             13.,1.,2.,3.,1.,13.,1.,14.,5.,1.,-1.,
             2.,2.,0.,1.,6.,100.,9.,10.,2.,1.,3.,
             15.,5.,17.,15.,5.,1.,0.,2.,15.,6.,1.,
             0.,2.,14.,6.,1.,0.,2.,14.,5.,1.,0.,
             2.,14.,4.,1.,0.,2.,14.,7.,1.,0.,2.,
             15.,4.,1.,0.,2.,13.,6.,1.,0.,2.,13.,
             5.,1.,0.,2.,13.,4.,1.,0.,2.,15.,3.,
             1.,0.,2.,14.,3.,1.,0.,2.,13.,3.,1.,
             0.,2.,14.,8.,1.,14.,3.,15.,2.,1.,14.,
             3.,14.,2.,1.,14.,3.,13.,2.,1.,31.,4.,
             1.,2.,-1.,-1.,1.,2.,17.,205.,10.,-1.,2.,
             2.,9.,13.,14.,38.,13.,14.,1.,0.,4.,14.,
             15.,1.,0.,4.,13.,15.,1.,0.,4.,15.,1.,
             1.,0.,4.,14.,1.,1.,0.,4.,13.,1.,1.,
             0.,4.,15.,15.,1.,0.,4.,14.,14.,1.,0.,
             4.,14.,13.,1.,0.,4.,13.,13.,1.,0.,4.,
             15.,14.,1.,0.,4.,15.,2.,1.,0.,4.,14.,
             2.,1.,0.,4.,13.,2.,1.,0.,4.,15.,13.,
             1.,0.,4.,15.,12.,1.,0.,4.,14.,12.,1.,
             0.,4.,13.,12.,1.,0.,4.,15.,3.,1.,0.,
             4.,14.,3.,1.,0.,4.,13.,3.,1.,0.,4.,
             15.,11.,1.,0.,4.,14.,11.,1.,0.,4.,13.,
             11.,1.,0.,4.,15.,4.,1.,0.,4.,14.,4.,
             1.,0.,4.,13.,4.,1.,0.,4.,15.,10.,1.,
             0.,4.,14.,10.,1.,0.,4.,13.,10.,1.,0.,
             4.,15.,5.,1.,0.,4.,14.,5.,1.,0.,4.,
             13.,5.,1.,0.,4.,15.,6.,1.,0.,4.,14.,
             6.,1.,0.,4.,13.,6.,1.,0.,4.,14.,7.,
             1.,0.,4.,14.,8.,1.,76.,6.,1.,-1.,4.,
             1.,0.,4.,38.],dtype=np.float64)
        lakes = cclc.get_lake_parameters_from_array(lakes_as_array,scale_factor=3)
        cumulative_flow  = np.asarray([[2,1,1,1,3],
                                       [1,3,4,1,1],
                                       [1,8,3,2,1],
                                       [1,1,4,1,1],
                                       [1,3,1,1,2]],
                                       dtype=np.int32, order='C')
        reconnected_cumulative_flow_expected_out  = np.asarray([[2,1,1,1,3],
                                                                [1,21,23,1,1],
                                                                [1,15,5,4,2],
                                                                [6,1,6,1,1],
                                                                [1,3,1,1,2]],
                                                                dtype=np.int32,
                                                                order='C')
        connected_catchments,reconnected_cumulative_flow = \
            cclc.connect_coarse_lake_catchments(lakes,
                                                field.Field(coarse_catchments,
                                                            grid="LatLong",nlat=5,nlong=5),
                                                field.Field(basin_catchment_numbers,
                                                            grid="LatLong",nlat=15,nlong=15),
                                                field.Field(river_directions,
                                                            grid="LatLong",nlat=5,nlong=5),
                                                scale_factor=3,
                                                correct_cumulative_flow=True,
                                                cumulative_flow=
                                                field.Field(cumulative_flow,
                                                            grid="LatLong",nlat=5,nlong=5))
        np.testing.assert_array_equal(connected_catchments.get_data(),
                                      expected_connected_catchments)
        np.testing.assert_array_equal(reconnected_cumulative_flow.get_data(),
                                      reconnected_cumulative_flow_expected_out)

    def testConnectCoarseLakeCatchmentsTwo(self):
        expected_connected_coarse_catchments =  \
                             np.asarray([[ 2, 2, 1, 1, 1, 1, 1, 1],
                                         [ 1, 1, 1, 1, 1, 1, 1, 1],
                                         [ 1, 1, 1, 1, 1, 1, 1, 1],
                                         [ 1, 1, 1, 1, 1, 1, 1, 1]],
                                         dtype=np.int32, order='C')
        coarse_catchments =  np.asarray([[ 153, 153,  44,  44,   44,  44,  44,  44],
                                         [  44,  44,  44,  44,   44,  44,  44,3303],
                                         [  44,  44,  44,  44,   44,  44,  44,3303],
                                         [4440,4440,4440, 4440,9906,9906,  44,  44]],
                                         dtype=np.int32, order='C')
        basin_catchment_numbers = np.asarray([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1,  2, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1,  2, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1,  2, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1,  2, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  0,  0,  0,  0,  0,
              0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1]],
            dtype=np.int32, order='C')
        river_directions =  np.asarray([
            [ 0, -2, 2, 4, 4, 4, 4, 4],
            [ 0,  4, 4, 4, 4, 4, 4, -2],
            [ 8,  8, 4, 4, 4, 4, 4,  8],
            [ 6, -2, 4, 6, -2, 4, 8, 4]],
         dtype=np.int32, order='C')
        lakes_as_array = np.asarray([4.,43.,1.,-1.,0.,11.,14.,6.,11.,14.,1.,0.,1.,
                                     11.,15.,1.,0.,1.,11.,13.,1.,0.,1.,11.,16.,1.,
                                     0.,1.,11.,17.,1.,5.,2.,11.,12.,1.,11.,3.,1.,
                                     -1.,3.,6.,0.,1.,6.,58.,2.,-1.,0.,11.,4.,9.,
                                     11.,4.,1.,1.,2.,11.,5.,1.,1.,2.,11.,3.,1.,
                                     1.,2.,11.,6.,1.,1.,2.,11.,7.,1.,1.,2.,11.,
                                     8.,1.,1.,2.,11.,9.,1.,8.,3.,11.,2.,1.,16.,
                                     4.,11.,10.,1.,34.,6.,1.,1.,-1.,-1.,1.,1.,9.,
                                     33.,3.,-1.,0.,5.,23.,4.,5.,23.,1.,1.,2.,6.,
                                     23.,1.,1.,2.,7.,23.,1.,1.,2.,8.,23.,1.,9.,
                                     4.,1.,-1.,2.,7.,0.,1.,4.,18.,4.,-1.,0.,2.,
                                     5.,1.,2.,5.,1.,1.,2.,1.,-1.,1.,1.,0.,1.,
                                     1.])
        lakes = cclc.get_lake_parameters_from_array(lakes_as_array,scale_factor=3)
        cumulative_flow  = np.asarray([
            [ 1,  1, 6, 5, 4, 3, 2, 1],
            [ 22, 20,11, 4, 3, 2, 1, 2],
            [ 1,  8, 7, 6, 5, 4, 3,  1],
            [ 1,  3, 1, 1, 3, 1, 2, 1]],
            dtype=np.int32, order='C')
        reconnected_cumulative_flow_expected_out  = np.asarray([
            [ 1,  1, 6, 5, 4, 3, 2, 1],
            [ 30, 28,13, 6, 5, 4, 3, 2],
            [ 1,  14, 13, 12, 11, 10, 3,  1],
            [ 1,  3, 1, 1, 3, 1, 2, 1]],
            dtype=np.int32, order='C')
        connected_catchments,reconnected_cumulative_flow = \
            cclc.connect_coarse_lake_catchments(lakes,
                                                field.Field(coarse_catchments,
                                                            grid="LatLong",nlat=4,nlong=8),
                                                field.Field(basin_catchment_numbers,
                                                            grid="LatLong",nlat=12,nlong=24),
                                                field.Field(river_directions,
                                                            grid="LatLong",nlat=4,nlong=24),
                                                scale_factor=3,
                                                correct_cumulative_flow=True,
                                                cumulative_flow=
                                                field.Field(cumulative_flow,
                                                            grid="LatLong",nlat=5,nlong=5))
        np.testing.assert_array_equal(connected_catchments.get_data(),
                                      expected_connected_coarse_catchments)
        np.testing.assert_array_equal(reconnected_cumulative_flow.get_data(),
                                      reconnected_cumulative_flow_expected_out)

if __name__ == "__main__":
    unittest.main()
