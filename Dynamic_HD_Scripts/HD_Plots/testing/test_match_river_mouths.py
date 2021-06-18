'''
Unit tests for the match_river_mouth module.
Created on Apr 24, 2016

@author: thomasriddick
'''

import unittest
import numpy as np

from HD_Plots.utilities import match_river_mouths as mtch_rm

class ConflictResolverTests(unittest.TestCase):

    params = mtch_rm.Params('testing')

    allowed_pairing = (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,7,params))
    disallowed_pairing = (mtch_rm.RiverMouth(1,1,1,5,params),mtch_rm.RiverMouth(1,1,1,2,params))
    test_configuration = [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                          (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                          (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                          (mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,4,params))]

    test_possible_pairings_for_all_idnums_two_possiblities = [[(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                                               (mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,2,params))],
                                                              [(mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                                               (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,2,params))]]

    two_possibilities_expected_results = [[(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params)),None],
                                          [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,2,params)),None],
                                          [None,None],
                                          [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                           (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,2,params))],
                                          [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                           (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,1,params))],
                                          [None,(mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,1,params))],
                                          [None,(mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,2,params))]]

    test_possible_pairings_for_all_idnums_four_possiblities = [[(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,4,params))],
                                                               [(mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,4,params))],
                                                               [(mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,4,params))],
                                                               [(mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                                                (mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,4,params))]]

    test_conflict = [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                     (mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                     (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                     (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,2,params))]

    test_conflict_to_resolve = [(mtch_rm.RiverMouth(2,3,701,1,params),mtch_rm.RiverMouth(1,1,702,1,params)),
                                (mtch_rm.RiverMouth(2,3,701,1,params),mtch_rm.RiverMouth(101,201,100,2,params)),
                                (mtch_rm.RiverMouth(100,200,99,2,params),mtch_rm.RiverMouth(1,1,702,1,params)),
                                (mtch_rm.RiverMouth(100,200,99,2,params),mtch_rm.RiverMouth(101,201,100,2,params))]

    test_conflicts_to_resolve = [[(mtch_rm.RiverMouth(2,3,701,1,params),mtch_rm.RiverMouth(1,1,702,1,params)),
                                  (mtch_rm.RiverMouth(2,3,701,1,params),mtch_rm.RiverMouth(101,201,100,2,params)),
                                  (mtch_rm.RiverMouth(100,200,99,2,params),mtch_rm.RiverMouth(1,1,702,1,params)),
                                  (mtch_rm.RiverMouth(100,200,99,2,params),mtch_rm.RiverMouth(101,201,100,2,params))],
                                 [(mtch_rm.RiverMouth(2,3,701,3,params),mtch_rm.RiverMouth(1,1,702,3,params)),
                                  (mtch_rm.RiverMouth(2,3,701,3,params),mtch_rm.RiverMouth(101,201,100,4,params)),
                                  (mtch_rm.RiverMouth(100,200,99,4,params),mtch_rm.RiverMouth(1,1,702,3,params)),
                                  (mtch_rm.RiverMouth(100,200,99,4,params),mtch_rm.RiverMouth(101,201,100,4,params))],
                                 [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]]

    test_conflict_resolution_expected_results = [(mtch_rm.RiverMouth(2,3,701,1,params),mtch_rm.RiverMouth(1,1,702,1,params)),
                                                 (mtch_rm.RiverMouth(100,200,99,2,params),mtch_rm.RiverMouth(101,201,100,2,params))]

    resolve_conflicts_pairs_expected_results = [(mtch_rm.RiverMouth(2,3,701,1,params),mtch_rm.RiverMouth(1,1,702,1,params)),
                                                (mtch_rm.RiverMouth(100,200,99,2,params),mtch_rm.RiverMouth(101,201,100,2,params)),
                                                (mtch_rm.RiverMouth(2,3,701,3,params),mtch_rm.RiverMouth(1,1,702,3,params)),
                                                (mtch_rm.RiverMouth(100,200,99,4,params),mtch_rm.RiverMouth(101,201,100,4,params))]

    test_configuration =  [(mtch_rm.RiverMouth(13,-5,135,1,params),mtch_rm.RiverMouth(3,22,133,1,params)),
                           (mtch_rm.RiverMouth(102,109,340,2,params),mtch_rm.RiverMouth(104,123,280,2,params)),
                           None,None]

    def testIsPairingAllowedInConfiguration(self):
        self.assertTrue(mtch_rm.ConflictResolver.is_pairing_allowed_in_configuration(self.allowed_pairing,
                                                                                     self.test_configuration))
        self.assertFalse(mtch_rm.ConflictResolver.is_pairing_allowed_in_configuration(self.disallowed_pairing,
                                                                                      self.test_configuration))

    def testAddToAllowedConfigurationsTwoPoints(self):
        result = mtch_rm.ConflictResolver.\
                add_to_allowed_configurations([],0,self.test_possible_pairings_for_all_idnums_two_possiblities)
        self.assertEqual(7, len(result), "Not producing correct set of possible configurations for"
                                         " the case of two pairs of river mouths")
        for configuration,expected_configuration in zip(result,self.two_possibilities_expected_results):
            for item,expectation in zip(configuration,expected_configuration):
                if item is None or expectation is None:
                    self.assertIs(item,None)
                else:
                    self.assertEqual(item[0].get_idnum(),expectation[0].get_idnum(),
                                     "Not producing correct set of possible configurations for"
                                     " the case of two pairs of river mouths. \nProblem is in {0}\n"
                                     "Which should be: {1}\n"
                                     "Configuration: {2}".format(item,expectation,configuration))
                    self.assertEqual(item[1].get_idnum(),expectation[1].get_idnum(),
                                     "Not producing correct set of possible configurations for"
                                     " the case of two pairs of river mouths. \nProblem is in {0}\n"
                                     "Which should be: {1}\n:"
                                     "Configuration: {2}".format(item,expectation,configuration))

    def testAddToAllowedConfigurationsFourPoints(self):
        result = mtch_rm.ConflictResolver.\
                add_to_allowed_configurations([],0,self.test_possible_pairings_for_all_idnums_four_possiblities)
        self.assertEqual(209, len(result), "Not producing correct set of possible configurations for"
                                           " the case of four pairs of river mouths")

    def testGeneratePossibleInconflictPairings(self):
        result = mtch_rm.ConflictResolver.generate_possible_inconflict_pairings(self.test_conflict)
        for configuration,expected_configuration in zip(result,
                                                        self.two_possibilities_expected_results):
            for item,expectation in zip(configuration,expected_configuration):
                if item is None or expectation is None:
                    self.assertIs(item,None)
                else:
                    self.assertEqual(item[0].get_idnum(),expectation[0].get_idnum(),
                                     "Not producing correct set of possible configurations for"
                                     " the case of two pairs of river mouths. \nProblem is in {0}\n"
                                     "Which should be: {1}\n"
                                     "Configuration: {2}".format(item,expectation,configuration))
                    self.assertEqual(item[1].get_idnum(),expectation[1].get_idnum(),
                                     "Not producing correct set of possible configurations for"
                                     " the case of two pairs of river mouths. \nProblem is in {0}\n"
                                     "Which should be: {1}\n:"
                                     "Configuration: {2}".format(item,expectation,configuration))

    def testEvaulateConfiguration(self):
        score = mtch_rm.ConflictResolver.evaulate_configuration(self.test_configuration, self.params)
        self.assertAlmostEqual(score,226.692503611,places=7,msg="Not producing correct configuration"
                                                                " evaulation in test")

    def testResolveConflict(self):
        resolved_pairs = mtch_rm.ConflictResolver.\
            resolve_conflict(self.test_conflict_to_resolve,self.params)
        for item,expectation in zip(resolved_pairs,self.test_conflict_resolution_expected_results):
            self.assertEqual(item[0].get_idnum(),expectation[0].get_idnum(),
                             "Conflict resolution test not producing expected results")
            self.assertEqual(item[1].get_idnum(),expectation[1].get_idnum(),
                             "Conflict resolution test not producing expected results")

    def testResolveConflicts(self):
        resolved_pairs,unresolved_pairs = mtch_rm.ConflictResolver.\
            resolve_conflicts(self.test_conflicts_to_resolve,self.params)
        for pair,expectation in zip(resolved_pairs,self.resolve_conflicts_pairs_expected_results):
            self.assertEqual(pair[0].get_idnum(),expectation[0].get_idnum(),
                             "Resolve conflicts not producing expected results")
            self.assertEqual(pair[1].get_idnum(),expectation[1].get_idnum(),
                             "Resolve conflicts not producing expected results")
        self.assertEqual(len(unresolved_pairs), 1,"Not correct returning unresolved pairs from "
                                                  "conflict resolution")

class MatchingRiverMouthTests(unittest.TestCase):

    params = mtch_rm.Params('testing')

    river_mouth_outflow_field = np.array([[0,  0,1353,   0,   0],
                                          [1,999,2354,   0,   0],
                                          [0,  0,3450, 450,   0],
                                          [0,  0,  21,1001,1000]])

    test_ref_rmouth_field = np.array([[0,   0, 0, 0,   0],
                                      [0,1120, 0, 0,   0],
                                      [0,   0, 0, 0,   0],
                                      [0,   0, 0, 0,1450]])

    test_data_rmouth_field = np.array([[0,   0, 0,   0, 0],
                                       [1125,0, 0,   0, 0],
                                       [0,   0, 0,1430, 0],
                                       [0,   0, 0,   0, 0]])

    river_mouth_outflow_expected_results = [mtch_rm.RiverMouth(0,2,1353,0,params),
                                            mtch_rm.RiverMouth(1,2,2354,1,params),
                                            mtch_rm.RiverMouth(2,2,3450,2,params),
                                            mtch_rm.RiverMouth(3,3,1001,3,params),
                                            mtch_rm.RiverMouth(3,4,1000,4,params)]

    river_mouth_point_outflows_ref = [mtch_rm.RiverMouth(1,2,2354,3,params),
                                      mtch_rm.RiverMouth(2,2,17250,1,params),
                                      mtch_rm.RiverMouth(106,86,1001,2,params),
                                      mtch_rm.RiverMouth(3,4,1000,0,params)]

    river_mouth_point_outflows_data = [mtch_rm.RiverMouth(4,14,1000,0,params),
                                       mtch_rm.RiverMouth(2,2,17200,2,params),
                                       mtch_rm.RiverMouth(3,10,277,4,params),
                                       mtch_rm.RiverMouth(1,2,2354,3,params),
                                       mtch_rm.RiverMouth(103,83,1200,1,params)]

    expected_candidate_pairs = [(mtch_rm.RiverMouth(2,2,17250,1,params),mtch_rm.RiverMouth(2,2,17200,2,params)),
                                (mtch_rm.RiverMouth(3,4,1000,0,params),mtch_rm.RiverMouth(3,10,277,4,params)),
                                (mtch_rm.RiverMouth(1,2,2354,3,params),mtch_rm.RiverMouth(1,2,2354,3,params)),
                                (mtch_rm.RiverMouth(3,4,1000,0,params),mtch_rm.RiverMouth(1,2,2354,3,params)),
                                (mtch_rm.RiverMouth(106,86,1001,2,params),mtch_rm.RiverMouth(103,83,1200,1,params))]

    expected_matches = [(mtch_rm.RiverMouth(1,1,1120,0,params),mtch_rm.RiverMouth(1,0,1125,0,params)),
                        (mtch_rm.RiverMouth(3,4,1450,1,params),mtch_rm.RiverMouth(2,3,1430,1,params))]

    def testFindRiverMouthPoints(self):
        found_river_mouths = mtch_rm.find_rivermouth_points(self.river_mouth_outflow_field,self.params)
        self.assertListEqual(found_river_mouths,self.river_mouth_outflow_expected_results)

    def testGeneratingCandidatePairs(self):
        candidiate_pairs = mtch_rm.generate_candidate_pairs(self.river_mouth_point_outflows_ref,
                                                           self.river_mouth_point_outflows_data)
        for pair,expected_pair in zip(candidiate_pairs,self.expected_candidate_pairs):
            self.assertTupleEqual(pair,expected_pair)

    def testGenerateMatches(self):
        matches,conflicts = mtch_rm.generate_matches(self.test_ref_rmouth_field,
                                                     self.test_data_rmouth_field,
                                                     self.params)
        for match,expected_match in zip(matches,self.expected_matches):
            self.assertEqual(match[0].get_idnum(),expected_match[0].get_idnum(),
                             "Generating matches from two field is not producing"
                             " expected results")
            self.assertEqual(match[1].get_idnum(),expected_match[1].get_idnum(),
                             "Generating matches from two field is not producing"
                             " expected results")
        self.assertEqual(len(conflicts),0)

class ConflictCheckerTests(unittest.TestCase):

    params = mtch_rm.Params('testing')

    conflict_association_test_data = [[(mtch_rm.RiverMouth(1,1,1,11,params),mtch_rm.RiverMouth(1,1,1,10,params)),
                                       (mtch_rm.RiverMouth(1,1,1,11,params),mtch_rm.RiverMouth(1,1,1,11,params))],
                                      [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params))],
                                      [(mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,4,params))],
                                      [(mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                       (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                       (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,3,params))],
                                      [(mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                       (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                       (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,4,params))],
                                      [(mtch_rm.RiverMouth(1,1,1,9,params),mtch_rm.RiverMouth(1,1,1,1,params))],
                                      [(mtch_rm.RiverMouth(1,1,1,8,params),mtch_rm.RiverMouth(1,1,1,9,params)),
                                       (mtch_rm.RiverMouth(1,1,1,8,params),mtch_rm.RiverMouth(1,1,1,6,params))],
                                      [(mtch_rm.RiverMouth(1,1,1,10,params),mtch_rm.RiverMouth(1,1,1,9,params))],
                                      [(mtch_rm.RiverMouth(1,1,1,5,params),mtch_rm.RiverMouth(1,1,1,5,params))],
                                      [(mtch_rm.RiverMouth(1,1,1,6,params),mtch_rm.RiverMouth(1,1,1,5,params))],
                                      [(mtch_rm.RiverMouth(1,1,1,7,params),mtch_rm.RiverMouth(1,1,1,6,params))]]

    associated_pairs_expected_results = [[(mtch_rm.RiverMouth(1,1,1,11,params),mtch_rm.RiverMouth(1,1,1,10,params)),
                                          (mtch_rm.RiverMouth(1,1,1,11,params),mtch_rm.RiverMouth(1,1,1,11,params))],
                                         [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                          (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                          (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                          (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                          (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                          (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                          (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,4,params)),
                                          (mtch_rm.RiverMouth(1,1,1,9,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                          (mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,4,params))],
                                         [(mtch_rm.RiverMouth(1,1,1,8,params),mtch_rm.RiverMouth(1,1,1,9,params)),
                                          (mtch_rm.RiverMouth(1,1,1,8,params),mtch_rm.RiverMouth(1,1,1,6,params)),
                                          (mtch_rm.RiverMouth(1,1,1,10,params),mtch_rm.RiverMouth(1,1,1,9,params)),
                                          (mtch_rm.RiverMouth(1,1,1,7,params),mtch_rm.RiverMouth(1,1,1,6,params))],
                                         [(mtch_rm.RiverMouth(1,1,1,5,params),mtch_rm.RiverMouth(1,1,1,5,params)),
                                          (mtch_rm.RiverMouth(1,1,1,6,params),mtch_rm.RiverMouth(1,1,1,5,params))]]

    conflict_check_pseudo_data = [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                  (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                  (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                  (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,4,params)),
                                  (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,5,params)),
                                  (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,6,params)),
                                  (mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,7,params)),
                                  (mtch_rm.RiverMouth(1,1,1,5,params),mtch_rm.RiverMouth(1,1,1,7,params)),
                                  (mtch_rm.RiverMouth(1,1,1,6,params),mtch_rm.RiverMouth(1,1,1,7,params)),
                                  (mtch_rm.RiverMouth(1,1,1,7,params),mtch_rm.RiverMouth(1,1,1,8,params)),
                                  (mtch_rm.RiverMouth(1,1,1,8,params),mtch_rm.RiverMouth(1,1,1,8,params)),
                                  (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,9,params))]

    conflict_check_expected_conflicts_index_zero = [[(mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                                     (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                                     (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,9,params))],
                                                    [(mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,4,params)),
                                                     (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,5,params)),
                                                     (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,6,params))]]

    conflict_check_expected_unconflicted_pairs_index_zero = [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                                             (mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,7,params)),
                                                             (mtch_rm.RiverMouth(1,1,1,5,params),mtch_rm.RiverMouth(1,1,1,7,params)),
                                                             (mtch_rm.RiverMouth(1,1,1,6,params),mtch_rm.RiverMouth(1,1,1,7,params)),
                                                             (mtch_rm.RiverMouth(1,1,1,7,params),mtch_rm.RiverMouth(1,1,1,8,params)),
                                                             (mtch_rm.RiverMouth(1,1,1,8,params),mtch_rm.RiverMouth(1,1,1,8,params))]

    conflict_check_expected_conflicts_index_one = [[(mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,7,params)),
                                                    (mtch_rm.RiverMouth(1,1,1,5,params),mtch_rm.RiverMouth(1,1,1,7,params)),
                                                    (mtch_rm.RiverMouth(1,1,1,6,params),mtch_rm.RiverMouth(1,1,1,7,params))],
                                                   [(mtch_rm.RiverMouth(1,1,1,7,params),mtch_rm.RiverMouth(1,1,1,8,params)),
                                                    (mtch_rm.RiverMouth(1,1,1,8,params),mtch_rm.RiverMouth(1,1,1,8,params))]]

    conflict_check_expected_unconflicted_pairs_index_one = [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params)),
                                                            (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                                            (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                                            (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,4,params)),
                                                            (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,5,params)),
                                                            (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,6,params)),
                                                            (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,9,params))]

    conflict_check_expected_unconflicted_pairs_top_level = [(mtch_rm.RiverMouth(1,1,1,1,params),mtch_rm.RiverMouth(1,1,1,1,params))]

    conflict_check_expected_conflicts_top_level = [[(mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,2,params)),
                                                    (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,3,params)),
                                                    (mtch_rm.RiverMouth(1,1,1,2,params),mtch_rm.RiverMouth(1,1,1,9,params))],
                                                   [(mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,4,params)),
                                                    (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,5,params)),
                                                    (mtch_rm.RiverMouth(1,1,1,3,params),mtch_rm.RiverMouth(1,1,1,6,params))],
                                                   [(mtch_rm.RiverMouth(1,1,1,4,params),mtch_rm.RiverMouth(1,1,1,7,params)),
                                                    (mtch_rm.RiverMouth(1,1,1,5,params),mtch_rm.RiverMouth(1,1,1,7,params)),
                                                    (mtch_rm.RiverMouth(1,1,1,6,params),mtch_rm.RiverMouth(1,1,1,7,params))],
                                                   [(mtch_rm.RiverMouth(1,1,1,7,params),mtch_rm.RiverMouth(1,1,1,8,params)),
                                                    (mtch_rm.RiverMouth(1,1,1,8,params),mtch_rm.RiverMouth(1,1,1,8,params))]]

    def testAssociateConflicts(self):
        associated_pairs = mtch_rm.ConflictChecker.associate_conflicts(self.conflict_association_test_data,0)
        for conflict,expected_conflict in zip(associated_pairs,
                                              self.associated_pairs_expected_results):
            self.assertEqual(len(conflict),len(expected_conflict))
            for pair,expected_pair in zip(conflict,expected_conflict):
                self.assertTupleEqual(pair,expected_pair)

    def testCheckPairSetForConflicts(self):
        #test index 0
        conflicts,unconflicted_pairs = mtch_rm.ConflictChecker. \
            check_pair_set_for_conflicts(self.conflict_check_pseudo_data,0)
        for conflict,expected_conflict in zip(conflicts,
                                              self.conflict_check_expected_conflicts_index_zero):
            self.assertEqual(len(conflict),len(expected_conflict))
            for pair,expected_pair in zip(conflict,expected_conflict):
                self.assertTupleEqual(pair,expected_pair)
        self.assertEqual(len(unconflicted_pairs),
                         len(self.conflict_check_expected_unconflicted_pairs_index_zero))
        for unconflicted_pair,expected_unconflicted_pair in \
            zip(unconflicted_pairs,
                self.conflict_check_expected_unconflicted_pairs_index_zero):
            self.assertTupleEqual(unconflicted_pair,expected_unconflicted_pair)
        #test index 1
        conflicts,unconflicted_pairs = mtch_rm.ConflictChecker. \
            check_pair_set_for_conflicts(self.conflict_check_pseudo_data,1)
        for conflict,expected_conflict in zip(conflicts,
                                              self.conflict_check_expected_conflicts_index_one):
            self.assertEqual(len(conflict),len(expected_conflict))
            for pair,expected_pair in zip(conflict,expected_conflict):
                self.assertTupleEqual(pair,expected_pair)
        self.assertEqual(len(unconflicted_pairs),
                         len(self.conflict_check_expected_unconflicted_pairs_index_one))
        for unconflicted_pair,expected_unconflicted_pair in \
            zip(unconflicted_pairs,
                self.conflict_check_expected_unconflicted_pairs_index_one):
                self.assertTupleEqual(unconflicted_pair,expected_unconflicted_pair)

    def testCheckPairSetsForConflicts(self):
        conflicts,unconflicted_pairs = mtch_rm.ConflictChecker. \
            check_pair_sets_for_conflicts(self.conflict_check_pseudo_data)
        self.assertEqual(len(conflicts),len(self.conflict_check_expected_conflicts_top_level))
        for conflict,expected_conflict in zip(conflicts,
                                              self.conflict_check_expected_conflicts_top_level):
            self.assertEqual(len(conflict),len(expected_conflict))
            for pair,expected_pair in zip(conflict,expected_conflict):
                self.assertTupleEqual(pair,expected_pair)
        self.assertEqual(len(unconflicted_pairs),
                         len(self.conflict_check_expected_unconflicted_pairs_top_level))
        for pair,expected_pair in zip(unconflicted_pairs,
                                      self.conflict_check_expected_unconflicted_pairs_top_level):
            self.assertTupleEqual(pair,expected_pair)

if __name__ == "__main__":
    unittest.main()
