# -*- coding: utf-8 -*-
import sys
import numpy

from test_ciresan2012 import test_columns

if __name__ == '__main__':
    # example:
    # python code/4way_test_ciresan2012.py 0 ciresan2012_bs12_nw14_d1_4Layers_cc1.pkl ciresan2012_bs12_nw16_d1_4Layers_cc1.pkl ciresan2012_bs12_nw18_d1_4Layers_cc1.pkl ciresan2012_bs12_nw20_d1_4Layers_cc1.pkl
    # used to make data for a venn diagram library which expects the following
    # order of sets for combinations of 4 supplied models:
    # A       B       C       D     A_B     A_C     A_D     B_C     B_D     C_D   A_B_C   A_B_D   A_C_D    B_C_D    A_B_C_D
    combos = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [1,0,1,0],
        [1,0,0,1],
        [0,1,1,0],
        [0,1,0,1],
        [0,0,1,1],
        [1,1,1,0],
        [1,1,0,1],
        [1,0,1,1],
        [0,1,1,1],
        [1,1,1,1]
    ]
    results = []
    assert len(sys.argv) == 6
    all_models = numpy.array(sys.argv[2:])
    for combo in combos:
        models = all_models[numpy.array(combo) == 1]
        predictions, acc = test_columns(int(sys.argv[1]), models)
        results.append(acc)
    print results
