import itertools
import math

import numpy as np
from deap import gp

from gptools.gp_util import erc_array, np_protectedDiv, np_sigmoid, np_relu, np_if
from gptools.weighted_generators import RealArray

GRANULARITY = 100


def maybe_add_function(func, num_inputs, name, pset, rundata):
    # short-circuiting!
    if hasattr(rundata, 'excluded_functions') and rundata.excluded_functions and name in rundata.excluded_functions:
        print('Skipping function {} as excluded.'.format(name))
    else:
        print('Adding function {} to pset.'.format(name))
        pset.addPrimitive(func, [RealArray] * num_inputs, RealArray, name=name)


def get_pset_weights(data, num_features, rundata):
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(RealArray, num_features), RealArray, "f")
    maybe_add_function(np.add, 2, "vadd", pset, rundata)
    maybe_add_function(np.subtract, 2, "vsub", pset, rundata)
    maybe_add_function(np.multiply, 2, "vmul", pset, rundata)
    maybe_add_function(np_protectedDiv, 2, "vdiv", pset, rundata)
    maybe_add_function(np_sigmoid, 1, "sigmoid", pset, rundata)
    maybe_add_function(np_relu, 1, "relu", pset, rundata)
    maybe_add_function(np.maximum, 2, "max", pset, rundata)
    maybe_add_function(np.minimum, 2, "min", pset, rundata)
    maybe_add_function(np_if, 3, "np_if", pset, rundata)

    # deap you muppet
    pset.context["array"] = np.array

    num_ercs = math.ceil(num_features / 10)

    weights = {RealArray: []}

    # so we get as many as we do terms...
    if rundata.use_ercs:
        num_ercs = int(num_ercs)

        print("Using {:d} ERCS".format(num_ercs))
        for i in range(num_ercs):  # range(num_features):
            pset.addEphemeralConstant("rand", erc_array, RealArray)

    for t in pset.terminals[RealArray]:
        weights[RealArray].append(t)

    return pset, weights


def getNeighbourFeats(n_index, f_index, data, neighbours):
    these_neighbours = neighbours[:, n_index]
    return data[these_neighbours, f_index]
