import copy
import random

from deap import gp

from gpmallt.rundata_lo import rd as rundata


def maxheight(v):
    return max(i.height for i in v)

# stolen from gp.py....because you can't pickle decorated functions.
def wrap(func, *args, **kwargs):
    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        if maxheight(ind) > rundata.max_height:
            new_inds[i] = random.choice(keep_inds)
    return new_inds


def lim_xmate_aic(ind1, ind2):
    """
    Basically, keep only changes that obey max depth constraint on a tree-wise (NOT individual-wise) level.
    :param ind1:
    :param ind2:
    :return:
    """
    keep_inds = [copy.deepcopy(ind1), copy.deepcopy(ind2)]
    new_inds = list(xmate_aic(ind1, ind2))
    for i, ind in enumerate(new_inds):
        for j, tree in enumerate(ind):
            if tree.height > rundata.max_height:
                new_inds[i][j] = keep_inds[i][j]
    return new_inds


def xmate_aic(ind1, ind2):
    min_size = min(len(ind1), len(ind2))
    for i in range(min_size):
        ind1[i], ind2[i] = gp.cxOnePoint(copy.deepcopy(ind1[i]), copy.deepcopy(ind2[i]))
    return ind1, ind2


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    ind[i1] = indx[0]
    return ind,


def lim_xmut(ind, expr):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut, ind, expr=expr)
    # print(res)
    return res


def str_ind(ind):
    return tuple(str(i) for i in ind)
