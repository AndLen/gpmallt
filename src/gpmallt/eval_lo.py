import hnswlib
import numpy as np
from numba import float64, intp, guvectorize, njit
from scipy.spatial import KDTree

from gptools.array_wrapper import ArrayWrapper
from gptools.gp_util import evaluate_trees, add_to_string_cache, cachedError


# https://stackoverflow.com/questions/53631460/using-numpy-isin-element-wise-between-2d-and-1d-arrays
# TODO: make this not loopy...
# Return [0,1] where 1 is perfect recreation
# @njit

def local_cost_wrapper(non_cache, rd):
    return local_nn_cost(non_cache[1], rd)


num_threads = 1


def nn_kdtree(embedding, num_neighbours):
    kdt = KDTree(embedding)
    num_instances = embedding.shape[0]
    distances, labels = kdt.query(embedding, num_neighbours + 1)
    return labels, distances


def nn_approx(embedding, num_neighbours):
    p = hnswlib.Index(space='l2', dim=embedding.shape[1])
    # TODO: optimise params to have reasonable recall...fair? unsupervised. ha.
    p.init_index(max_elements=embedding.shape[0])
    p.add_items(embedding, num_threads=num_threads)
    labels, distances = p.knn_query(embedding, num_neighbours + 1, num_threads=num_threads)
    # bloody unsigned nonsense
    labels = labels.astype(int)
    return labels, distances


@njit
def rle(inarray):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)
        https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        """
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        # not string safe!
        isclose = np_isclose(ia[1:], ia[:-1])
        y = np.logical_not(isclose)
        w = y.nonzero()[0]
        i = np.append(w, np.array(n - 1))  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        # run lengths, start positions, and values.
        return (z, p, ia[i])


@njit
def _within_tol(a, b, rtol, atol):
    return np.less_equal(np.abs(a - b), atol + rtol * np.abs(b))


@njit
def np_isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    # from https://github.com/numba/numba/pull/4610/
    # Based on NumPy impl.
    # https://github.com/numpy/numpy/blob/d9b1e32cb8ef90d6b4a47853241db2a28146a57d/numpy/core/numeric.py#L2180-L2292

    xfin = np.asarray(np.isfinite(a))
    yfin = np.asarray(np.isfinite(b))
    if np.all(xfin) and np.all(yfin):
        return _within_tol(a, b, rtol, atol)
    else:
        r = _within_tol(a, b, rtol, atol)
        if equal_nan:
            return r | (~xfin & ~yfin)
        else:
            return r


@njit(fastmath=True)
def sort_shared_distances(labels, distances):
    for idx in range(distances.shape[0]):
        x = distances[idx]
        run_lengths, start_poses, _ = rle(x)
        for i in range(run_lengths.size):
            run_length = run_lengths[i]
            start_pos = start_poses[i]
            if run_length > 1:
                labels[idx][start_pos:start_pos + run_length].sort()


def local_nn_cost(embedding, rd):
    k = rd.all_orderings_sorted.shape[1]

    # labels, distances = nn_kdtree(embedding, k)
    labels, distances = nn_approx(embedding, k)

    # throw away itself.
    labels = labels[:, 1:]
    distances = distances[:, 1:]

    sort_shared_distances(labels, distances)
    # TODO: The end one could be swapped with another which is the same dist..hmm...
    # ignore the first column as it is itself?
    num_instances = embedding.shape[0]

    costs = np.zeros((num_instances))

    # for i in range(len(labels)):
    #     _single_cost_sorted(rundata.all_orderings[i], rundata.all_orderings_sorted[i], rundata.all_orderings_argsorted[i],
    #                         labels[i], costs[i])
    _single_cost_sorted(rd.all_orderings, rd.all_orderings_sorted, rd.all_orderings_argsorted,
                        labels, costs)

    cost = costs.sum()
    Z = num_instances  # * k
    # just normalisation. worst case when no correct neighbours.
    normalised = cost / Z
    return normalised,


# @njit(fastmath=True)
@guvectorize([(intp[:], intp[:], intp[:], intp[:], float64[:])], '(n),(n),(n),(n)->()', nopython=True, fastmath=True)
def _single_cost_sorted(actual_nbs_unsorted, actual_nbs_sorted, actual_nbs_argsorted, embedd_nbs, inst_cost):
    num_neighbours = actual_nbs_sorted.shape[0]
    # inst_cost = 0.
    # ones that are actually there
    sorted_index = np.searchsorted(actual_nbs_sorted, embedd_nbs)
    # since can't numba with mode="clip"
    sorted_index[sorted_index == num_neighbours] = num_neighbours - 1
    yindex = np.take(actual_nbs_argsorted, sorted_index)
    # yindex = np.take(actual_nbs_argsorted, sorted_index, mode="clip")
    # print(yindex)
    mask = actual_nbs_unsorted[yindex] == embedd_nbs
    occuring_idxs = np.nonzero(mask)
    # TODO: remove once dev'ed
    # assert np.equal(occuring_idxs,np.nonzero(np.isin(embedd_nbs, actual_nbs_sorted))).all()
    # wasn't found! So cost of '1'
    inst_cost[0] += (num_neighbours - occuring_idxs[0].shape[0])
    # the ones that were found, how far off are they?
    occuring_cost = 0.
    for idx in occuring_idxs[0]:
        actual_idx = np.where(actual_nbs_unsorted == embedd_nbs[idx])[0][0]
        idx_delta = abs(actual_idx - idx)
        occuring_cost += idx_delta / max(num_neighbours - actual_idx, actual_idx)
    if occuring_cost > 0:
        occuring_cost /= occuring_idxs[0].size
    assert occuring_cost <= 1
    inst_cost += occuring_cost
    # return inst_cost


def approx_local_nn_accuracy(embedding, rd):
    return 1 - local_nn_cost(embedding, rd)  # approx_local_nn_cost(orderings,embedding)


def evalGPMaLLO(rd, data_t, toolbox, individual):
    add_to_string_cache(individual)
    dat_array = evaluate_trees(data_t, toolbox, individual)

    hashable = ArrayWrapper(dat_array)
    args = (dat_array, rd)
    local_cost = cachedError(hashable, local_nn_cost, rd, args=args, kargs={}, index=0)

    return local_cost
