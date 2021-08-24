import hnswlib
from numba import guvectorize, intp, float64, njit

var_dict = {}
import numpy as np


def init_worker(data_T_flat, data_T_shape, all_orderings, all_orderings_sorted, all_orderings_argsorted):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['data_T'] = np.frombuffer(data_T_flat).reshape(data_T_shape)
    var_dict['all_orderings'] = all_orderings
    var_dict['all_orderings_sorted'] = all_orderings_sorted
    var_dict['all_orderings_argsorted'] = all_orderings_argsorted


def evaluate_trees_with_compiler_indep(compiler, individual_str):
    data_t = var_dict['data_T']
    num_instances = data_t.shape[1]
    num_trees = len(individual_str)

    # result = []
    result = np.zeros(shape=(num_trees, num_instances))

    for i, f in enumerate(individual_str):
        # Transform the tree expression in a callable function
        func = compiler(expr=f)
        comp = func(*data_t)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)

        result[i] = comp
        # result.append(comp)

    # dat_array = np.array(result).transpose()
    dat_array = result.T
    return dat_array


def nn_approx(embedding, num_neighbours):
    p = hnswlib.Index(space='l2', dim=embedding.shape[1])
    # TODO: optimise params to have reasonable recall...fair? unsupervised. ha.
    _m = 16
    if num_neighbours >= 40:
        _m = 24
    if num_neighbours >= 50:
        _m = 30
    if num_neighbours >= 100:
        _m = 48
    p.init_index(max_elements=embedding.shape[0], M=_m)
    p.add_items(embedding, num_threads=1)
    try:
        labels, distances = p.knn_query(embedding, num_neighbours + 1, num_threads=1)
    except:
        print("Error!!")
        print(embedding)
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


@njit
def sort_shared_distances(labels, distances):
    for idx in range(distances.shape[0]):
        x = distances[idx]
        run_lengths, start_poses, _ = rle(x)
        for i in range(run_lengths.size):
            run_length = run_lengths[i]
            start_pos = start_poses[i]
            if run_length > 1:
                labels[idx][start_pos:start_pos + run_length].sort()


def local_nn_cost_indep(embedding, cost_measure):
    k = var_dict['all_orderings_sorted'].shape[1]

    # labels, distances = nn_kdtree(embedding, k)
    labels, distances = nn_approx(embedding, k)

    # throw away itself.
    labels = labels[:, 1:]
    distances = distances[:, 1:]

    sort_shared_distances(labels, distances)
    # TODO: The end one could be swapped with another which is the same dist..hmm...
    # ignore the first column as it is itself?
    num_instances = embedding.shape[0]

    cost = np.inf

    if cost_measure == 'deviation':
        costs = np.zeros((num_instances, 1))
        # deviation_penalty = np.asarray(deviation_penalty).reshape(1,1)
        _single_cost_sorted_indep(var_dict['all_orderings'], var_dict['all_orderings_sorted'],
                                  var_dict['all_orderings_argsorted'],
                                  labels, costs)

        cost = costs.sum()
        Z = num_instances  # * K
        # just normalisation. worst case when no correct neighbours.
        cost = cost / Z
    elif cost_measure == 'deviation_weighted':
        costs = np.zeros((num_instances, 1))
        # deviation_penalty = np.asarray(deviation_penalty).reshape(1,1)
        _single_cost_sorted_indep_weighted(var_dict['all_orderings'], var_dict['all_orderings_sorted'],
                                           var_dict['all_orderings_argsorted'],
                                           labels, costs)

        cost = costs.sum()
        Z = num_instances  # * K
        # just normalisation. worst case when no correct neighbours.
        cost = cost / Z
    else:
        raise ValueError('{} is not recognised as a valid cost measure.'.format(cost_measure))
    return cost,


# @njit(fastmath=True)
@guvectorize([(intp[:], intp[:], intp[:], intp[:], float64[:])], '(n),(n),(n),(n),(m)', nopython=True, fastmath=True)
def _single_cost_sorted_indep(actual_nbs_unsorted, actual_nbs_sorted, actual_nbs_argsorted, embedd_nbs, inst_cost):
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


@guvectorize([(intp[:], intp[:], intp[:], intp[:], float64[:])], '(n),(n),(n),(n),(m)', nopython=True, fastmath=True)
def _single_cost_sorted_indep_weighted(actual_nbs_unsorted, actual_nbs_sorted, actual_nbs_argsorted, embedd_nbs,
                                       inst_cost):
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
    non_occuring_idxs = np.nonzero(~mask)

    costs = np.zeros_like(actual_nbs_sorted)
    # wasn't found! So cost of '1'
    for idx in non_occuring_idxs[0]:
        costs[idx] = 1
    cost_weights = np.linspace(1, 0, num_neighbours)
    # the ones that were found, how far off are they?
    norm_factor = occuring_idxs[0].size
    for idx in occuring_idxs[0]:
        actual_idx = np.where(actual_nbs_unsorted == embedd_nbs[idx])[0][0]
        idx_delta = abs(actual_idx - idx)
        costs[idx] = (idx_delta / max(num_neighbours - actual_idx, actual_idx)) / norm_factor

    sum_weighted_cost = np.sum(costs * cost_weights)
    inst_cost[0] = sum_weighted_cost
