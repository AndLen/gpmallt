import math
import signal

from deap import base, tools
from deap import creator
from deap.tools import HallOfFame
from multiprocessing.sharedctypes import RawArray
from sklearn.metrics import pairwise_distances

import gptools.weighted_generators as wg
from gpmallt import my_ea
from gpmallt.eval_independent import init_worker, local_nn_cost_indep
from gpmallt.eval_lo import evalGPMaLLO, sort_shared_distances
from gpmallt.gp_design import get_pset_weights
from gpmallt.rundata_lo import rd
from gptools.ParallelToolbox import ParallelToolbox
from gptools.gp_util import *
from gptools.multitree import *


# https://github.com/erikbern/ann-benchmarks
# https://github.com/nmslib/hnswlib
# https://github.com/nmslib/nmslib

def main():
    pop = toolbox.population(n=rd.pop_size)
    stats_cost = tools.Statistics(lambda ind: ind.fitness.values[0])
    mstats = tools.MultiStatistics(cost=stats_cost)
    mstats.register("min", np.min, axis=0)
    mstats.register("median", np.median, axis=0)
    mstats.register("max", np.max, axis=0)
    hof = HallOfFame(1)

    eval_func = partial(local_nn_cost_indep, cost_measure=rd.gpmalmo_fitness_function)

    pop, logbook = my_ea.ea(pop, toolbox, rd.cxpb, rd.mutpb, rd.elitism, rd.gens, eval_func, stats=mstats,
                            halloffame=hof, verbose=True)
    return pop, mstats, hof, logbook


def make_ind(toolbox, creator, num_trees):
    return creator.Individual([toolbox.tree() for _ in range(num_trees)])


def get_extra_args():
    # gpmallt specific
    arg_list = arg_parse_helper("-lf", "--local-fitness-function", help="GP-MalLO Fitness function to use", type=str,
                                dest="gpmalmo_fitness_function", choices=['deviation', 'deviation_weighted'],
                                default='deviation_weighted')
    arg_parse_helper("-ep", "--erc-probabilistic", help="Enables probabilistic ERCS", dest="erc_probabilistic",
                     action='store_true', arg_list=arg_list)
    arg_parse_helper("-dp", "--deviation-penalty", help="Penalty weighting when using the deviation measure",
                     dest="deviation_penalty", type=float, arg_list=arg_list)
    arg_parse_helper("-nn","--nearest-neighbours",help='Number of nearest neighbours to consider in the local cost',dest="num_neighbours",type=int,arg_list=arg_list)
    return arg_list


if __name__ == "__main__":
    arg_list = get_extra_args()
    init_data(rd, additional_arguments=arg_list)

    # GPMAL stuff.
    rd.pairwise_distances = pairwise_distances(rd.data)
    rd.ordered_neighbours = np.argsort(rd.pairwise_distances, axis=1)
    #just in case
    rd.num_neighbours = min(rd.num_instances-1,rd.num_neighbours)
    rd.neighbours = [x for x in range(1, 1 + rd.num_neighbours)]
    rd.identity_ordering = np.array([x for x in range(len(rd.neighbours))])
    rd.all_orderings = rd.ordered_neighbours[:, rd.neighbours]
    rd.ordered_distances = np.sort(rd.pairwise_distances, axis=1)[:, rd.neighbours]
    sort_shared_distances(rd.all_orderings, rd.ordered_distances)
    # pre-compute!!
    rd.all_orderings_sorted = np.sort(rd.all_orderings)
    rd.all_orderings_argsorted = np.argsort(rd.all_orderings)
    print(rd.neighbours)

    pset, weights = get_pset_weights(rd.data, rd.num_features, rd)
    rd.pset = pset
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * rd.nobj)
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)

    toolbox = ParallelToolbox()  #

    toolbox.register("expr", wg.w_genHalfAndHalf, pset=pset, weighted_terms=weights, min_=0, max_=rd.max_depth)
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual", make_ind, toolbox, creator, rd.max_trees)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("try_cache", try_cache, rd)

    toolbox.register("evaluate", evalGPMaLLO, rd, rd.data_t, toolbox)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", lim_xmate_aic)

    toolbox.register("expr_mut", wg.w_genFull, weighted_terms=weights, min_=0, max_=rd.max_depth)

    toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut)


    assert math.isclose(rd.cxpb + rd.mutpb, 1), "Probabilities of operators should sum to ~1."

    print(rd)

    # probably the initializer stuff...?
    do_parallel = True
    if do_parallel:
        # parallelise it
        import multiprocessing

        # gotta copy arrays and such
        raw_data_T_shape = rd.data_t.shape
        raw_data_T = RawArray('d', raw_data_T_shape[0] * raw_data_T_shape[1])
        raw_data_T_np = np.frombuffer(raw_data_T, dtype=np.float64).reshape(raw_data_T_shape)
        np.copyto(raw_data_T_np, rd.data_t)
        raw_all_orderings = rd.all_orderings
        raw_all_orderings_sorted = rd.all_orderings_sorted
        raw_all_orderings_argsorted = rd.all_orderings_argsorted
        pool = multiprocessing.Pool(processes=rd.threads, initializer=init_worker, initargs=(
            raw_data_T, raw_data_T_shape, raw_all_orderings, raw_all_orderings_sorted,
            raw_all_orderings_argsorted))
        toolbox.register("map", pool.map)


        def signal_handler(sig, frame):
            print('Errored. Closing threads.')
            pool.terminate()
            pool.join()
            sys.exit(1)


        # https://stackoverflow.com/questions/2148888/python-trap-all-signals
        catchable_sigs = {signal.SIGINT}  # set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
        for sig in catchable_sigs:
            signal.signal(sig, signal_handler)

    pop, stats, hof, logbook = main()

    final_output(hof, toolbox, logbook, pop, rd,classify=True)
    if do_parallel:
        pool.terminate()
        pool.join()

