import random
from copy import deepcopy
from operator import attrgetter

from deap import tools
from gpmallt.rundata_lo import rd
from gptools.array_wrapper import ArrayWrapper
from gptools.gp_util import output_ind, check_uniqueness_output, \
    get_arrays_cached, try_cache

LOG_GENS = 100

def ea(population, toolbox, cxpb, mutpb, elitism, ngen, eval_func, stats=None,
       halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(1, ngen + 1):
        evaled = eval_population_cached_simple(population, toolbox, eval_func, rd)
        sorted_elite = sorted(population, key=attrgetter("fitness"), reverse=True)
        direct_elite = deepcopy(sorted_elite[:elitism])

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(evaled), **record)
        if verbose:
            print(logbook.stream)

        if gen % LOG_GENS == 0:
            clean_up = gen % 100 == 0
            output_ind(halloffame[0], toolbox, rd, suffix="-" + str(gen), del_old=clean_up,classify=True)

        offspring = toolbox.select(deepcopy(population), len(population) - elitism)
        # Vary the pool of individuals
        offspring = varOrUniqueBatched(offspring, toolbox, len(offspring), cxpb, mutpb)

        population[:elitism] = direct_elite
        population[elitism:] = offspring

    # one last time.
    eval_population_cached_simple(population, toolbox, eval_func, rd)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=ngen, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    return population, logbook

def varOrUniqueBatched(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) == 1.0, (
        "The sum of the crossover and mutation probabilities must be equal to 1.0.")

    offspring = []
    no_change = 0
    while len(offspring) < lambda_ and no_change < 10:
        #how many pairs of candidates do we make?
        candidates = []
        num_copies_to_go = lambda_ - len(offspring)
        #makes 2*num_to_go, with a min of 8 individuals (threading).
        for i in range(max(4,num_copies_to_go)):
            op_choice = random.random()
            if op_choice < cxpb:  # Apply crossover
                ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
                ind1, ind2 = toolbox.mate(ind1, ind2)
                # just in case
                ind1.str = None
                ind2.str = None
                ind1.output = None
                ind2.output = None
                candidates.append(ind1)
                candidates.append(ind2)
            else:
                ind1 = toolbox.clone(random.choice(population))
                ind1, = toolbox.mutate(ind1)
                ind1.str = None
                ind1.output = None
                candidates.append(ind1)
                ind2 = toolbox.clone(random.choice(population))
                ind2, = toolbox.mutate(ind2)
                ind2.str = None
                ind2.output = None
                candidates.append(ind2)
        #now evaluate them in parallel!
        #it shouldn't select more than we need.
        _ = get_arrays_cached(candidates, toolbox, rd)
        num_produced = check_uniqueness_output(candidates, lambda_, offspring, rd)

        #safeguard infinite loops
        if num_produced == 0:
            no_change += 1
        else:
            no_change = 0

    if len(offspring) < lambda_:
        num_copies = lambda_ - len(offspring)
        print('Only {} offspring produced, reproducing {} individuals to get to {}.'.format(len(offspring), num_copies,
                                                                                            lambda_))
        offspring.extend(list(map(toolbox.clone, random.sample(population, num_copies))))
    assert len(offspring) == lambda_, ('Must produce exactly {} offspring, not {}.'.format(lambda_,len(offspring)))
    return offspring

def eval_population_cached_simple(population, toolbox, eval_func, rundata, cache=0):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    non_cached = []

    arrays = get_arrays_cached(invalid_ind, toolbox,rundata)

    # try and get from cache, and then only thread the leftovers
    for ind, dat_array in zip(invalid_ind, arrays):
        # dat_array = evaluateTrees(rd.data_t, toolbox, ind)
        hashable = ArrayWrapper(dat_array)
        res = try_cache(rundata, hashable, cache)
        if res:
            ind.fitness.values = res
        if not res:
            non_cached.append((ind, dat_array))

    # print("Non-cached: " + str(len(non_cached)))
    non_cached_arrays = [x[1] for x in non_cached]
    # we don't want to pass the individual between threads!
    fitnesses = toolbox.map(eval_func, non_cached_arrays)
    for non_cache, fit in zip(non_cached, fitnesses):
        non_cache[0].fitness.values = fit
        # add to cache
        if cache >= 0:
            rundata.fitnessCache[cache][ArrayWrapper(non_cache[1])] = fit
            rundata.stores = rundata.stores + 1

    return non_cached
