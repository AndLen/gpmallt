import argparse
import gzip as gz
import json
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special._ufuncs import expit
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from gpmallt.eval_independent import evaluate_trees_with_compiler_indep
from gptools.array_wrapper import ArrayWrapper
from gptools.multitree import str_ind
from gptools.read_data import read_data


def np_protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def np_sigmoid(gamma):
    return expit(gamma)

def np_relu(x):
    return x * (x > 0)


def np_if(a, b, c):
    return np.where(a < 0, b, c)


def erc_array():
    return random.uniform(-1, 1)

string_cache = set()


def add_to_string_cache(ind):
    hash = str_ind(ind)
    string_cache.add(hash)
    ind.str = hash


def check_uniqueness_output(inds, num_to_produce, offspring, rundata):
    num_uniques = 0
    for i in range(len(inds)):
        ind = inds[i]
        if len(offspring) == num_to_produce:
            break
        else:
            hashable = ArrayWrapper(ind.output)
            # not loggable since it's not a "real" individual?
            if not try_cache(rundata, hashable, loggable=True):
                offspring.append(ind)
                del ind.fitness.values
                num_uniques += 1
    return num_uniques


import glob, os

clfs = {'KNN': KNeighborsClassifier(n_neighbors=3),
        'RF': RandomForestClassifier(random_state=0, n_estimators=100),
        }


def output_ind(ind, toolbox, rd, suffix="", compress=False, csv_file=None, tree_file=None, del_old=False,
               aug_file=False, classify=False):
    """ Does some stuff

    :param aug_file: Whether to also save the augmented dataset
    :param ind: the GP Individual. Assumed two-objective
    :param toolbox: To evaluate the tree
    :param rd: dict-like object containing data_t (feature-major array), outdir (string-like),
    dataset (name, string-like), labels (1-n array of class labels)
    :param suffix: to go after the ".csv/tree"
    :param compress: boolean, compress outputs or not
    :param csv_file: optional path/buf to output csv to
    :param tree_file: optional path/buf to output tree to
    :param del_old: delete previous generations or not
    :param classify: Should RF etc also be run?

    """
    old_files = glob.glob(rd.outdir + "*.tree" + ('.gz' if compress else ''))
    old_files += glob.glob(rd.outdir + "*.csv" + ('.gz' if compress else ''))
    output = evaluate_trees(rd.data_t, toolbox, ind)
    columns = ['C' + str(i) for i in range(output.shape[1])]
    df = pd.DataFrame(output, columns=columns)
    df["class"] = rd.labels

    compression = "gzip" if compress else None

    f_name = ('{}' + ('-{}' * len(ind.fitness.values)) + '{}').format(rd.dataset, *ind.fitness.values, suffix)

    if csv_file:
        df.to_csv(csv_file, index=None)
    else:
        outfile = f_name + '.csv'
        if compress:
            outfile = outfile + '.gz'
        p = Path(rd.outdir, outfile)
        df.to_csv(p, index=None, compression=compression)

    if aug_file:
        outfile = f_name + '-aug.csv'
        combined_array = np.concatenate((output, rd.data), axis=1)
        aug_columns = columns + ['X' + str(i) for i in range(rd.data.shape[1])]
        df_aug = pd.DataFrame(combined_array, columns=aug_columns)
        df_aug["class"] = rd.labels
        if compress:
            outfile = outfile + '.gz'
        p = Path(rd.outdir, outfile)
        df_aug.to_csv(p, index=None, compression=compression)

    if tree_file:
        tree_file.write(str(ind[0]))
        for i in range(1, len(ind)):
            tree_file.write('\n')
            tree_file.write(str(ind[i]))
    else:
        outfile = f_name + '.tree'
        if compress:
            outfile = outfile + '.gz'

        p = Path(rd.outdir, outfile)
        with gz.open(p, 'wt') if compress else open(p, 'wt') as file:
            file.write(str(ind[0]))
            for i in range(1, len(ind)):
                file.write('\n')
                file.write(str(ind[i]))

    if classify:
        fitness_values = ind.fitness.values
        outdir = rd.outdir
        dataset_name = rd.dataset
        labels = rd.labels
        data = output
        classify_and_save_results(data, labels, dataset_name, fitness_values, outdir, suffix)
    if del_old:
        # print(old_files)
        for f in old_files:
            try:
                os.remove(f)
            except OSError as e:  ## if failed, report it back to the user ##
                print("Error: %s - %s." % (e.filename, e.strerror))


def classify_and_save_results(data, labels, dataset_name, fitness_values, outdir, suffix=""):
    for clf in clfs:
        # sometimes the classifier throws a hissy
        try:
            skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
            scores = []
            for train_index, test_index in skf.split(data, labels):
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                est = clone(clfs[clf])
                # TODO: check...
                model = est.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                scores.append(score)

            mean_acc = sum(scores) / 10
            f_name = ('{}' + ('-{}' * len(fitness_values)) + '-{}-{:.3f}{}.json') \
                .format(dataset_name, *fitness_values, clf, mean_acc, suffix)
            resultsPath = Path(outdir, f_name)
            with open(resultsPath, 'w') as f:
                json.dump(scores, fp=f, cls=NumpyEncoder, sort_keys=True, indent=4, separators=(',', ': '))
        except:
            continue


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def evaluate_trees_with_compiler(data_t, compiler, individual):
    num_instances = data_t.shape[1]
    num_trees = len(individual)

    if not individual.str or len(individual.str) == 0:
        raise ValueError(individual)

    # result = []
    result = np.zeros(shape=(num_trees, num_instances))

    for i, f in enumerate(individual.str):
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


def evaluate_trees(data_t, toolbox, individual):
    return evaluate_trees_with_compiler(data_t, toolbox.compile, individual)


def get_arrays_cached(invalid_ind, toolbox, rundata, in_parallel=True):
    for ind in invalid_ind:
        if (not hasattr(ind, 'str')) or (ind.str is None):
            add_to_string_cache(ind)

    if in_parallel:
        proxy = partial(eval_if_not_cached_compiler, toolbox.compile)
        arrays = toolbox.map(proxy, invalid_ind)
        for i, ind in enumerate(invalid_ind):
            # we need to do this in here, as if it gets assigned in a different thread it doesn't copy back over.
            ind.output = arrays[i]
    else:
        proxy = partial(eval_if_not_cached, toolbox, rundata.data_t)
        arrays = list(map(proxy, invalid_ind))

    return arrays


def eval_if_not_cached_compiler(compiler, ind):
    # basically here so we don't have to pass around the toolbox (multiprocessing issues)
    if (not hasattr(ind, 'output')) or (ind.output is None):
        # reduces the copying between threads.
        ind.output = evaluate_trees_with_compiler_indep(compiler, ind.str)
    return ind.output


def eval_if_not_cached(toolbox, data_t, ind):
    if (not hasattr(ind, 'output')) or (ind.output is None):
        ind.output = eval_tree_wrapper(toolbox, data_t, ind)
    return ind.output


def eval_tree_wrapper(toolbox, data_t, ind):
    return evaluate_trees(data_t, toolbox, ind)


def update_experiment_data(data, ns):
    dict = vars(ns)
    for i in dict:
        if dict[i]:
            setattr(data, i, dict[i])
        else:
            if not hasattr(data, i):
                setattr(data, i, None)
        # data[i] = dict[i]


warnOnce = False


def try_cache(rundata, hashable, index=0, loggable=True):
    if index == -1:
        return
    if loggable:
        rundata.accesses = rundata.accesses + 1

    res = rundata.fitnessCache[index].get(hashable)
    if rundata.accesses % 1000 == 0:
        print("Caches size: " + str(rundata.stores) + ", Accesses: " + str(
            rundata.accesses) + " ({:.2f}% hit rate)".format(
            (rundata.accesses - rundata.stores) * 100 / rundata.accesses))
    return res


def cachedError(hashable, errorFunc, rundata, args, kargs, index=0):
    # global accesses
    if (not hasattr(rundata, 'fitnessCache')) or (rundata.fitnessCache is None):
        if not rundata.warnOnce:
            print("NO CACHE.")
            rundata.warnOnce = True
        return errorFunc(*args, **kargs)

    res = try_cache(rundata, hashable, index)
    if not res:
        res = errorFunc(*args, **kargs)
        rundata.fitnessCache[index][hashable] = res
        rundata.stores = rundata.stores + 1
    # else:
    return res


def arg_parse_helper(*args, **kwargs):
    """ Helps to build additional command line arguments as a list to pass into init_data.
    kwargs should contain 'arg_list' if a list has already been made (e.g. if this isn't the first time that
    this method is being called)
    """

    if 'arg_list' in kwargs:
        # don't add the arg_list..
        copied = kwargs.copy()
        copied.pop('arg_list')
        kwargs['arg_list'].append([args, copied])
    else:
        return [[args, kwargs]]


def init_data(rd, additional_arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logfile", help="log file path", type=str, default="log.out")
    parser.add_argument("-d", "--dataset", help="Dataset file name", type=str, default="wine")
    parser.add_argument("--dir", help="Dataset directory", type=str,
                        default="/home/lensenandr/datasetsPy/")
    parser.add_argument("-g", "--gens", help="Number of generations", type=int, default=1000)
    parser.add_argument("-od", "--outdir", help="Output directory", type=str, default="./")
    parser.add_argument("--erc", dest='use_ercs', help="Use Ephemeral Random Constants?", action='store_true')
    parser.add_argument("--trees", dest="max_trees", help="How many (or maximum if dynamic) trees to use", type=int)
    parser.add_argument("-cr", dest="cxpb", help="crossover rate", type=float)
    parser.add_argument("-mr", dest="mutpb", help="mutation rate", type=float)
    parser.add_argument("-p",dest="pop_size",help="population size",type=int,default=100)
    parser.add_argument("-e",dest="elitism",help="top-n elitism rae",type=int,default=10)
    parser.add_argument('-ef',"--excluded-functions",help="Functions to exclude from the function set (if any)",nargs='+',dest='excluded_functions')
    parser.add_argument("-threads", help="Number of threads to use", dest="threads", type=int, default=1)
    parser.add_argument("-cae", "--classify-at-end", help="Enables performing k-fold classification at the end.",
                     dest="classify_at_end", action="store_true")
    if additional_arguments:
        for arg in additional_arguments:
            # this'll get set in rundata
            parser.add_argument(*arg[0], **arg[1])

    args = parser.parse_args()
    print(args)
    update_experiment_data(rd, args)

    if(rd.outdir):
        if len(list(Path(rd.outdir).glob('*1000.tree'))) > 0:
            print('Skipping as outdir already has completed file')
            sys.exit(0)

    file = Path(args.dir) / (args.dataset + '.data')
    all_data = read_data(file)
    data = all_data["data"]
    rd.num_instances = data.shape[0]
    rd.num_features = data.shape[1]
    rd.labels = all_data["labels"]
    rd.data = data
    rd.data_t = data.T


def final_output(hof, toolbox, logbook, pop, rundata, classify=False):
    for res in hof:
        output_ind(res, toolbox, rundata, compress=False, classify=classify)
    p = Path(rundata.outdir, rundata.logfile + '.gz')
    with gz.open(p, 'wt') as file:
        file.write(str(logbook))
    pop_stats = [str(p.fitness) for p in pop]
    pop_stats.sort()
    hof_stats = [str(h.fitness) for h in hof]
    # hof_stats.sort()
    print("POP:")
    print("\n".join(pop_stats))
    print("PF:")
    print("\n".join(hof_stats))
