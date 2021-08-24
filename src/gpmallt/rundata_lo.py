import cachetools
from defaultlist import defaultlist


class RD(object):
    def __init__(self):
        self.data = None
        self.data_t = None
        self.labels = None
        self.outdir = None
        self.pairwise_distances = None
        self.ordered_neighbours = None
        self.neighbours = None
        self.all_orderings = None
        self.all_orderings_sorted = None
        self.all_orderings_argsorted = None
        self.identity_ordering = None
        self.nobj = 1
        self.fitnessCache = defaultlist(lambda: cachetools.LRUCache(maxsize=1e6))
        self.accesses = 0
        self.stores = 0

        self.max_depth = 8  # 7#12#8
        self.max_height = 14  # 10#17#14
        self.pop_size = 100  # 1024#100
        self.cxpb = 0.8
        self.mutpb = 0.2
        self.elitism = 10
        self.max_trees = 2#5#2
        self.gens = 1000

        self.num_instances = 0
        self.num_features = 0

        self.num_neighbours = 10


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


rd = RD()
