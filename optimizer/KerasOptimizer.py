from deap import base, algorithms
from deap import creator
from deap import tools
from optimizer.Strategy import Strategy
import random
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt


class KerasOptimizer:
    dataset = None
    max_iteration = 100
    initial_population = 50
    layer_size = 2
    classes = 2
    input_shape = None
    weights = None
    include_top = False
    crossover_prob = 0.7
    mutation_probability = 0.01

    toolbox = None
    hyperparam_list = []
    hyperparam_dict = {}
    hyperparam_index_dict = {}

    model = None

    show_graph = False
    train_function = None

    def __init__(self, dataset=None, max_iteration=100, initial_population=20, layer_size=2, classes=2,
                 input_shape=(224, 224, 3), weights=None, crossover_prob=0.7, mutation_probability=0.01):
        self.initial_population = initial_population
        self.dataset = dataset
        self.layer_size = layer_size
        self.max_iteration = max_iteration
        self.classes = classes
        self.input_shape = input_shape
        self.weights = weights
        self.toolbox = base.Toolbox()
        self.crossover_prob = crossover_prob
        self.mutation_probability = mutation_probability

    def select_optimizer_strategy(self, strategy):

        if "FitnessFunc" in globals():
            del globals()["FitnessFunc"]

        if strategy == Strategy.MAXIMIZE:
            creator.create("FitnessFunc", base.Fitness, weights=(1.0,))
        else:
            creator.create("FitnessFunc", base.Fitness, weights=(-1.0,))

        if "Individual" in globals():
            del globals()["Individual"]

        creator.create("Individual", list, fitness=creator.FitnessFunc)

    def add_hyperparameter(self, hyperparam_name, hyperparam_value):
        self.toolbox.register(hyperparam_name, random.choice, hyperparam_value)
        self.hyperparam_list.append(getattr(self.toolbox, hyperparam_name))
        self.hyperparam_dict[hyperparam_name] = len(self.hyperparam_list) - 1
        self.hyperparam_index_dict[len(self.hyperparam_list) - 1] = self.hyperparam_list[len(self.hyperparam_list) - 1]
        return self

    def show_graph_on_end(self,show=True):
        self.show_graph = show

    def run(self, model_function, train_function):
        self.toolbox.register("individual", tools.initCycle, creator.Individual, self.hyperparam_list,
                              n=1)

        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        self.model = model_function()
        self.train_function = train_function

        population_size = self.initial_population
        number_of_generations = 4

        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=self.crossover_prob, stats=stats,
                                       mutpb=self.mutation_probability, ngen=number_of_generations, halloffame=hof,
                                       verbose=True)

        best_parameters = hof[0]  # save the optimal set of parameters
        print('Best parameters:', best_parameters)

        if self.show_graph:
            gen = log.select("gen")
            max_ = log.select("max")
            avg = log.select("avg")
            min_ = log.select("min")

            evolution = pd.DataFrame({'Generation': gen,
                                      'Max AUROC': max_,
                                      'Average': avg,
                                      'Min AUROC': min_})

            plt.title('Parameter Optimisation')
            plt.plot(evolution['Generation'], evolution['Min AUROC'], 'b', color='C1',
                     label='Min')
            plt.plot(evolution['Generation'], evolution['Average'], 'b', color='C2',
                     label='Average')
            plt.plot(evolution['Generation'], evolution['Max AUROC'], 'b', color='C3',
                     label='Max')

            plt.legend(loc='lower right')
            plt.ylabel('AUROC')
            plt.xlabel('Generation')
            plt.xticks([0, 5, 10, 15, 20])
            plt.show()

    def evaluate(self, individual):
        print('START---------------------------------------------')
        print('Individual:', individual)
        batch_size = individual[self.hyperparam_dict['batch_size']]
        epochs = individual[self.hyperparam_dict['epochs']]
        learning_rate = individual[self.hyperparam_dict['learning_rate']]

        decay = 1e-6

        score = self.train_function({
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "decay": decay,
            "momentum" : individual[self.hyperparam_dict['momentum']]
        })
        print('Score:', score, 'Individual:', individual)
        print('END--------------------------------------')
        return score
        print('END--------------------------------------')

    def mutate(self, individual):
        gene = random.randint(0, individual.__len__() - 1)
        individual[gene] = self.hyperparam_index_dict[gene]()
        return individual,

    def trainModel(self, hyperparams):
        pass
