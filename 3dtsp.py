#!/usr/bin/env python
# coding: utf-8

# In[198]:


import numpy as np
import random


class Location:
    """
    Holds the coordinates of a location and calculate any distance origniating from these coordinates
    """

    def __init__(self,xyz):
        self.coordinates = np.array(xyz)
    
    def calculate_distance(self,location_b):
        return np.linalg.norm(self.coordinates-location_b.coordinates, 2) # second degree norm is the euclidean distance between two vectors/coordinates
    
    
class Path:
    """
    Holds all locations of a full path and calculate cost and fitness of the path
    """

    def __init__(self,path):
        self.path = path
        self.cost = 0
        self.fitness = 0
    
    def _calculate_cost(self):
        for i in range(len(self.path)-1):
            self.cost += self.path[i].calculate_distance(self.path[i+1])
    
    def calculate_fitness(self):
        self._calculate_cost()
        self.fitness = 1/self.cost # fitness is just 1/cost
      
    
def _ordered_crossover(path_a, path_b):
    """
    Using Ordered Crossover (OX1) to breed new generations
    """

    child_a, child_b = [],[]

    start_point, end_point = sorted([random.randrange(len(path_a.path)), random.randrange(len(path_a.path))]) # pick two random crossover points
    
    child_a = list(path_b.path[start_point:end_point])
    child_b = list(path_a.path[start_point:end_point])
    
    residual_a = [location for location in path_a.path if location not in child_a]
    residual_b = [location for location in path_b.path if location not in child_b]
    
    child_a = residual_a[0:start_point] + child_a
    child_b = residual_b[0:start_point] + child_b
    
    del residual_a[0:start_point]
    del residual_b[0:start_point]
    
    child_a += residual_a
    child_b += residual_b
    
    child_a, child_b = Path(child_a), Path(child_b)
    child_a.calculate_fitness()
    child_b.calculate_fitness()
    
    if child_a.fitness >= child_b.fitness: # only return the fitter child to accelerate convergence
        return child_a
    else:
        return child_b

    
def _mutate(path_a, mutation_rate):
    """
    Mutation by swapping two random locations
    """

    if random.random() <= mutation_rate:
        i, j = random.randrange(len(path_a.path)), random.randrange(len(path_a.path))
        location_a, location_b = path_a.path[i], path_a.path[j]
        path_a.path[i] = location_b
        path_a.path[j] = location_a
    return path_a


def _construct_path(locations):
    """
    Create random permutations of all locations
    """

    random_path = Path(np.random.permutation(locations))
    random_path.calculate_fitness()
    return random_path

def read_input_file():
    """
    Read input file and store all locations into an np array
    """
    
    with open('input.txt') as file:
        input_file = [line.rstrip() for line in file]
        input_size = int(input_file.pop(0))
        locations = [Location(list(map(int,location.split(' ')))) for location in input_file]
        locations = np.array(locations)
    return input_size, locations


def create_population(locations, population_size):
    """
    Create a population with randomly constructed paths
    """

    population = []
    for i in range(population_size):
        population.append(_construct_path(locations))
    return population


def rank_population(population):
    """
    Rank the population based on fitness scores
    """

    population_rank = {}
    for i in range(len(population)):
        population_rank[i] = population[i].fitness
    population_rank = dict(sorted(population_rank.items(), key=lambda item: item[1], reverse=True)) # ranked by fitness while maintaining the index order of the paths for future reference
    return list(population_rank.keys()),list(population_rank.values())


def create_mating_pool(population, fitness_ranks, fitness_values, elite_size):
    """
    Create the mating pool for next generations by using elitism and roulette wheel-based selection
    """

    mating_pool = []
    for i in range(elite_size):
        mating_pool.append(population[fitness_ranks[i]]) # keep the elites for next generations
    total_fitness = sum(fitness_values)
    probabilities = [fitness/total_fitness for fitness in fitness_values] # calculate and assign probability based on fitness (roulette wheel)
    roulette_wheel_ranks = np.random.choice(fitness_ranks, len(population), probabilities) # random selection based on probability
    for i in range(len(roulette_wheel_ranks)-elite_size):
        mating_pool.append(population[roulette_wheel_ranks[i]])
    return mating_pool


def breed_next_generation(mating_pool, elite_size, mutation_rate):
    """
    Breed the next generation by carrying over the elites and crossovering and mutating the rest of the mating pool
    """

    next_generation = []
    for i in range(elite_size):
        next_generation.append(mating_pool[i]) # elitism continued
    
    for i in range(len(mating_pool)-elite_size): # rest of the mating pool will get crossover and mutation
        new_child = _ordered_crossover(mating_pool[random.randrange(len(mating_pool))],
                                       mating_pool[random.randrange(len(mating_pool))])
        new_child = _mutate(new_child, mutation_rate)
        next_generation.append(new_child)
        
    return next_generation


def main(population_size=100, elite_size=10, mutation_rate=0.01, iterations=250):
    
    input_size, locations = read_input_file()
    
    if input_size <= 50: # finetuning runtime vs optimality based on input class/size
        iterations = 500
    elif input_size <= 100:
        iterations = 400
    elif input_size <= 200:
        iterations = 300
    else:
        iterations = 200
        
    population = create_population(locations, population_size)
    
    for i in range(iterations): # iterative genetic algorithm
        rank, value = rank_population(population)
        mating_pool = create_mating_pool(population, rank, value, elite_size)
        population = breed_next_generation(mating_pool, elite_size, mutation_rate)
    
    rank, value = rank_population(population)
    best_path = population[rank[0]].path
    best_path.append(best_path[0]) # add starting location

    with open('../../Masters/Year 1/Fall 2022/CSCI 561 - Foundations of AI/HW 1/output.txt', 'w', newline='\n') as f: # write output to txt
        f.write('\n'.join([' '.join(map(str,location.coordinates)) for location in best_path]))
        
        
if __name__ == "__main__":
    main()


# ' '.join(map(str,population[0].path[0].coordinates))

# In[ ]:




