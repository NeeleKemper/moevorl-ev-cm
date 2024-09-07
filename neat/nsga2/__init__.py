"""

    Implementation of NSGA-II as a reproduction method for NEAT.
    More details on the README.md file.

    @autor: Hugo Aboud (@hugoaboud)
    Improved by Neele Kemper
"""
from __future__ import division
from itertools import count
from operator import add

import numpy as np
from pymoo.util.dominator import Dominator
from pymoo.algorithms.moo.nsga2 import NonDominatedSorting, calc_crowding_distance
from neat.config import ConfigParameter, DefaultClassConfig
from neat.species import Species


##
#   NSGA-II Fitness
#   Stores multiple fitness values
#   Overloads operators allowing integration to unmodified neat-python
##

class NSGA2Fitness:
    def __init__(self, *values):
        self.values = values
        self.rank = 0
        self.dist = 0.0

    def set(self, *values):
        self.values = values

    def add(self, *values):
        self.values = list(map(add, self.values, values))

    def dominates(self, other):
        """Check if this fitness dominates the other using pymoo's Dominator."""
        # For minimization problems, if solution 'a' dominates 'b',
        # the Dominator will return 1, -1 if 'b' dominates 'a', and 0 if they are non-dominated.
        result = Dominator.get_relation(self.values, other.values)
        return True if result == 1 else False

    def __gt__(self, other):
        # comparison of fitnesses on tournament, use crowded-comparison operator
        # this is also used by max/min
        if (isinstance(other, NSGA2Fitness)):
            return (self.rank > other.rank) or (self.rank == other.rank and self.dist > other.dist)
        return self.rank > other

    def __ge__(self, other):
        # population.run() compares fitness to the fitness threshold for termination
        # it's the only place where the next line should be called
        # it's also the only place where score participates of evolution
        # besides that, score is a value for reporting the general evolution
        return self.values[0] >= other

    # -
    def __sub__(self, other):
        # used only by reporting->neat.math_util to calculate fitness (score) variance
        # return self.score - other
        return self.values[0] - other

    # float()
    def __float__(self):
        # used only by reporting->neat.math_util to calculate mean fitness (score)
        # return self.score
        return float(self.values[0])

    # str()
    def __str__(self):
        # return "rank:{0},score:{1},values:{2}".format(self.rank, self.score, self.values)
        return "rank:{0},dist:{1},values:{2}".format(self.rank, self.dist, self.values)


##
#   NSGA-II Reproduction
#   Implements "Non-Dominated Sorting" and "Crowding Distance Sorting" to reproduce the population
##

class NSGA2Reproduction(DefaultClassConfig):
    @classmethod
    def parse_config(cls, param_dict):

        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('survival_threshold', float, 0.2)])

    def __init__(self, config, reporters, stagnation, rnd):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.rnd = rnd

        # Parent population and species
        # This population is mixed with the evaluated population in order to achieve elitism
        self.parent_pop = []
        self.parent_species = {}

    # new population, called by the population constructor
    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
        return new_genomes

    # NSGA-II step 1: fast non-dominated sorting
    # This >must< be called by the fitness function (aka eval_genomes)
    # after a NSGA2Fitness was assigned to each genome
    def sort(self, population, species, pop_size, generation):
        # Filter out stagnated species genomes
        remaining_species = {}  # remaining species
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            # stagnant species: remove genomes from child population
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
                population = {id: g for id, g in population.items() if g not in stag_s.members}
            # non stagnant species: append species to parent species dictionary
            else:
                remaining_species[stag_sid] = stag_s

        # No genomes left.
        if not remaining_species:
            print('no remaining species')
            species.species = {}
            return {}

        ## NSGA-II : step 1 : merge and sort
        child_pop = [g for _, g in population.items()] + self.parent_pop

        # Merge parent P(t) species and child (Qt) species,
        species.species = remaining_species
        for id, sp in self.parent_species.items():
            if (id in species.species):
                species.species[id].members.update(sp.members)
            else:
                species.species[id] = sp

        # Using pymoo's non-dominated sorting
        F = np.array([ind.fitness.values for ind in child_pop])

        # Non-Dominated Sorting
        nds = NonDominatedSorting()
        front_indices_list = nds.do(F)

        # Assign ranks to individuals based on fronts
        for rank, front_indices in enumerate(front_indices_list, start=1):
            for idx in front_indices:
                child_pop[idx].fitness.rank = rank

        self.parent_pop = []
        added_keys = set()

        for front_indices in front_indices_list:
            fitness_values = F[front_indices]
            # Calculate crowding distance for this front
            crowding_distances = calc_crowding_distance(fitness_values)

            for i, idx in enumerate(front_indices):
                child_pop[idx].fitness.dist = crowding_distances[i]

            current_front = [child_pop[i] for i in front_indices]
            current_front.sort(key=lambda x: (x.fitness.dist, x.fitness.values[0]), reverse=True)

            remaining_space = pop_size - len(self.parent_pop)
            if remaining_space <= 0:
                break

            for individual in current_front:
                if individual.key not in added_keys:
                    self.parent_pop.append(individual)
                    added_keys.add(individual.key)
                    if len(self.parent_pop) >= pop_size:
                        break

        ## NSGA-II : post step 2 : Clean Species
        species.genome_to_species = {}
        for _, sp in species.species.items():
            sp.members = {id: g for id, g in sp.members.items() if g in self.parent_pop}
            # map genome to species
            for id, g in sp.members.items():
                species.genome_to_species[id] = sp.key
        # Remove empty species
        species.species = {id: sp for id, sp in species.species.items() if len(sp.members) > 0}

        self.parent_species = {}
        for id, sp in species.species.items():
            self.parent_species[id] = Species(id, sp.created)
            self.parent_species[id].members = dict(sp.members)
            self.parent_species[id].representative = sp.representative

        ## NSGA-II : end : return parent population P(t+1) to be assigned to child population container Q(t+1)
        return {g.key: g for g in self.parent_pop}

    def reproduce(self, config, species, pop_size, generation):
        new_population = {}

        # Calculate the rank of each species (average rank)
        species_ranks = {spec_id: sp.calculate_rank() for spec_id, sp in species.species.items()}
        # Allocate offspring proportionally based on inverse rank (lower rank is better)
        rank_sum = sum(1.0 / rank for rank in species_ranks.values())
        species_sizes = {
            spec_id: max(int((1.0 / rank) / rank_sum * pop_size), 1)
            for spec_id, rank in species_ranks.items()
        }

        # Ensure we're not exceeding the desired population size due to rounding
        num_remaining = pop_size - sum(species_sizes.values())
        if num_remaining > 0:
            extra_species = self.rnd.choices(list(species.species.keys()), k=num_remaining)
            for spec_id in extra_species:
                species_sizes[spec_id] += 1
        for spec_id, sp in species.species.items():
            # Sort members by fitness
            members = sorted(sp.members.values(), key=lambda x: (x.fitness.rank, -x.fitness.dist))
            # Determine number of elites and add them to the new population
            n_elites = min(int(self.reproduction_config.survival_threshold * len(members)), species_sizes[spec_id])
            elites = members[:n_elites]
            for elite in elites:
                gid = elite.key
                new_population[gid] = elite

            # Determine number of offspring excluding elites
            num_offspring = species_sizes[spec_id] - n_elites

            # If we have space for offspring, proceed with reproduction
            if num_offspring > 0:
                # Proceed with tournament selection among non-elite members
                for _ in range(num_offspring):
                    parent1 = self.tournament_select(members)
                    parent2 = self.tournament_select(members)

                    gid = next(self.genome_indexer)
                    child = config.genome_type(gid)
                    child.configure_crossover(parent1, parent2, config.genome_config)
                    child.mutate(config.genome_config)
                    new_population[gid] = child

                    if len(new_population) >= pop_size:
                        break

        return new_population

    def tournament_select(self, members, n=2):
        """
        Select one member based on a tournament among n members.
        :param members: List of available members for the tournament.
        :param n: Number of members participating in each tournament.
        :return: Selected member.
        """
        # Get random unique candidates for the tournament
        n = min(n, len(members))
        candidates = self.rnd.sample(members, n)

        # Select the best one based on rank and crowding distance
        winner = min(candidates, key=lambda x: (x.fitness.rank, -x.fitness.dist))

        return winner
