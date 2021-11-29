#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *****************
    Genetic Algorithm
    *****************


"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2021, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'GeneticAlgorithm'
]


import numpy as np

import logging
log = logging.getLogger(__name__)


randGen = np.random.default_rng()


# ============================================================= #
# --------------------- GeneticAlgorithm ---------------------- #
# ============================================================= #
class GeneticAlgorithm(object):
    """

        .. versionadded:: 1.2.0
    """

    def __init__(self, populate, fitness, mutation, populationSize=10):
        self._childScores = []
        self._genScores = []
        self._bestScore = 0
        self.bestGenome = None

        self.generation = 0
        self.populate = populate
        self.fitness = fitness
        self.mutation = mutation
        self.populationSize = populationSize


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def populationSize(self):
        """
        """
        return self._populationSize
    @populationSize.setter
    def populationSize(self, popsize):
        if not isinstance(popsize, int):
            raise TypeError(
                '<populationSize> should be integer.'
            )
        elif popsize < 2:
            raise ValueError(
                '<populationSize> should at least be greater than '
                '2 in order for the genetic algorithm to work.'
            )
        self._populationSize = popsize


    @property
    def populate(self):
        """
        """
        return self._populate
    @populate.setter
    def populate(self, pop):
        # Quick test to see if the function acts as expected
        try:
            population = pop(3)
        except:
            raise TypeError(
                '<populate()> must be callable.'
            )

        if not isinstance(population, np.ndarray):
            raise TypeError(
                '<populate()> must be callable and must return'
                f' a {np.ndarray}.'
            )
        elif population.ndim != 2:
            raise IndexError(
                'Result of <populate()> must be 2D, current is '
                f'of dimension {population.ndim}.'
            )
        elif population.shape[0] != 3:
            raise IndexError(
                'Result of <populate()> must be shaped like '
                '(population_size, genome_size).'
            )
        else:
            self._populate = pop


    @property
    def fitness(self):
        """
        """
        return self._fitness
    @fitness.setter
    def fitness(self, fit):
        # Quick test to see if the function acts as expected
        try:
            scores = fit(
                self.populate(3)
            )
        except:
            raise TypeError(
                '<fitness()> must be callable.'
            )

        if not isinstance(scores, np.ndarray):
            raise TypeError(
                '<fitness()> must be callable and must return'
                f' a {np.ndarray}.'
            )
        elif scores.ndim != 1:
            raise IndexError(
                'Result of <fitness()> must be 1D, current is '
                f'of dimension {scores.ndim}.'
            )
        elif scores.size != 3:
            raise IndexError(
                'Result of <fitness()> must be shaped like '
                '(population_size,).'
            )
        else:
            self._fitness = fit


    @property
    def mutation(self):
        """
        """
        return self._mutation
    @mutation.setter
    def mutation(self, mut):
        # Quick test to see if the function acts as expected
        try:
            individual = self.populate(1)[0]
            mutant = mut(
                individual
            )
        except:
            raise TypeError(
                '<mutation()> must be callable.'
            )

        if not isinstance(mutant, np.ndarray):
            raise TypeError(
                '<mutation()> must be callable and must return'
                f' a {np.ndarray}.'
            )
        elif mutant.ndim != 1:
            raise IndexError(
                'Result of <mutation()> must be 1D, current is '
                f'of dimension {mutant.ndim}.'
            )
        elif mutant.size != individual.size:
            raise IndexError(
                'Result of <mutation()> must be shaped like '
                '(genome_size,).'
            )
        else:
            self._mutation = mut
    

    @property
    def generations(self):
        """
        """
        return np.arange(self.generation + 1)


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def evolve(self,
            score_threshold=0.8,
            generation_max=1000,
            max_stagnating_generations=100,
            **kwargs
        ):
        """
        """
        if score_threshold > 1.:
            raise ValueError(
                f'<scoreMin={score_threshold}> should be < 1.'
            )

        random_individuals = kwargs.get('random_individuals', 1)
        if not isinstance(random_individuals, int):
            raise TypeError(
               f'<random_individuals={random_individuals}> should be integer.'
            )
        elif random_individuals > self.populationSize:
            raise ValueError(
                f'<random_individuals={random_individuals}> is greater than the '
                f'population size {self.populationSize}.'
            )
        parent_selection = kwargs.get('selection', 'TNS')
        crossover_method = kwargs.get('crossover', 'TPCO')
        beElitist = kwargs.get('elitism', True)

        log.info(
            'Genetic algorithm launched.'
        )

        modGen = generation_max//10 if generation_max>=10 else 1
        nStag = 0

        # Initialization of a starting population of solutions
        population = self.populate(self.populationSize)#, **kwargs)
        popDtype = population.dtype
        self.bestGenome = population[0]

        # Genetic Loop
        while self.generation < generation_max:
            # Evaluate the fitness of the population
            populationScores = self.fitness(population)
            self._childScores.append(populationScores)

            # Keep track of the best individual and score
            maxId = np.argmax(populationScores)
            score = populationScores[maxId]
            self._genScores.append(score)
            if score > self._bestScore:
                self._bestScore = score
                self.bestGenome = population[maxId]
                nStag = 0 # the score has changed, reset nStag

            # Show status
            if self.generation%modGen == 0:
                log.debug(
                    f'Generation {self.generation}, '
                    f'best score: {self._bestScore}.'
                )

            # Break the loop
            if score >= score_threshold:
                # if the score exceeds the minimal required
                log.info(
                    'Minimal required score reached at '
                    f'generation {self.generation}.'
                )
                break
            elif (nStag==max_stagnating_generations) and (self._bestScore!=0):
                # If the genome is no longer evolving
                log.info(
                    f'Maximal score has stagnated for {nStag} '
                    'generations, interrupting the evolution at '
                    f'generation {self.generation}.'
                )
                break

            # If we are still here, the scoreMin has not been
            # reached yet and the genetic loop may go on.

            # Sort the population by decreasing score
            decreasingIdx = np.argsort(populationScores)[::-1]
            population = population[decreasingIdx]
            populationScores = populationScores[decreasingIdx]

            # Select the two best parents and keep them for the
            # next generation --> 'Elitism'
            nextGeneration = np.zeros(
                population.shape,
                dtype=popDtype
            )
            nextGeneration[0:2] = population[0:2]

            # Perform crossovers with the rest of the population
            for i in range(int(self.populationSize/2) - 1):
                # Select 2 parents from the population with a
                # probability weighted by their score
                parents = self._selectPair(
                    population=population,
                    scores=populationScores,
                    method=parent_selection
                )
                # Make two children out of the parents by crossing
                # their genomes
                child1, child2 = self._crossOver(
                    parents=parents,
                    method=crossover_method
                )
                # Randomly mutate one gene of each child
                child1 = self.mutation(
                    genome=child1
                )
                child2 = self.mutation(
                    genome=child2
                )
                # Add the children to the new generation
                nextGeneration[2 + i*2] = child1
                nextGeneration[2 + i*2 + 1] = child2

            # Add some randomness to avoid local minima by replacing
            # the last member of the next generation by an individual
            # with random genome
            if random_individuals > 0:
                nextGeneration[-random_individuals:] = self.populate(random_individuals)

            # Do the loop again
            population = nextGeneration
            self.generation += 1
            nStag += 1

        else:
            log.info(
                'Genetic algorithm reached maximal generation '
                f'{generation_max}. Best score: {self._bestScore}.'
            )
            self.generation -= 1

        return


    def plot(self, **kwargs):
        """
            kwargs:
                figname
                figsize
                grid
        """
        import matplotlib.pyplot as plt
        
        # Initialize the figure
        fig, ax = plt.subplots(
            figsize=kwargs.get('figsize', (10, 5))
        )
        plt.plot(
            self.generations,
            self._genScores,
            color='black',
            linestyle=':',
            linewidth=2,
            zorder=1
        )
        childrenScores = np.array(self._childScores)
        childrenScores = np.ma.masked_equal(childrenScores, 0)
        hexb = plt.hexbin(
            x=np.repeat(self.generations, childrenScores.shape[1]),
            y=childrenScores.flatten(),
            gridsize=50,
            cmap='Greys',
            edgecolor='none',
            bins=None,
            mincnt=1
        )
        cb = plt.colorbar(hexb)
        scat = ax.scatter(
            self.generations,
            self._genScores,
            100,
            c=self._genScores,
            cmap='RdYlGn',
            edgecolor='none',
            linewidth=0.5,
            zorder=2,
            norm=plt.Normalize(0, 1)
        )

        if kwargs.get('grid', False):
            plt.grid()

        ax.set_ylim(0, 1)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Score')
        cb.set_label('Children')

        # Save or show the figure
        figname = kwargs.get('figname', '')
        if figname != '':
            plt.savefig(
                figname,
                dpi=300,
                bbox_inches='tight',
                transparent=True
            )
            log.info(f"Figure '{figname}' saved.")
        else:
            plt.show()
        plt.close('all')


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    @staticmethod
    def _selectPair(population, scores, method='TNS'):
        """
            method ['FPS']
                FPS: Fitness Proportionate Selection
                TNS: Tournament Selection
                RKS: Rank Selection
        """
        if method == 'FPS':
            # Fitness Proportionate Selection
            # Same as Roulette Wheel Selection, parents are as
            # likely to be picked as their score is high.
            if (scores!=0).sum() < 2:
                # Equal probability to select any parent
                scores[:] = 1
            scoreSum = np.sum(scores)
            probabilities = None if scoreSum==0 else scores/scoreSum
            
            return randGen.choice(
                population,
                size=2,
                replace=False,
                p=probabilities
            )

        elif method == 'TNS':
            # Tournament Selection
            # Select k individuals from the population and find
            # the best out of these to make a parent.
            k = np.max(
                (2, int(np.ceil(population.shape[0]/10)))
            )
            selectIdx = np.zeros(2, dtype=int)
            for i in range(2):
                indices = randGen.choice(
                    np.arange(population.shape[0]),
                    size=k,
                    replace=False,
                )
                maxScoreIdx = np.argmax(scores[indices])
                selectIdx[i] = indices[maxScoreIdx]
            return population[selectIdx]

        elif method == 'RKS':
            # Rank selection
            # Individuals are ranked according to their scores,
            # and are more likely to be picked according to their rank.
            # The amplitude of score differences doesn't count.
            ranks = np.argsort(scores) + 1        
            rankSum = np.sum(ranks)        
            return randGen.choice(
                population,
                size=2,
                replace=False,
                p=ranks/rankSum
            )

        else:
            raise ValueError(
                "Argument <method> from <_selectPair()> "
                "must be in ['FPS', 'TNS', 'RKS']."
            ) 


    @staticmethod
    def _crossOver(parents, method='TPCO'):
        """ 
            method ['SPCO', 'TPCO', 'UNCO']
                SPCO: Single-point crossover
                TPCO: Two-point and k-point crossover
                UNCO: Uniform crossover
        """
        parent1 = parents[0]
        parent2 = parents[1]
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if parent1.size == 1:
            # No need to perform crossovers on size one genomes
            pass

        elif method=='SPCO':
            # Single-point crossover
            idx = randGen.integers(
                low=1,
                high=parent1.size
            )
            child1[idx:] = parent2[idx:]
            child2[idx:] =  parent1[idx:]
        
        elif method=='TPCO':
            # Two-point crossover
            idx = randGen.integers(
                low=0,
                high=parent1.size,
                size=2
            )
            idx = np.sort(idx)
            child1[idx[0]:idx[1]] = parent2[idx[0]:idx[1]]
            child2[idx[0]:idx[1]] = parent1[idx[0]:idx[1]]
        
        elif method=='UNCO':
            # Uniform crossovers
            idx = randGen.integers(
                low=0,
                high=2,
                size=parent1.size
            )
            mask = idx.astype(bool)
            child1[mask] = parent2[mask]
            child2[mask] = parent1[mask]

        else:
            raise ValueError(
                "Argument <method> from <_crossOver()> "
                "must be in ['SPCO', 'TPCO', 'UNCO']."
            )

        return child1, child2
# ============================================================= #
# ============================================================= #

