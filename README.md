# moevorl-ev-cm
The electrification of transportation requires the development of smart charging management systems for electric vehicles to optimize grid performance and enhance user satisfaction. 
In this study, we propose the use of Multi-Objective Evolutionary Reinforcement Learning (MOEvoRL) to optimize electric vehicle charging strategies. 
Our approach focuses on maximizing the batteries' state of charge, increasing photovoltaic power consumption, reducing peak loads, and smoothing the overall load on the grid. 
Simultaneously, it adheres to essential grid constraints, such as load balancing and grid connection node limits, to ensure grid stability, efficiency, and real-world applicability.

MOEvoRL utilizes the exploratory power of Evolutionary Algorithms and the sequential decision-making strengths of Reinforcement Learning. 
By employing neuroevolution, we optimize both the weights and topologies of policy networks.

Our approach employs the Non-dominated Sorting Genetic Algorithm II, Strength Pareto Evolutionary Algorithm 2, and a modified NeuroEvolution of Augmenting Topologies as optimizers and benchmarks their performance against the gradient-based Multi-Objective Deep Deterministic Policy Gradient (MODDPG).

The results show that MOEvoRL approaches are superior to MODDPG in terms of generalization, robustness, constraint compliance, and multi-objective optimization capabilities. 
This positions MOEvoRL as a robust strategy for managing electric vehicle charging while maintaining grid stability.
