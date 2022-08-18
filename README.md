# HPB Agent Architecture
An agent architecture for controlling an autonomous agent in stochastic, noisy environments.
The architecture combines the partially observable Markov decision process (POMDP) model with the belief-desire-intention (BDI) framework.
The Hybrid POMDP-BDI agent architecture takes the best features from the two approaches, that is,
the online generation of reward- maximizing courses of action from POMDP theory, and sophisticated multiple goal management from BDI theory.
See the article for details: G. Rens &amp; D Moodley. A hybrid POMDP-BDI agent architecture with online stochastic planning and plan caching. J. of Cognitive Systems (2017)

This is an early version of the architecture not meant for public distribution.
The algorithms and domain are in one file, which is actually not good practice.

## Instructions
To start the agent in the TrickyTreats domain, simply run TrickyTreats-for-HPB-arch_w_PlanLib.py

experiment(trials,  nuof_acts,  h) is the main function being called.

Change these parameters as desired.
       trials: the nuof trials
       nuof_acts: the nuof plan-act iterations per trial
       h: horizon
  
For instance, set experiment(30, 100, 4) or experiment(10, 100, 3)
