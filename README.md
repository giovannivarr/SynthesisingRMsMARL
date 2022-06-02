# SynthesisingRMsMARL
Repository containing code of the experiments appearing in the paper "Synthesising Reward Machines for Cooperative Multi-Agent Reinforcement Learning".

# How to run the code
There are two possible ways of running our code:
1. By running the file `run.py`, where the variable `experiment` specifies which kind of experiment to perform;
2. By running the file `run_compare.py`, specifying the experiment using always the variable `experiment`. Experiments from these files compare reward machines obtained generated using MCMAS against those crafted by hand from [[1]](#1). 


# Acknowledgements 
Many of the files are originally or have been adapted from other files in the repository at [github.com/cyrusneary/rm-cooperative-marl](https://github.com/cyrusneary/rm-cooperative-marl). We thank the authors for their availability in sharing the code and for their original work in RM-based MARL. 

## References
<a id="1">[1]</a> 
Neary, C., Xu, Z., Wu, B., & Topcu, U. (2021, May). 
Reward Machines for Cooperative Multi-Agent Reinforcement Learning. 
In Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems (pp. 934-942).
