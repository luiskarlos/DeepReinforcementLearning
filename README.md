# introduction
> AlphaGo → AlphaGo Zero → AlphaZero
> In March 2016, Deepmind’s AlphaGo beat 18 times world champion Go player Lee Sedol 4–1 in a series watched by over 200 million people. A machine had learnt a super-human strategy for playing Go, a feat previously thought impossible, or at the very least, at least a decade away from being accomplished.
> [How to build your own AlphaZero AI using Python and Keras](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188)

## What is in this repo?
This is a refactor of [AppliedDataSciencePartners/DeepReinforcementLearning](https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning)
part of the article [How to build your own AlphaZero AI using Python and Keras](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188)

The purpose of the refactor is to make easy to add new games and also add more documentation and explain the algorithms

# How to run it?

1. Install [Python 3.6.*](https://www.python.org/downloads/)
   1. Do not install a newer version it will not compile
1. Install libraries in requirements.txt
1. Execute [main.py](./main.py)

# Road map

1. Improve documentation
1. Fix jupyter notebook
1. Add unit test
1. Add profiler instrumentation
    1. Learning vs MCTS
    1. Memory usage
1. Add [Dependency Injector](https://python-dependency-injector.ets-labs.org/)
1. Add simple games to make it easier to understand
1. Change logger to [Resource provider](https://python-dependency-injector.ets-labs.org/providers/resource.html#resource-provider)
    1. Make logger less intrusive
    1. Improve log data
1. Add stats to a DB (remove some logs)
1. Update to tensorflow 2
1. Add online demo
    1. Player VS NPC
    1. NPC VS NPC
    1. Interactive learning process

# Wondering
1. Should all configuration be related to the game?
    1. Should the complexity of the game be reflected in config?
    1. if so the configuration we should have defaults and game specific

# Resource Links

1. Article [How to build your own AlphaZero AI using Python and Keras](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188)
1. Video [AlphaGo Zero Tutorial Part 2 - Monte Carlo Tree Search](https://www.youtube.com/watch?v=NjeYgIbPMmg)
1. [Multiplayer AlphaZero](https://arxiv.org/pdf/1910.13012.pdf)
1. [AlphaZero implementation and tutorial](https://towardsdatascience.com/alphazero-implementation-and-tutorial-f4324d65fdfc)
