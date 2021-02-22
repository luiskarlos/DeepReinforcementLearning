# introduction
> AlphaGo is the first computer program to defeat a professional human Go player, the first to defeat a Go world champion, and is arguably the strongest Go player in history.
>> [Deepmind](https://deepmind.com/research/case-studies/alphago-the-story-so-far)

## What is in this repo?
This is a refactor of [AppliedDataSciencePartners/DeepReinforcementLearning](https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning)
part of the article [How to build your own AlphaZero AI using Python and Keras](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188)

### The purpose of the refactor
1. Make experimentation easier 
1. Add more documentation
1. Explain the algorithms
1. Improve performance

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
 1. Config
    1. Make config adjustable by game
        1. Parameters can vary from game to game, having all in one config makes it difficult for experimentation 
    1. Use defaults
    

# Wondering

# Resource Links

1. Article [How to build your own AlphaZero AI using Python and Keras](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188)
1. Video [AlphaGo Zero Tutorial Part 2 - Monte Carlo Tree Search](https://www.youtube.com/watch?v=NjeYgIbPMmg)
1. [Multiplayer AlphaZero](https://arxiv.org/pdf/1910.13012.pdf)
1. [AlphaZero implementation and tutorial](https://towardsdatascience.com/alphazero-implementation-and-tutorial-f4324d65fdfc)
