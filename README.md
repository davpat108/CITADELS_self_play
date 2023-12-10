# CITADELS_self_play

## Table of Contents

- [Introduction](#introduction)
- [Structure](#structure)
- [Usage](#usage)
- [Training method](#training)
- [Chellenges](#chellenges)


## Introduction <a name = "introduction"></a>
This is my hobby project for creating an agent that can play the board game [Citadels](https://www.ultraboardgames.com/citadels/deluxe.php) against itself. The game and the algorithms are both coded from scratch. The goal is to create an agent that can play the game at a high level. I made the simplification that the game is always played by 6 people, and currently only with a set cards and characters. I implemented the deep learning assisted [MCCFR](https://arxiv.org/pdf/1811.00164.pdf).

## Structure <a name = "structure"></a>
#### Game
Game is made out of 3 main classes: 'Game', 'Agent' and 'Option'. Where the game contains the public informations, the agent the private informations, while also having a get_options function that based on the game returns all the options the agent can choose from. The option class contains the information about the option, while also containing a carry_out function, that brings the game from one state to another. All of this is in the game folder.
![structure](struct.png)

---
#### Algorithms
Algorithms contains the self-play methods. Currently the only implemented algorithm is the deep learing assisted monte carlo CFR. Due to the complexity of the action space in citadels, the neural network only predicts the value in a given state, while the policy is calculated from scratch.

## Usage <a name = "usage"></a>
1. Install pytorch, and requirements.txt:
```python
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install -r requirements.txt
```
2. Generate test data pickle file
```python
python generate_test_data.py
```
3. Start training
```python
python train_from_scratch.py
```
4. Test against random opponent
```python
python compare_to_random.py 
```

## Training method <a name = "training"></a>

#### MCCFR data generation
For validation I decided to create nodes with well calculated policies and values. I did this by creating games, playing them randomly to completion, then step back 0-20 steps and run the CFR on that state. After this only using the root of the game tree. This ensures that the strategy values settled to a good strategy and node values reflect the reality without being meaningless uniform distributions.

After generating validation data, in a similar fashion I created training data, playing games randomly to completion, then taking 0-100 steps back, and running CFR on that state. The training data is any game node that is backpropagated at least 200 times.

#### Training a neural network to help the algorithm
In the original deep MCCFR the NN helps by making an initial guess at the node values and strategies. In my version I made attempts to get a sequence to probabilty distribution transformer model to do this but It never managed to learn anything really, so I settled with a linear NN making predictions only at the winning probablities and not the strategies.


## Chellenges <a name = "chellenges"></a>
As I understand, MCCFR has a hard time to deal with fully private decisions, like picking the roles at the start of each citadel round. The problem is that the decision doesn't change anything about the public information, what it changes is:
- the picking players role
- the next players options to chose from
- the picking players knowledge based on the role cards he got and the role card he gives forward

All of these are private informations, if I just sample it away I'm going to make the rolepick decison inconsequential.


#### The proposed solution
I treated the whole rolepick phase as a single node where everyone is playing. This means the strategy, the values and the regrets are [player_num, option_num] shaped matrices intsead of [1, option_num] shaped vectors in these kind of nodes. The options in this node a sampled through playing 10 role pick phases randomly, resulting in 10 child nodes. At the end of the role picking, the child nodes are set up by setting up the first roles player (so for example the player who picked the assasin if thats the number 1 role), then sampling roles based on that players knowledge of what the other players could be (this knowledge is based on the orders the players sit, who has the crown and the role cards the player could pick from). The role pick decisions are reflected in this players anticipations of what the other players roles could be.


The strategy matrices are used by taking their weighted average on their first axis and turning them into a [1, option_num] shaped vector just like the rest. To simulate the advantage of picking sooner, the averaging is weighted [6, 5, 4, 3, 2, 1], where 6 is the weight of the crowned player and 1 is the weight of the last player.



## Results
