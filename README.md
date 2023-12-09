# CITADELS_self_play

## Table of Contents

- [Introduction](#introduction)
- [Structure](#structure)
- [Usage](#usage)
- [Methods](#methods)

## Introduction <a name = "introduction"></a>
This is my hobby project for creating an agent that can play the board game [Citadels](https://www.ultraboardgames.com/citadels/deluxe.php) against itself. The game and the algorithms are both coded from scratch. The goal is to create an agent that can play the game at a high level. I made the simplification that the game is always played by 6 people, and currently only with a set cards and characters.

## Structure <a name = "structure"></a>
#### Game
Game is made out of 3 main classes: 'Game', 'Agent' and 'Option'. Where the game contains the public informations, the agent the private informations, while also having a get_options function that based on the game returns all the options the agent can choose from. The option class contains the information about the option, while also containing a carry_out function, that brings the game from one state to another. All of this is in the game folder.
![structure](struct.png)

---
#### Algorithms
Algorithms contains the self-play methods. Currently the only implemented algorithm is the deep learing assisted monte carlo CFR. Due to the complexity of the action space in citadels, the neural network only predicts the value in a given state, while the policy traditionally.

## Usage <a name = "usage"></a>
First generate test data with generate_test_data.py, then to start from zero run train_from_scratch.py. Still mostly work in progress, but should work.


## Chellenges <a name = "methods"></a>
As I understand, MCCFR has a hard time to deal with fully private decisions, like picking the roles at the start of each citadel round. The problem is that the decision doesn't change anything about the public information, what it changes is:
- the picking players role
- the next players options to chose from
- the picking players knowledge based on the role cards he got and the role card he gives forward

All of these are private informations, if I just sample it away I'm going to make the rolepick decison inconsequential.


#### The proposed solution
I treated the whole rolepick phase as a single node where everyone is playing. This means the strategy, the values and the regrets are [player_num, option_num] shaped matrices intsead of [1, option_num] shaped vectors in these kind of nodes. The options in this node a sampled through playing 10 role pick phases randomly, resulting in 10 child nodes. At the end of the role picking, the child nodes are set up by setting up the first roles player (so for example the player who picked the assasin if thats the number 1 role), then sampling roles based on that players knowledge of what the other players could be (this knowledge is based on the orders the players sit, who has the crown and the role cards the player could pick from). The role pick decisions are reflected in this players anticipations of what the other players roles could be.


The strategy matrices are used by taking their weighted average on their first axis and turning them into a [1, option_num] shaped vector just like the rest. To simulate the advantage of picking sooner, the averaging is weighted [6, 5, 4, 3, 2, 1], where 6 is the weight of the crowned player and 1 is the weight of the last player.