# Blackjack

Training a reinforced learning model to learn to play Blackjack. It is based on Farama's Gymnasium [ToyText](https://gymnasium.farama.org/environments/toy_text/blackjack/) environment.
    
    gymnasium.make("Blackjack-v1")

### Blackjack Rules

Blackjack is a card game where the goal is to beat the dealer by getting cards that total closer to 21 without exceeding it.


The game begins with the dealer having one face-up and one face-down card, while the player has two face-up cards. Cards are drawn from an infinite deck (with replacement).

Card values are as follows:
* Face cards (Jack, Queen, King) are worth 10 points.
* Aces can be worth 11 or 1 point.
* Numerical cards (2-9) are worth their number.

The player can request additional cards (hit) until they decide to stop (stick) or exceed 21 (bust), resulting in an immediate loss.

  After the player sticks, the dealer reveals their facedown card, and draws cards until their sum is 17 or greater. If the dealer goes bust, the player wins.

If neither the player nor the dealer busts, the outcome (win, lose, draw) is decided by whose sum is closer to 21.

 
### Action space
Two possible actions: stick (0) or hit (1).

  

### Observation Space
The observation is a 3-tuple of integers that includes: the player's current sum, the value of the dealer's face-up card (1-10, where 1 is an ace), and whether the player has a usable ace (0 or 1)

### Starting state
|Observation| Min | Max |
|--|--| --|
| Player current sum | 4| 12 |
| Dealer showing card value | 2 | 11 |
| Usable Ace | 0 | 1 |

Note: An ace will always be counted as usable (11) unless it busts the player.

### Reward policy
-   win game: +1
-   lose game: -1
-   draw game: 0
-   win game with natural  blackjack: +1.5 (if  [natural](https://gymnasium.farama.org/environments/toy_text/blackjack/#nat)  is True) +1 (if  [natural](https://gymnasium.farama.org/environments/toy_text/blackjack/#nat)  is False)

 **natural=False**: Determines if an extra reward is given for starting with a natural blackjack (ace and ten, summing to 21).

### End
The finish if the following happens:
	 a)   The player hits and the sum of hand exceeds 21.
	 b)  The player sticks.