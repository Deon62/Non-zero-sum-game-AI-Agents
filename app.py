import nashpy as nash
import numpy as np

# Define the payoff matrices
# Rows: Agent A's strategies (Share, Withhold)
# Columns: Agent B's strategies (Share, Withhold)

A_payoffs = np.array([[8, 3],
                      [9, 2]])

B_payoffs = np.array([[8, 9],
                      [3, 2]])

# Create the game
game = nash.Game(A_payoffs, B_payoffs)

# Display the game
print("Game Matrix (A, B):")
print(game)

# Find Nash equilibria
print("\nNash Equilibria:")
equilibria = list(game.support_enumeration())
for eq in equilibria:
    print(f"Agent A Strategy: {eq[0]}, Agent B Strategy: {eq[1]}")
