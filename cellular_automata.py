"""
Artificial Life - Conway's Game of Life
A Cellular Automaton demonstrating emergent complexity and artificial life 
from simple deterministic rules.
"""

import time
import os
import sys

# Windows requires this to support clear screen seamlessly in some terminals
os.system("") 

WIDTH = 50
HEIGHT = 20

def typing_print(text, delay=0.03):
    """Outputs text with a typewriter effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def clear_screen():
    # Uses ANSI escape codes to clear screen and move cursor to top-left
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()

def init_grid():
    """Initializes a grid with various famous shapes."""
    grid = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    
    # Glider (moves diagonally)
    grid[1][2] = 1
    grid[2][3] = 1
    grid[3][1] = 1
    grid[3][2] = 1
    grid[3][3] = 1

    # Blinker (oscillator)
    grid[10][20] = 1
    grid[10][21] = 1
    grid[10][22] = 1

    # Toad (oscillator)
    grid[15][10] = 1
    grid[15][11] = 1
    grid[15][12] = 1
    grid[14][11] = 1
    grid[14][12] = 1
    grid[14][13] = 1

    # Beacon (oscillator)
    grid[5][30] = 1
    grid[5][31] = 1
    grid[6][30] = 1
    grid[6][31] = 1
    grid[7][32] = 1
    grid[7][33] = 1
    grid[8][32] = 1
    grid[8][33] = 1

    # R-pentomino (creates a huge chaotic explosion before settling)
    grid[10][40] = 1
    grid[10][41] = 1
    grid[11][39] = 1
    grid[11][40] = 1
    grid[12][40] = 1

    return grid

def print_grid(grid, generation):
    frame = f"Generation: {generation}\n"
    frame += "-" * (WIDTH + 2) + "\n"
    for y in range(HEIGHT):
        frame += "|"
        for x in range(WIDTH):
            if grid[y][x]:
                frame += "█"
            else:
                frame += " "
        frame += "|\n"
    frame += "-" * (WIDTH + 2) + "\n"
    sys.stdout.write('\033[H' + frame)
    sys.stdout.flush()

def count_neighbors(grid, x, y):
    count = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = (x + dx) % WIDTH, (y + dy) % HEIGHT # Toroidal (wrap-around) universe
            if grid[ny][nx]:
                count += 1
    return count

def update_grid(grid):
    new_grid = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for y in range(HEIGHT):
        for x in range(WIDTH):
            neighbors = count_neighbors(grid, x, y)
            
            # Rule 1: Underpopulation or Overpopulation -> Death
            if grid[y][x] == 1 and (neighbors < 2 or neighbors > 3):
                new_grid[y][x] = 0
            # Rule 2: Survival
            elif grid[y][x] == 1 and (neighbors == 2 or neighbors == 3):
                new_grid[y][x] = 1
            # Rule 3: Reproduction
            elif grid[y][x] == 0 and neighbors == 3:
                new_grid[y][x] = 1
                
    return new_grid

def main():
    clear_screen()
    typing_print("=== 🧬 Artificial Life: Cellular Automata ===", delay=0.04)
    typing_print("Conway's Game of Life demonstrates how complex intelligent-like", delay=0.03)
    typing_print("behavior (Emergence) arises from simple rules.", delay=0.03)
    time.sleep(2)
    clear_screen()
    
    grid = init_grid()
    generation = 0
    
    try:
        while generation <= 150: # Run for 150 generations
            print_grid(grid, generation)
            grid = update_grid(grid)
            generation += 1
            time.sleep(0.08)
    except KeyboardInterrupt:
        pass
        
    print("\n\nEvolution paused. Simulation complete!")

if __name__ == "__main__":
    main()
