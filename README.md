# Bipartite Euclidean Matching Problem
Algorithm to solve the bipartite euclidean matching problem by simulating each point in both point sets as magnets with opposing polarity (by set). Run simulation, allow points to find best match. Using matplotlib.animation to generate inline visualization of this process.

## `magnet_alignment.py`
This algorithm uses physical magnetism equation to simulate magnets with real attraction.

## `kabsch_alignment.py`
This algorithm uses Kabsch algorithm to simulate geometric attraction with rigid transformation (no physical magnet equation used) and Hungarian algorithm to compute the bipartite matching.