import numpy as np
from perlin_noise import PerlinNoise


def generate_perlin_noise_map(
        width: int,
        height: int,
        octaves: int
        ) -> np.ndarray:
    noise = PerlinNoise(octaves=octaves)
    noise_map = np.array([[noise([i / width, j / height]) for j in range(width)] for i in range(height)])
    noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
    return noise_map
