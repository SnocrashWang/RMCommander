import pygame
import pymunk

print("pygame version:", pygame.__version__)
print("pymunk version:", pymunk.__version__)

# 初始化测试
pygame.init()
space = pymunk.Space()
print("Pygame and Pymunk initialized successfully!")