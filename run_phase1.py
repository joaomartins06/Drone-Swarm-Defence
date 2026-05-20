import pygame
pygame.init()
screen = pygame.display.set_mode((600, 600))
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
pygame.quit()