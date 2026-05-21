import pygame
import numpy as np
from sim.core import Vec2, IX, IY
from sim.core.types import Measurement
from sim.entities.target import Target
from sim.entities.radar import Radar
from sim.estimation.ekf import EKF
from sim.core.world import World   



class PygameRenderer:

    def __init__(self, width: int = 800, height: int = 800, world_center: Vec2 = Vec2(0, 0),
                 meters_per_pixel: float = 5.0, bg_color: tuple = (10, 10, 20)) -> None:
        #length and width of the pygame window in pixels
        self.width = width
        self.height = height
        #set the center of the world
        self.world_center = world_center
        #how many meters un each pixel
        self.meters_per_pixel = meters_per_pixel
        #background color for the window
        self.bg_color = bg_color

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Drone Swarm vs Air Defense — Phase 1")
        self.font = pygame.font.SysFont("monospace", 14)


    def world_to_screen(self, world_pos: Vec2) -> tuple[int, int]:
        #this just converts world coordinates (in meters) to pixel coordinates
        x = int((world_pos.x - self.world_center.x) / self.meters_per_pixel + self.width / 2)
        y = int((self.world_center.y - world_pos.y) / self.meters_per_pixel + self.height / 2)
        return (x, y)
    

    def draw_target(self, target: Target) -> None:

        #convert world position to screen position
        px, py = self.world_to_screen(target.position)
        #draw a circle on the screen with radius of 5 pixels
        pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 5)
        
        #I am going to also add a line that shows the direction of the target
        #computes the direction where it is heading using the heading angle
        heading = target.state_view.heading_rad
        line_length_m = 20.0
        end_world = Vec2(target.position.x + line_length_m * np.cos(heading),
                        target.position.y + line_length_m * np.sin(heading))
        #convert this endpoint into screen coordinates
        end_px, end_py = self.world_to_screen(end_world)

        #draw a line from the center of the target to the endpoint
        pygame.draw.line(self.screen, (255, 255, 255), (px, py), (end_px, end_py), 2)

    
    def draw_radar(self, radar: Radar) -> None:
        #convert world position to screen position
        px, py = self.world_to_screen(radar.position)
        #draw a triangle for the radar
        size = 8
        triangle = [(px,py - size),(px - size,py + size),(px + size,   py + size)]
        pygame.draw.polygon(self.screen, (0, 255, 255), triangle)

        #To be generalized when we have multiple radars
        #draw a circle where inside pfa > 0.5 and outside pfa < 0.5
        if radar.pfa > 0 and radar.pfa < 1 and radar.target.rcs > 0:
            #compute radius
            r50 = (np.log(0.5) / np.log(radar.pfa) * radar.C * radar.target.rcs) ** 0.25
            #turn it into a radius in pixels
            r50_px = int(r50 / self.meters_per_pixel)
            pygame.draw.circle(self.screen, (0, 80, 80), (px, py), r50_px, 1)


    def draw_measurement(self, radar: Radar) -> None:
        #if there is no measurement, do nothing
        if radar.last_measurement is None:
            return
        
        #get the measurement in cartesian coordinates (meters)
        meas_pos = radar.last_measurement.to_cartesian(radar.position)
        #turn it into screen coordinates
        px, py = self.world_to_screen(meas_pos)

        #put a cross '+' in the measurement position
        size = 5
        pygame.draw.line(self.screen, (255, 50, 50), (px - size, py), (px + size, py), 2)
        pygame.draw.line(self.screen, (255, 50, 50), (px, py - size), (px, py + size), 2)

    
    def draw_ekf(self, ekf: EKF) -> None:
        #draw the estimated position as a green circle
        est_pos = Vec2(ekf.state_view.x, ekf.state_view.y)
        px, py = self.world_to_screen(est_pos)
        pygame.draw.circle(self.screen, (50, 255, 50), (px, py), 5)

        #also draw the uncertainty ellipse
        cov = ekf.covariance
        P_xy = cov[np.ix_([IX, IY], [IX, IY])]

        #compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(P_xy)
        #angle of the major axis (first eigenvector)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        #pygame is a bit weird drawsing tilted elipses, so we need to fo this :/
        #compute the rotation matrix using the angle
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a],
                        [sin_a,  cos_a]])
        #sample the number of points
        N = 64
        points = []
        for t in np.linspace(0, 2 * np.pi, N, endpoint=False):
            #compute normal elipse (angle = 0) for 3 std
            local = np.array([3 * np.sqrt(np.maximum(eigvals, 0)) * np.cos(t),
                              3 * np.sqrt(np.maximum(eigvals, 0)) * np.sin(t)])
            #rotate it
            rotated = rot @ local
            #turn this point into a Vec2 ( we sum the center of the ellipse)
            world_pt = Vec2(est_pos.x + rotated[0], est_pos.y + rotated[1])
            #append it as a screen coordinates
            points.append(self.world_to_screen(world_pt))

        pygame.draw.lines(self.screen, (50, 255, 50), True, points, 1)

    
    def draw(self, world: World, ekf: EKF) -> None:
        #clear the previous frame
        self.screen.fill(self.bg_color)

        #loop through the world entities
        for entity in world.entities:
            #if it is radar, draw a radar and its measurement
            if isinstance(entity, Radar):
                self.draw_radar(entity)
                #notice this will plot the last measurement of the radar
                self.draw_measurement(entity)

            #if it is a target, it will draw a target
            if isinstance(entity, Target):
                self.draw_target(entity)

        #draw the EKF estimate and uncertainty
        self.draw_ekf(ekf)

        #draw the text info about the simulation
        self.draw_text(world, ekf)

        #update the display
        pygame.display.flip()
        #keep the window responsive
        pygame.event.pump()


    def draw_text(self, world: World, ekf: EKF) -> None:
        t = world.clock.t
        sv = ekf.state_view

        #find the target remaining waypoints
        remaining = 0
        for entity in world.entities:
            if isinstance(entity, Target):
                remaining = len(entity.waypoints)
                break
        
        #lines to be displayed 
        lines = [f"t = {t:.1f} s",
                f"est pos = ({sv.x:.1f}, {sv.y:.1f}) m",
                f"est speed = {sv.speed:.1f} m/s",
                f"waypoints left = {remaining}",]
        color = (220, 220, 220)
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, color)
            self.screen.blit(surf, (8, 8 + i * 16))


    def should_quit(self) -> bool:
        #check if the user has closed the window or pressed escape
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return True
        return False
    
    def close(self) -> None:
        #closes the simulation
        pygame.quit()
        
            