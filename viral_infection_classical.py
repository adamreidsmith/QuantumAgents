'''
Agents memory:
    Position
    Direction
    Dead or not
    Sick or not
    Immune or not
    Healthy or not

Rules:
    - Agents move in a (biased) random walk.
    - Agents can either be dead, sick, immune, or healthy, but not a combination of those.
    - When sick, agents recover and become immune with probability p_recover, or die with probability p_die.
    - When immune, agents lose their immunity with probability p_lose_immunity.
    - When healthy agents are nearby sick agents, they become sick with probability p_infect.
    - When healthy/immune agents are nearby other healthy/immune agents, they reproduce with probability p_reproduce.
    - Dead agents are removed from the simulation.

Notations:
    Outputs: {
        h0: healthy and don't reproduce
        h1: healthy and reproduce
        i0: immune and don't reproduce
        i1: immune and reproduce
        s: sick 
        d: dead
    }

    Inputs: {
        00: No sick or healthy agents
        01: Nearby sick agents
        10: Nearby healthy/immune agents
        11: Nearby sick and healthy/immune agents
    }

    Causal states: {
        h: healthy
        i: immune
        s: sick 
        d: dead
    }

Transition probabilities:
    P(y=h0 | x=00, s=h) = 1
    P(y=h0 | x=01, s=h) = 1 - p_infect
    P(y=h0 | x=10, s=h) = 1 - p_reproduce
    P(y=h0 | x=11, s=h) = (1 - p_infect) * (1 - p_reproduce)

    P(y=h1 | x=10, s=h) = p_reproduce
    P(y=h1 | x=11, s=h) = (1 - p_infect) * p_reproduce

    P(y=s  | x=01, s=h) = p_infect
    P(y=s  | x=11, s=h) = p_infect

    P(y=h0 | x=00, s=i) = p_lose_immunity
    P(y=h0 | x=01, s=i) = p_lose_immunity
    P(y=h0 | x=10, s=i) = p_lose_immunity * (1 - p_reproduce)
    P(y=h0 | x=11, s=i) = p_lose_immunity * (1 - p_reproduce)

    P(y=h1 | x=10, s=i) = p_lose_immunity * p_reproduce
    P(y=h1 | x=11, s=i) = p_lose_immunity * p_reproduce

    P(y=i0 | x=00, s=i) = 1 - p_lose_immunity
    P(y=i0 | x=01, s=i) = 1 - p_lose_immunity
    P(y=i0 | x=10, s=i) = (1 - p_lose_immunity) * (1 - p_reproduce)
    P(y=i0 | x=11, s=i) = (1 - p_lose_immunity) * (1 - p_reproduce)

    P(y=i1 | x=10, s=i) = (1 - p_lose_immunity) * p_reproduce
    P(y=i1 | x=11, s=i) = (1 - p_lose_immunity) * p_reproduce

    P(y=i0 | x=00, s=s) = p_recover
    P(y=i0 | x=01, s=s) = p_recover
    P(y=i0 | x=10, s=s) = p_recover
    P(y=i0 | x=11, s=s) = p_recover

    P(y=d  | x=00, s=s) = p_die
    P(y=d  | x=01, s=s) = p_die
    P(y=d  | x=10, s=s) = p_die
    P(y=d  | x=11, s=s) = p_die

    P(y=s  | x=00, s=s) = 1 - p_recover - p_die
    P(y=s  | x=10, s=s) = 1 - p_recover - p_die
    P(y=s  | x=01, s=s) = 1 - p_recover - p_die
    P(y=s  | x=11, s=s) = 1 - p_recover - p_die

    P(y=d  | x=00, s=d) = 1
    P(y=d  | x=01, s=d) = 1
    P(y=d  | x=10, s=d) = 1
    P(y=d  | x=11, s=d) = 1

    All others are zero.

Update rule:
    - The output with {h0, h1} -> h and {i0, i1} -> i is the new causal state.
    - lambda(x, y, s) = y[0]
'''

import random
from math import sqrt, log2

import pygame
import numpy as np
import matplotlib.pyplot as plt

from constants import *


def f_norm2(a):
    '''Fast norm for an array of shape (2,)'''

    return sqrt(a[0] ** 2 + a[1] ** 2)


def f_c_norm(a):
    '''Fast norm for a complex array of shape (n,)'''

    aac = (a * a.conj()).real
    return sqrt(sum(aac))


class Agent:
    def __init__(self, world_size: int, step_size: float, probabilities: dict, sick: bool = False):
        '''
        Parameters
        ----------
        world_size : int
            Size of the simulation grid.
        step_size : float
            Distance the agent moves on each step.
        probabilities : dict
            Dictionary of probabilities, keyed by strings 'p_reproduce', 'p_die',
            'p_infect', 'p_lose_immunity', and 'p_recover'.
        sick : bool
            Whether or not the agent is sick. Default: False
        '''

        self.world_size = world_size
        self.step_size = step_size
        self.p = probabilities

        # Initial position of the agent
        self.position = np.random.rand(2) * self.world_size

        # Agent travels with a biased random walk
        theta = 2 * np.pi * random.random()
        self.direction = np.array([[np.cos(theta)], [np.sin(theta)]])

        # Quantum parameters
        self.state = 's' if sick else 'h'

    def move(self):
        '''Move the agent in a biased random walk.'''

        # Update the direction by rotating the current direction by a random angle between -pi/8 and pi/8
        theta = (np.random.rand() - 0.5) * np.pi / 4
        ct, st = np.cos(theta), np.sin(theta)
        rotation = np.array([[ct, -st], [st, ct]])
        self.direction = rotation @ self.direction
        self.direction /= f_norm2(self.direction)  # Ensure the direction remains normalized

        # Update the position to step forward
        self.position += self.direction.reshape(2) * self.step_size
        self.position %= self.world_size

    def update_state(self, input_state: str):
        '''Update the state of the agent according to the transition probabilities.'''

        q = random.random()
        match self.state:
            case 'h':
                match input_state:
                    case '00':
                        output_state = 'h0'
                    case '01':
                        output_state = 's' if q < self.p['p_infect'] else 'h0'
                    case '10':
                        output_state = 'h1' if q < self.p['p_reproduce'] else 'h0'
                    case '11':
                        if q < self.p['p_infect']:
                            output_state = 's'
                        elif q > 1 - (1 - self.p['p_infect']) * self.p['p_reproduce']:
                            output_state = 'h1'
                        else:
                            output_state = 'h0'
            case 'i':
                match input_state:
                    case '00' | '01':
                        output_state = 'h0' if q < self.p['p_lose_immunity'] else 'i0'
                    case '10' | '11':
                        p1 = self.p['p_lose_immunity'] * (1 - self.p['p_reproduce'])
                        p2 = self.p['p_lose_immunity'] * self.p['p_reproduce']
                        p3 = (1 - self.p['p_lose_immunity']) * self.p['p_reproduce']
                        if q < p1:
                            output_state = 'h0'
                        elif q < p1 + p2:
                            output_state = 'h1'
                        elif q < p1 + p2 + p3:
                            output_state = 'i1'
                        else:
                            output_state = 'i0'
            case 's':
                if q < self.p['p_recover']:
                    output_state = 'i0'
                elif q > 1 - self.p['p_die']:
                    output_state = 'd'
                else:
                    output_state = 's'
            case 'd':
                output_state = 'd'

        self.state = output_state[0]
        return output_state

    def update(self, input_val: str):
        output_state = self.update_state(input_val)
        self.move()
        return output_state


class Simulation:
    def __init__(
        self,
        world_size: int,
        step_size: float,
        n_agents: int,
        n_sick: int,
        max_agents: int = 300,
        infection_radius: float = 20,
        p_lose_immunity: float = 0.1,
        p_recover: float = 0.05,
        p_infect: float = 0.9,
        p_die: float = 0.01,
        p_reproduce: float = 0.01,
        fps: int = 0,
        icon_size: int = 20,
        display: bool = True,
        maxit: int = 0,
        entropy_it: int = 50,
    ):
        '''
        Parameters
        ----------
        world_size : int
            Simulation is run on a board of size (world_size, world_size) pixels.
        step_size : float
            The distance, in pixels, agents move each iteration.
        n_agents : int
            The number of agents at the start of the simulation.
        n_sick : int
            The number of agents which are sick at the start of the simulation.
        max_agents : int
            The maximum number of agents to allow in the simulation.
        infection_radius : float,
            The maximum distance, in pixels, a healthy agent must be from a sick agent
            to contract an infection.
        p_lose_immunity : float
            The probability an immune agent loses its immunity on a given iteration.
        p_recover : float
            The probability a sick agent recovers on a given iteration.
        p_infect : float
            The probability a healthy agent near a sick agent contracts an infection on
            a given iteration.
        p_die : float
            The probability a sick agent dies on a given iteration.
        p_reproduce : float
            The probability a healthy/immune agent near another healthy/immune agent
            reproduces on a given iteration.
        fps : int
            Maximum frame rate of the simulation. Set to 0 to run as fast as possible.
        icon_size : int
            The size of the agent icons in the simulation window.
        display : bool
            Whether or not to display a visualization of the simulation.
        '''

        self.world_size = world_size
        self.step_size = step_size
        self.max_agents = max_agents
        self.infection_radius = infection_radius
        self.display = display
        self.maxit = maxit
        self.entropy_it = entropy_it
        self.n_agents_start = n_agents
        self.n_sick_start = n_sick

        self.p = {
            'p_lose_immunity': p_lose_immunity,
            'p_recover': p_recover,
            'p_infect': p_infect,
            'p_die': p_die,
            'p_reproduce': p_reproduce,
        }

        # Initialize the agents
        self.agents = {Agent(self.world_size, self.step_size, self.p, sick=True) for _ in range(n_sick)}
        self.agents.update(
            {Agent(self.world_size, self.step_size, self.p, sick=False) for _ in range(n_agents - n_sick)}
        )

        self.n_dead = self.n_born = 0
        self.stats = {'total': [], 'healthy': [], 'immune': [], 'sick': [], 'dead': [], 'born': []}

        # Visualization attributes
        self.surface = None
        self.close_clicked = False
        self.fps = fps
        self.bg_colour = pygame.Color('black')
        self.icon_size = icon_size
        self.icons = {
            icon: pygame.transform.scale(pygame.image.load('icons/' + path), (self.icon_size, self.icon_size))
            for icon, path in zip([1, 0, 2], ['person_grey.png', 'person_green.png', 'person_red.png'])
        }

        self.causal_state_occurrence = {s: 0 for s in 'hisd'}

        self.it = 0

    def step(self):
        self.it += 1
        # n**2 algorithm, but should be fast enough for this application
        computed_dists = {}
        agents_born = set()
        for agent in self.agents:
            if agent.state in 'sd':
                # If the agent is sick or dead, it cannot reproduce or be infected
                output_state = agent.update('00')
                continue
            nearby_sick = nearby_healthy_immune = False
            for other_agent in self.agents:
                if agent == other_agent:
                    continue

                # Cache the distance so we don't have to compute each one twice
                id1, id2 = id(other_agent), id(agent)
                dist = computed_dists.get((id2, id1), None) or computed_dists.get((id1, id2), None)
                if dist is None:
                    dist = f_norm2(agent.position - other_agent.position)
                    computed_dists[(id1, id2)] = dist

                if dist <= self.infection_radius:
                    if other_agent.state == 's':
                        nearby_sick = True
                    elif other_agent.state in 'ih':
                        nearby_healthy_immune = True

                    if nearby_sick and nearby_healthy_immune:
                        break

            # Formulate the input state
            input_state = f'{int(nearby_healthy_immune)}{int(nearby_sick)}'
            output_state = agent.update(input_state)

            # If the agent reproduces, add a healthy agent to the simulation
            if output_state in ('h1', 'i1') and len(self.agents) + len(agents_born) < self.max_agents:
                agents_born.add(Agent(self.world_size, self.step_size, self.p, sick=False))

        self.n_born += len(agents_born)
        self.agents.update(agents_born)

        # Get the counts of the current agent statuses
        dead = []
        n_sick = n_healthy = n_immune = 0
        for agent in self.agents:
            if self.it > self.entropy_it:
                self.causal_state_occurrence[agent.state] += 1
            match agent.state:
                case 'd':
                    dead.append(agent)
                    self.n_dead += 1
                case 's':
                    n_sick += 1
                case 'i':
                    n_immune += 1
                case 'h':
                    n_healthy += 1
        n_total = len(self.agents)

        # Remove dead agents
        for agent in dead:
            self.agents.remove(agent)

        print(
            f'{self.it} | Agents: {n_total} | Healthy: {n_healthy} | Immune: {n_immune} | Sick: {n_sick} | Dead: {self.n_dead} | Born: {self.n_born}'  # | C = {self.compute_classical_entropy()}'
        )

        self.stats['total'].append(n_total)
        self.stats['healthy'].append(n_healthy)
        self.stats['immune'].append(n_immune)
        self.stats['sick'].append(n_sick)
        self.stats['dead'].append(self.n_dead)
        self.stats['born'].append(self.n_born)

        if self.maxit and self.it >= self.maxit:
            self.close_clicked = True

        if n_total == n_healthy and not self.display:
            self.reset()

    def run(self):
        if self.display:
            self.create_window()
            while not self.close_clicked:  # Until the user closes the window, play a frame
                if self.fps:
                    pygame.time.Clock().tick(self.fps)  # Set the frame rate to self.fps frames per second
                self.handle_event()
                self.step()
                self.draw()
            pygame.quit()
        else:
            # try:
            #     while not self.close_clicked:
            #         self.step()
            # except KeyboardInterrupt:
            #     pass
            while not self.close_clicked:
                self.step()
        return self.compute_classical_entropy()

    def reset(self):
        self.__init__(
            world_size=self.world_size,
            step_size=self.step_size,
            n_agents=self.n_agents_start,
            n_sick=self.n_sick_start,
            max_agents=self.max_agents,
            infection_radius=self.infection_radius,
            p_lose_immunity=self.p['p_lose_immunity'],
            p_recover=self.p['p_recover'],
            p_infect=self.p['p_infect'],
            p_die=self.p['p_die'],
            p_reproduce=self.p['p_reproduce'],
            fps=self.fps,
            icon_size=self.icon_size,
            display=self.display,
            maxit=self.maxit,
            entropy_it=self.entropy_it,
        )

    def create_window(self):
        '''Open a window on the display and return its surface.'''

        title = 'Viral Infection'
        pygame.init()
        surface = pygame.display.set_mode((self.world_size, self.world_size))
        pygame.display.set_caption(title)
        self.surface = surface

    def handle_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close_clicked = True
                break

    def draw_agents(self):
        for agent in self.agents:
            self.surface.blit(
                self.icons[['h', 'i', 's', 'd'].index(agent.state)],
                dest=agent.position - self.icon_size / 2,
            )

    def draw(self):
        if self.surface:
            self.surface.fill(self.bg_colour)
            self.draw_agents()
            pygame.display.update()

    def plot_stats(self):
        plt.figure(figsize=(15, 7))

        plt.plot(self.stats['total'], label='Total', c='b')
        plt.plot(self.stats['healthy'], label='Healthy', c='g')
        plt.plot(self.stats['immune'], label='Immune', c='grey')
        plt.plot(self.stats['sick'], label='Sick', c='r')
        # plt.plot(self.stats['dead'], label='Died', c='m')
        # plt.plot(self.stats['born'], label='Born', c='k')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Number of agents')

        plt.show()

    def compute_classical_entropy(self):
        total = sum(self.causal_state_occurrence.values())
        if total == 0:
            return 0
        cx = 0
        for count in self.causal_state_occurrence.values():
            p = count / total
            if p > 0:
                cx -= p * log2(p)
        return cx


if __name__ == '__main__':
    # trials = 10
    # entropy = []
    # for _ in range(trials):
    #     sim = Simulation(
    #         world_size=600,
    #         step_size=5,
    #         n_agents=200,
    #         n_sick=10,
    #         max_agents=300,
    #         infection_radius=12,
    #         p_lose_immunity=0.05,
    #         p_recover=0.02,
    #         p_infect=0.65,
    #         p_die=0.005,
    #         p_reproduce=0.018,
    #         display=False,
    #         maxit=20000,
    #         entropy_it=2000,
    #     )
    #     entropy.append(sim.run())
    #     print(entropy)
    #     print(sum(entropy) / len(entropy))

    # sim = Simulation(
    #     world_size=400,
    #     step_size=5,
    #     n_agents=200,
    #     n_sick=20,
    #     max_agents=1000,
    #     infection_radius=20,
    #     p_lose_immunity=0.3,
    #     p_recover=0.3,
    #     p_infect=0.7,
    #     p_die=0.3,
    #     p_reproduce=0.15,
    #     display=False,
    #     maxit=1000,
    #     entropy_it=50,
    # )
    # # 1.8060416647664803

    sim = Simulation(
        world_size=400,
        step_size=5,
        n_agents=1000,
        n_sick=100,
        max_agents=10000,
        infection_radius=25,
        p_lose_immunity=0.35,
        p_recover=0.5,
        p_infect=0.6,
        p_die=0.5,
        p_reproduce=0.12,
        display=False,
        maxit=1000,
        entropy_it=50,
    )

    sim.run()
    sim.plot_stats()
    print(sim.compute_classical_entropy())

'''
[1.3065033030771716, 1.3158801108881146, 1.3275959780639761, 1.3439256930092522, 1.3141329895447997, 1.3114020931256103, 1.3377332685221561, 1.3390316258220576, 1.282623882368068, 1.309782100549223]
1.3188611044970429

[1.3406125174449266, 1.3377074173561443, 1.309374287015186, 1.3069026045933474, 1.3093179139034843, 1.3285442112401948, 1.3077632414452465, 1.323743343792384, 1.3011134971775882, 1.321687378267038]
1.318676641223554
'''
