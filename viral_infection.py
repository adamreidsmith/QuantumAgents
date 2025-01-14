'''
Agents memory:
    Classical:
        Position
        Direction
    Quantum:
        Dead or not
        Sick or not
        Immune or not
        Healthy or not

Rules:
    - Agents move in a biased random walk.
    - Agents can either be dead, sick, immune, or healthy, but not a combination of those.
    - When sick, agents recover with probability `p_recover`, or die with probability `p_die`.
    - When an agent recovers, it becomes immune or healthy with equal probability.
    - When immune, agents lose their immunity with probability `p_lose_immunity`.
    - When healthy agents are nearby sick agents, they become sick with probability `p_infect`.
    - Healthy agents may randomly contract the infection with probability `p_random_infect`.
    - When healthy/immune agents are nearby other healthy/immune agents, they reproduce with probability `p_reproduce`.
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
    P(y=h0 | x=00, s=h) = 1 - p_random_infect
    P(y=h0 | x=01, s=h) = (1 - p_infect) * (1 - p_random_infect)
    P(y=h0 | x=10, s=h) = (1 - p_reproduce) * (1 - p_random_infect)
    P(y=h0 | x=11, s=h) = (1 - p_infect) * (1 - p_reproduce) * (1 - p_random_infect)

    P(y=h1 | x=10, s=h) = p_reproduce * (1 - p_random_infect)
    P(y=h1 | x=11, s=h) = (1 - p_infect) * p_reproduce * (1 - p_random_infect)

    P(y=s  | x=00, s=h) = p_random_infect
    P(y=s  | x=01, s=h) = p_infect * (1 - p_random_infect) + (1 - p_infect) * p_random_infect + p_infect * p_random_infect
    P(y=s  | x=10, s=h) = p_random_infect
    P(y=s  | x=11, s=h) = p_infect * (1 - p_random_infect) + (1 - p_infect) * p_random_infect + p_infect * p_random_infect

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

    P(y=h0 | x=00, s=s) = p_recover * 0.5
    P(y=h0 | x=01, s=s) = p_recover * 0.5
    P(y=h0 | x=10, s=s) = p_recover * 0.5
    P(y=h0 | x=11, s=s) = p_recover * 0.5

    P(y=i0 | x=00, s=s) = p_recover * 0.5
    P(y=i0 | x=01, s=s) = p_recover * 0.5
    P(y=i0 | x=10, s=s) = p_recover * 0.5
    P(y=i0 | x=11, s=s) = p_recover * 0.5

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
    - The output (where {h0, h1} -> h and {i0, i1} -> i) is the new causal state.
    - That is, `update_rule(x, y, s) = y[0]`
'''

from math import sqrt, log2
import random
from tqdm import tqdm

import pygame
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import partial_trace
from scipy.linalg import logm
from scipy.spatial import KDTree

from agent_encoding_sparse import QuantumAgentEncoder
from constants import *


def f_sq_norm2(a):
    '''Fast square norm for an array of shape (2,).'''

    return a[0] ** 2 + a[1] ** 2


def f_norm2(a):
    '''Fast norm for an array of shape (2,).'''

    return sqrt(f_sq_norm2(a))


def f_c_norm41(a):
    '''Fast norm for a complex array of shape (4,1).'''

    ac = a.conj()
    return sqrt(
        (a[0, 0] * ac[0, 0]).real + (a[1, 0] * ac[1, 0]).real + (a[2, 0] * ac[2, 0]).real + (a[3, 0] * ac[3, 0]).real
    )


def log2m(a):
    '''Base 2 matrix log.'''

    return logm(a) * LOG2_INV


def encode_agent(
    p_lose_immunity: float = 0.1,
    p_recover: float = 0.05,
    p_infect: float = 0.9,
    p_die: float = 0.01,
    p_reproduce: float = 0.01,
    p_random_infect: float = 0.0001,
) -> QuantumAgentEncoder:
    t_probs = {
        ('h0', 'h'): {
            '00': 1.0 - p_random_infect,
            '01': (1.0 - p_infect) * (1.0 - p_random_infect),
            '10': (1.0 - p_reproduce) * (1.0 - p_random_infect),
            '11': (1.0 - p_infect) * (1.0 - p_reproduce) * (1.0 - p_random_infect),
        },
        ('h1', 'h'): {
            '00': 0.0,
            '01': 0.0,
            '10': p_reproduce * (1.0 - p_random_infect),
            '11': p_reproduce * (1.0 - p_random_infect) * (1.0 - p_infect),
        },
        ('s', 'h'): {
            '00': p_random_infect,
            '01': p_infect * (1.0 - p_random_infect) + (1.0 - p_infect) * p_random_infect + p_infect * p_random_infect,
            '10': p_random_infect,
            '11': p_infect * (1.0 - p_random_infect) + (1.0 - p_infect) * p_random_infect + p_infect * p_random_infect,
        },
        ('h0', 'i'): {
            '00': p_lose_immunity,
            '01': p_lose_immunity,
            '10': p_lose_immunity * (1.0 - p_reproduce),
            '11': p_lose_immunity * (1.0 - p_reproduce),
        },
        ('h1', 'i'): {
            '00': 0.0,
            '01': 0.0,
            '10': p_lose_immunity * p_reproduce,
            '11': p_lose_immunity * p_reproduce,
        },
        ('i0', 'i'): {
            '00': 1.0 - p_lose_immunity,
            '01': 1.0 - p_lose_immunity,
            '10': (1.0 - p_lose_immunity) * (1.0 - p_reproduce),
            '11': (1.0 - p_lose_immunity) * (1.0 - p_reproduce),
        },
        ('i1', 'i'): {
            '00': 0.0,
            '01': 0.0,
            '10': (1.0 - p_lose_immunity) * p_reproduce,
            '11': (1.0 - p_lose_immunity) * p_reproduce,
        },
        ('h0', 's'): p_recover * 0.5,
        ('i0', 's'): p_recover * 0.5,
        ('d', 's'): p_die,
        ('s', 's'): 1.0 - p_recover - p_die,
        ('d', 'd'): 1.0,
    }

    def transition_probs(y, x, s):
        '''Given the input `x` and causal state `s`, returns the probability of emmiting output `y`.'''

        return p[x] if isinstance(p := t_probs.get((y, s), 0), dict) else p

    def update_rule(x, y, s):
        '''Given the input, output, and causal state, returns the new causal state.'''

        return y[0]

    inputs = ['00', '01', '10', '11']
    outputs = ['h0', 'h1', 'i0', 'i1', 's', 'd']
    causal_states = ['h', 'i', 's', 'd']

    encoder = QuantumAgentEncoder(
        causal_states=causal_states,
        inputs=inputs,
        outputs=outputs,
        transition_probs=transition_probs,
        update_rule=update_rule,
        method='broyden',
        initialize_jacobian=True,
        tol=1e-11,
        compute_full_unitary=True,
    )
    encoder.encode()

    return encoder


class Agent:
    def __init__(
        self,
        world_size: int,
        step_size: float,
        probabilities: dict[str, float],
        encoder: QuantumAgentEncoder | None = None,
        sick: bool = False,
        mode: str = QUANTUM,
    ):
        '''
        Parameters
        ----------
        world_size : int
            Size of the simulation grid.
        step_size : float
            Distance the agent moves on each step.
        probabilities : dict[str, float]
            Dictionary of probabilities with keys 'p_reproduce', 'p_die', 'p_infect', 'p_lose_immunity',
            'p_recover', and 'p_random_infect'.
        encoder : QuantumAgentEncoder
            Instance of the QuantumAgentEncoder class in which the agent has been encoded. If set to None,
            the agent must be classical.
        sick : bool
            Whether or not the agent is sick. Default: False
        mode : str
            Either 'quantum' or 'classical'. Whether the agent is quantum or classical. Default: 'quantum'
        '''

        self.world_size = world_size
        self.step_size = step_size
        self.p = probabilities
        self.encoder = encoder

        if mode.lower() == QUANTUM:
            self.quantum = True
        elif mode.lower() == CLASSICAL:
            self.quantum = False
        else:
            raise ValueError(f'Invalid mode: {mode}')

        if self.quantum and encoder is None:
            raise ValueError('An encoder must be provided for quantum agents')

        # Initial position of the agent
        self.position = (random.uniform(0, self.world_size), random.uniform(0, self.world_size))

        # Agent travels with a biased random walk
        theta = 2 * np.pi * random.random()
        self.direction = (np.cos(theta), np.sin(theta))

        # Quantum parameters
        self.memory_state = 's' if sick else 'h'

    def move(self):
        '''Move agent in a biased random walk'''

        # Update the direction by rotating the current direction by a random angle between -pi/8 and pi/8.
        # Working with tuples is uglier, but significantly faster than using numpy arrays.
        theta = (random.random() - 0.5) * PI_ON_4
        ct, st = np.cos(theta), np.sin(theta)
        x1, x2 = self.direction
        x1, x2 = (ct * x1 - st * x2, st * x1 + ct * x2)  # self.direction = rotation_matrix @ self.direction
        d_norm = f_norm2((x1, x2))
        self.direction = (x1 / d_norm, x2 / d_norm)  # Ensure the direction remains normalized

        # Update the position to step forward
        self.position = (
            (self.position[0] + self.direction[0] * self.step_size) % self.world_size,
            (self.position[1] + self.direction[1] * self.step_size) % self.world_size,
        )

    def update_classical(self, input_state: str):
        '''Update the state of the agent according to the transition probabilities'''

        q = random.random()
        match self.memory_state:
            case 'h':
                match input_state:
                    case '00':
                        output_state = 's' if q < self.p['p_random_infect'] else 'h0'
                    case '01':
                        output_state = 'h0' if q < (1 - self.p['p_infect']) * (1 - self.p['p_random_infect']) else 's'
                    case '10':
                        p1 = (1 - self.p['p_reproduce']) * (1 - self.p['p_random_infect'])
                        p2 = 1 - self.p['p_random_infect']
                        if q < p1:
                            output_state = 'h0'
                        elif q < p2:
                            output_state = 'h1'
                        else:
                            output_state = 's'
                    case '11':
                        p1 = (1 - self.p['p_infect']) * (1 - self.p['p_reproduce']) * (1 - self.p['p_random_infect'])
                        p2 = (1 - self.p['p_infect']) * self.p['p_reproduce'] * (1 - self.p['p_random_infect'])
                        if q < p1:
                            output_state = 'h0'
                        elif q < p1 + p2:
                            output_state = 'h1'
                        else:
                            output_state = 's'
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
                if q < self.p['p_recover'] / 2:
                    output_state = 'h0'
                elif q < self.p['p_recover']:
                    output_state = 'i0'
                elif q < self.p['p_recover'] + self.p['p_die']:
                    output_state = 'd'
                else:
                    output_state = 's'
            case 'd':
                output_state = 'd'

        self.memory_state = output_state[0]
        return output_state

    def update_quantum(self, input_state: str, check_update_rule: bool = False):
        '''Update the state of the agent via the quantum encoding.'''

        sv = self.encoder.run_compiled_evolution(input_state, self.memory_state)

        # Measure the output state
        outcome, post_measurement_state = sv.measure([2, 3, 4])
        output_state = self.encoder.outputs[int(outcome, base=2)]

        # Trace out the junk, output, and input states to obtain the new memory state
        quantum_memory_state = partial_trace(post_measurement_state, range(7)).to_statevector()._data.reshape(4, 1)

        # Make sure the new memory state agrees with the update rule
        if check_update_rule:
            assert np.allclose(
                quantum_memory_state,
                self.encoder.memory_state_map[self.encoder.update_rule(input_state, output_state, self.memory_state)],
            )

        # Get the value corresponding to the new memory state
        self.memory_state = min(
            self.encoder.memory_state_map.keys(),
            key=lambda k: f_c_norm41(quantum_memory_state - self.encoder.memory_state_map[k]),
        )

        return output_state

    def update(self, input_state: str, check_update_rule: bool = False):
        if self.quantum:
            output_state = self.update_quantum(input_state, check_update_rule)
        else:
            output_state = self.update_classical(input_state)
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
        p_random_infect: float = 0.0001,
        fps: int = 0,
        icon_size: int = 20,
        display: bool = True,
        maxit: int = 0,
        entropy_it: int = 0,
        mode: str = QUANTUM,
        use_tqdm: bool = False,
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
        p_random_infect : float
            The probability a healthy agent is randomly infected on a given iteration.
        fps : int
            Maximum frame rate of the simulation. Set to 0 to run as fast as possible.
        icon_size : int
            The size of the agent icons in the simulation window.
        display : bool
            Whether or not to display a visualization of the simulation.
        maxit : int
            Maximum number of iterations to run. Set to 0 to run indefinitely.
        entropy_it : int
            Iteration at which to begin accumulating stats to compute the entropy.
        mode : str
            Whether the simulation is quantum or classical.  One of 'quantum' or 'classical'.
        use_tqdm : bool
            Whether to display a tqdm progress bar or print a line of stats for each
            iteration. This has no effect if `maxit` is not a positive integer.
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
        self.use_tqdm = use_tqdm
        self.mode = mode.lower()
        assert self.mode in (QUANTUM, CLASSICAL), f'Invalid mode: {mode}'

        self.p = {
            'p_lose_immunity': p_lose_immunity,
            'p_recover': p_recover,
            'p_infect': p_infect,
            'p_die': p_die,
            'p_reproduce': p_reproduce,
            'p_random_infect': p_random_infect,
        }

        self.encoder = (
            encode_agent(p_lose_immunity, p_recover, p_infect, p_die, p_reproduce, p_random_infect)
            if self.mode == QUANTUM
            else None
        )

        # Initialize the set of agents
        self.agents = {
            Agent(
                world_size=self.world_size,
                step_size=self.step_size,
                probabilities=self.p,
                encoder=self.encoder,
                sick=True,
                mode=self.mode,
            )
            for _ in range(n_sick)
        }
        self.agents.update(
            {
                Agent(
                    world_size=self.world_size,
                    step_size=self.step_size,
                    probabilities=self.p,
                    encoder=self.encoder,
                    sick=False,
                    mode=self.mode,
                )
                for _ in range(n_agents - n_sick)
            }
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
        self.density_matrices = (
            {key: state @ state.T.conj() for key, state in self.encoder.memory_state_map.items()}
            if self.mode == QUANTUM
            else None
        )

        self.it = 0

    def step(self):
        self.it += 1

        pos_to_agent = {agent.position: agent for agent in self.agents}
        positions = list(pos_to_agent.keys())
        # Build a KD-Tree to store positions of agents and efficiently find nearby agents
        kdtree = KDTree(positions, leafsize=4) if positions else None

        # Find all the agents which are healthy or immune
        healthy_immune_agent_positions = []
        for i, pos in enumerate(positions):
            agent = pos_to_agent[pos]
            if agent.memory_state in 'sd':
                # Agents which are sick or dead cannot get sick or reproduce, so they are updated immediately with arbitrary input state
                agent.update('00')
            else:
                healthy_immune_agent_positions.append(pos)

        # Query the KD Tree to find all agents within self.infection_radius of all other agents
        nearby = (
            kdtree.query_ball_point(x=healthy_immune_agent_positions, r=self.infection_radius, return_sorted=False)
            if healthy_immune_agent_positions
            else None
        )

        agents_born = set()  # Keep track of born agents
        for i, pos in enumerate(healthy_immune_agent_positions):
            agent = pos_to_agent[pos]  # The agent to update
            nearby_pos_indices = nearby[i]  # Indices of agents near the main agent

            nearby_sick = nearby_healthy_immune = False
            for j in nearby_pos_indices:  # For each agent near the main agent
                other_agent = pos_to_agent[positions[j]]
                if other_agent is agent:
                    # Continue if they are the same agent
                    continue

                # Switch flags if a nearby agent is sick or healthy/immune
                if other_agent.memory_state == 's':
                    nearby_sick = True
                elif other_agent.memory_state in 'ih':
                    nearby_healthy_immune = True

                if nearby_sick and nearby_healthy_immune:
                    break

            # Formulate the input state and update the agent
            input_state = f'{int(nearby_healthy_immune)}{int(nearby_sick)}'
            output_state = agent.update(input_state)

            # If the agent reproduces, add a healthy agent to the simulation
            if output_state in ('h1', 'i1') and len(self.agents) + len(agents_born) < self.max_agents:
                agents_born.add(Agent(self.world_size, self.step_size, self.p, self.encoder, False, self.mode))

        self.n_born += len(agents_born)
        self.agents.update(agents_born)

        # Get the counts of the current agent statuses
        dead = []
        n_sick = n_healthy = n_immune = 0
        for agent in self.agents:
            if self.it > self.entropy_it:
                self.causal_state_occurrence[agent.memory_state] += 1
            match agent.memory_state:
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

        if not self.use_tqdm:
            tallies = [
                f'Iteration: {self.it}',
                f'Agents: {n_total}',
                f'Healthy: {n_healthy}',
                f'Immune: {n_immune}',
                f'Sick: {n_sick}',
                f'Dead: {self.n_dead}',
                f'Born: {self.n_born}',
            ]
            print(' | '.join(tallies))

        self.stats['total'].append(n_total)
        self.stats['healthy'].append(n_healthy)
        self.stats['immune'].append(n_immune)
        self.stats['sick'].append(n_sick)
        self.stats['dead'].append(self.n_dead)
        self.stats['born'].append(self.n_born)

        if self.maxit and self.it >= self.maxit:
            self.close_clicked = True

    def run(self) -> tuple[float, float]:
        if self.display:
            self.create_window()
            if self.use_tqdm and self.maxit:
                for _ in tqdm(range(self.maxit)):
                    if self.fps:
                        pygame.time.Clock().tick(self.fps)  # Set the frame rate to self.fps frames per second
                    self.handle_event()
                    self.step()
                    self.draw()
                    if self.close_clicked:
                        break
            else:
                while not self.close_clicked:  # Until the user closes the window, play a frame
                    if self.fps:
                        pygame.time.Clock().tick(self.fps)  # Set the frame rate to self.fps frames per second
                    self.handle_event()
                    self.step()
                    self.draw()
            pygame.quit()
        else:
            try:
                if self.use_tqdm and self.maxit:
                    for _ in tqdm(range(self.maxit)):
                        self.step()
                        if self.close_clicked:
                            break
                else:
                    while not self.close_clicked:
                        self.step()
            except KeyboardInterrupt:
                pass
        return self.compute_entropy()

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
            p_random_infect=self.p['p_random_infect'],
            fps=self.fps,
            icon_size=self.icon_size,
            display=self.display,
            maxit=self.maxit,
            entropy_it=self.entropy_it,
            mode=self.mode,
            use_tqdm=self.use_tqdm,
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
            x, y = agent.position
            self.surface.blit(
                self.icons['hisd'.index(agent.memory_state)],
                dest=(x - self.icon_size / 2, y - self.icon_size / 2),
            )

    def draw(self):
        if self.surface:
            self.surface.fill(self.bg_colour)
            self.draw_agents()
            pygame.display.update()

    def plot_stats(self, save=False, dpi=300):
        plt.rcParams.update(
            {
                'text.usetex': True,
                'font.family': 'serif',
                'font.serif': ['Computer Modern Roman'],
            }
        )

        # Convert time to years
        x = [d / 365 for d in range(len(self.stats['total']))]

        plt.figure(figsize=(10, 4.4))

        plt.plot(x, self.stats['total'], label='Total', c='b', linewidth=1)
        plt.plot(x, self.stats['healthy'], label='Healthy', c='g', linewidth=1)
        plt.plot(x, self.stats['immune'], label='Immune', c='grey', linewidth=1)
        plt.plot(x, self.stats['sick'], label='Sick', c='r', linewidth=1)
        # plt.plot(x, self.stats['dead'], label='Died', c='m')
        # plt.plot(x, self.stats['born'], label='Born', c='k')
        plt.legend()
        plt.xlabel('Time (years)')
        plt.ylabel('Number of agents')

        plt.tight_layout()

        if save:
            plt.savefig('statistics.png', dpi=dpi)
        else:
            plt.show()

    def compute_entropy(self):
        # if self.mode == QUANTUM:
        #     return self.compute_quantum_entropy()
        # return self.compute_classical_entropy()
        return self.compute_quantum_entropy(), self.compute_classical_entropy()

    def compute_quantum_entropy(self):
        total = sum(self.causal_state_occurrence.values())
        if total == 0:
            return 0
        rho = sum(
            (count / total) * self.density_matrices[state] for state, count in self.causal_state_occurrence.items()
        )

        eigenvalues = np.linalg.eigvals(rho)
        # Any complex part of the eigenvalues should be a result of numerical errors as
        # a valid density matrix always has non-negative real eigenvalues that sum to 1.
        # If not, `rho` is not a valid density matrix.
        assert np.all(np.abs(l.imag) < 1e-15 for l in eigenvalues), 'Eigenvalues have significant imaginary part'
        assert np.all(l.real > -1e-15 for l in eigenvalues), 'Eigenvalues are all non-negative'
        assert np.abs(np.sum(eigenvalues) - 1) < 1e-14, 'Eigenvalues do not sum to 1'
        eigenvalues = (max(float(l.real), 0.0) for l in eigenvalues)
        qx = sum(-l * log2(l) if l != 0 else 0 for l in eigenvalues)

        # qx = float(-(rho @ log2m(rho)).trace().real)
        return qx

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


def run_ebola_sim_realistic(mode: str, plot: bool = False, display: bool = False) -> float:
    # Ebola realistic (days)
    sim = Simulation(
        world_size=400,
        step_size=5,
        n_agents=100,
        n_sick=5,
        max_agents=1000,
        infection_radius=20,  # Highly infectious
        p_lose_immunity=0.000274,  # Chosen s.t. immunity lasts ~10 years
        p_recover=0.05,  # Chosen s.t. disease lasts ~10 days and p_die = p_recover
        p_infect=0.8,  # Highly infectious
        p_die=0.05,  # Chosen s.t. disease lasts ~10 days and p_die = p_recover
        p_reproduce=0.0003,  # Chosen s.t. birth rate ~0.093 births per woman per year
        p_random_infect=1.0 / (365 * 500),  # Chosen s.t. random infection happens in 500 years on average
        display=display,
        maxit=365 * 100,
        entropy_it=100,
        mode=mode,
        use_tqdm=True,
        fps=0,  # Set to 0 to run as fast as possible
    )
    ent = sim.run()
    print(f'Entropy: {ent}')
    if plot:
        sim.plot_stats(False, 400)
    return ent


def run_ebola_sim_unrealistic(mode: str, plot: bool = False, display: bool = False) -> float:
    # Probabilities yielding higher quantum advantage, but unrealistic model
    p = {
        'p_random_infect': 0.22566992770784103,
        'p_infect': 0.6495918662234014,
        'p_recover': 0.3317001281745728,
        'p_die': 0.0017300345176949122,
        'p_reproduce': 0.0037607666625816445,
        'p_lose_immunity': 0.5191374706898385,
    }

    sim = Simulation(
        world_size=400,
        step_size=5,
        n_agents=300,
        n_sick=10,
        max_agents=1000,
        infection_radius=20,
        p_lose_immunity=p['p_lose_immunity'],
        p_recover=p['p_recover'],
        p_infect=p['p_infect'],
        p_die=p['p_die'],
        p_reproduce=p['p_reproduce'],
        p_random_infect=p['p_random_infect'],
        display=display,
        maxit=365 * 100,
        entropy_it=100,
        mode=mode,
        use_tqdm=True,
        fps=0,  # Set to 0 to run as fast as possible
    )
    ent = sim.run()
    print(f'Entropy: {ent}')
    if plot:
        sim.plot_stats()
    return ent


if __name__ == '__main__':
    run_ebola_sim_realistic(QUANTUM, plot=True, display=True)
    # run_ebola_sim_unrealistic(QUANTUM, plot=True, display=True)
