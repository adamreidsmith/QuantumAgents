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

from math import sqrt

import pygame
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import partial_trace

from agent_encoding import QuantumAgentEncoder


def f_norm2(a):
    '''Fast norm for an array of shape (2,)'''

    return sqrt(a[0] ** 2 + a[1] ** 2)


def f_c_norm(a):
    '''Fast norm for a complex array of shape (n,)'''

    aac = (a * a.conj()).real
    return sqrt(sum(aac))


def encode_agent(p_lose_immunity=0.1, p_recover=0.05, p_infect=0.9, p_die=0.01, p_reproduce=0.01):
    def transition_probs(y, x, s):
        probs = {
            ('h0', 'h'): {
                '00': 1.0,
                '01': 1.0 - p_infect,
                '10': 1.0 - p_reproduce,
                '11': (1.0 - p_infect) * (1.0 - p_reproduce),
            },
            ('h1', 'h'): {
                '00': 0.0,
                '01': 0.0,
                '10': p_reproduce,
                '11': (1.0 - p_infect) * p_reproduce,
            },
            ('s', 'h'): {
                '00': 0.0,
                '01': p_infect,
                '10': 0.0,
                '11': p_infect,
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
            ('i0', 's'): p_recover,
            ('d', 's'): p_die,
            ('s', 's'): 1.0 - p_recover - p_die,
            ('d', 'd'): 1.0,
        }

        if (y, s) in probs:
            return p[x] if isinstance(p := probs[(y, s)], dict) else p
        return 0

    def update_rule(x, y, s):
        '''Given the input, output, and causal state, returns the new causal state.'''
        return y[0]

    inputs = ['00', '01', '10', '11']
    outputs = ['h0', 'h1', 'i0', 'i1', 's', 'd']
    causal_states = ['h', 'i', 's', 'd']

    input_encodings = QuantumAgentEncoder.encode_vals(list(range(len(inputs))))
    output_encodings = QuantumAgentEncoder.encode_vals(list(range(len(outputs))))

    encoder = QuantumAgentEncoder(
        causal_states=causal_states,
        inputs=inputs,
        outputs=outputs,
        transition_probs=transition_probs,
        update_rule=update_rule,
        input_encodings=input_encodings,
        output_encodings=output_encodings,
        numerical=False,
        verbose=False,
    )
    encoder.encode()

    return encoder


class Agent:
    def __init__(self, encoder: QuantumAgentEncoder, world_size: int, step_size: float, sick: bool = False):
        '''
        Parameters
        ----------
        encoder : QuantumAgentEncoder
            Instance of the QuantumAgentEncoder class in which the agent has been encoded.
        world_size : int
            Size of the simulation grid.
        step_size : float
            Distance the agent moves on each step.
        sick : bool
            Whether or not the agent is sick. Default: False
        '''

        self.world_size = world_size
        self.step_size = step_size
        self.encoder = encoder

        # Initial position of the agent
        self.position = np.random.rand(2) * self.world_size

        # Agent travels with a biased random walk
        theta = 2 * np.pi * np.random.rand()
        self.direction = np.array([[np.cos(theta)], [np.sin(theta)]])

        # Quantum parameters
        self.memory_state = 's' if sick else 'h'

    def move(self):
        '''Move agent in a biased random walk.'''

        # Update the direction by rotating the current direction by a random angle between -pi/8 and pi/8
        theta = (np.random.rand() - 0.5) * np.pi / 4
        ct, st = np.cos(theta), np.sin(theta)
        rotation = np.array([[ct, -st], [st, ct]])
        self.direction = rotation @ self.direction
        self.direction /= f_norm2(self.direction)  # Ensure the direction remains normalized

        # Update the position to step forward
        self.position += self.direction.reshape(2) * self.step_size
        self.position %= self.world_size

    def update_quantum(self, input_state: str):
        sv = self.encoder.run_evolution_circuit_manual(self.memory_state, input_state)

        # Measure the output state
        outcome, post_measurement_state = sv.measure([2, 3, 4])
        output_state = self.encoder.outputs[int(outcome, base=2)]

        # Trace out the junk, output, and input states to obtain the new memory state
        quantum_memory_state = np.asarray(partial_trace(post_measurement_state, range(7)).to_statevector()).reshape(
            4, 1
        )

        # Make sure the new memory state agrees with the update rule
        assert np.allclose(
            quantum_memory_state,
            self.encoder.memory_state_map[self.encoder.update_rule(input_state, output_state, self.memory_state)],
        )

        # Get the value corresponding to the new memory state
        self.memory_state = min(
            self.encoder.memory_state_map.keys(),
            key=lambda k: f_c_norm(quantum_memory_state - self.encoder.memory_state_map[k]),
        )

        return output_state

    def update(self, quantum_input: str):
        output_state = self.update_quantum(quantum_input)
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

        self.encoder = encode_agent(p_lose_immunity, p_recover, p_infect, p_die, p_reproduce)

        self.agents = {Agent(self.encoder, self.world_size, self.step_size, sick=True) for _ in range(n_sick)}
        self.agents.update(
            {Agent(self.encoder, self.world_size, self.step_size, sick=False) for _ in range(n_agents - n_sick)}
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

    def step(self):
        # n**2 algorithm, but should be fast enough for this application (bottleneck is the quantum simulation)
        computed_dists = {}
        agents_born = set()
        for agent in self.agents:
            if agent.memory_state in 'sd':
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
                    if other_agent.memory_state == 's':
                        nearby_sick = True
                    elif other_agent.memory_state in 'ih':
                        nearby_healthy_immune = True

                    if nearby_sick and nearby_healthy_immune:
                        break

            # Formulate the input state
            input_state = f'{int(nearby_healthy_immune)}{int(nearby_sick)}'
            output_state = agent.update(input_state)

            # If the agent reproduces, add a healthy agent to the simulation
            if output_state in ('h1', 'i1') and len(self.agents) + len(agents_born) < self.max_agents:
                agents_born.add(Agent(self.encoder, self.world_size, self.step_size, sick=False))

        self.n_born += len(agents_born)
        self.agents.update(agents_born)

        # Get the counts of the current agent statuses
        dead = []
        n_sick = n_healthy = n_immune = 0
        for agent in self.agents:
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

        print(
            f'Agents: {n_total} | Healthy: {n_healthy} | Immune: {n_immune} | Sick: {n_sick} | Dead: {self.n_dead} | Born {self.n_born}'
        )

        self.stats['total'].append(n_total)
        self.stats['healthy'].append(n_healthy)
        self.stats['immune'].append(n_immune)
        self.stats['sick'].append(n_sick)
        self.stats['dead'].append(self.n_dead)
        self.stats['born'].append(self.n_born)

        if n_healthy == n_total:
            self.close_clicked = True

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
            try:
                while True:
                    self.step()
            except KeyboardInterrupt:
                pass

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
                self.icons[self.encoder.causal_states.index(agent.memory_state)],
                dest=agent.position - self.icon_size / 2,
            )

    def draw(self):
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


if __name__ == '__main__':
    sim = Simulation(
        world_size=500,
        step_size=5,
        n_agents=200,
        n_sick=10,
        max_agents=300,
        infection_radius=12,
        p_lose_immunity=0.05,
        p_recover=0.02,
        p_infect=0.65,
        p_die=0.005,
        p_reproduce=0.018,
        display=False,
    )
    sim.run()
    sim.plot_stats()
