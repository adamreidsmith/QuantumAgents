import random
import time
import hashlib
from pathlib import Path
from copy import deepcopy
from itertools import product
from collections import defaultdict
from operator import itemgetter
from math import log2

import dill
import numpy as np
from numpy.linalg import eigvals
from scipy.special import factorial
from qiskit.quantum_info import partial_trace
from tqdm import tqdm

from agent_encoding_sparse import QuantumAgentEncoder, PartialQuantumAgentEncoder

from constants import NORTH, SOUTH, NS_TURNING, NS_STRAIGHT, EW


class TrafficLightBase:
    '''
    Base class for an input-output process representing a traffic light.
    ```
                     4     3
                 ||     :     |     :     ||
                 ||     :     |     :     ||
                 ||     :     |     :     ||
                 ||     :     |     :     ||
                 ||  ↓  :  ↳  |     :     ||
                 ||  ↓  :  ↳  |     :     ||
     =============                         =============
                                            ← ⬐ ← ⬐     2
     -------------                         -------------
    5     ⬏ → ⬏ →
     =============                         =============
                 ||     :     |  ↰  :  ↑  ||
                 ||     :     |  ↰  :  ↑  ||
                 ||     :     |     :     ||
                 ||     :     |     :     ||
                 ||     :     |     :     ||
                 ||     :     |     :     ||
                                 0     1
    ```
    '''

    def __init__(self, lane_capacity: int, light_times: tuple[int, ...]):
        self.lane_capacity = lane_capacity
        assert len(light_times) == len(set(light_times)), f'`light_times` has duplicate values: {light_times}'
        self.light_times = tuple(sorted(light_times))
        light_configs = (NORTH, SOUTH, NS_TURNING, NS_STRAIGHT, EW)

        self.inputs = tuple(product(range(self.lane_capacity + 1), repeat=6))
        self.outputs = tuple(product(light_configs, light_times))
        self.causal_states = light_configs

        two_lane_capacity_split = np.array_split(range(2 * self.lane_capacity + 1), len(self.light_times))
        two_lane_capacity_split = list(map(lambda s: set(map(int, s)), two_lane_capacity_split))

        def transition_probs(y, x, s):
            '''
            Given the input `x` and causal state `s`, returns the probability of emmiting output `y`.
            That is, number of vehicles in each lane and the current light configuration, returns the
            probability of the new light configuration and light duration.
            '''

            VPL = x  # vehicles per lane
            new_light_config, light_time = y
            current_light_config = s

            def adjust_prob_for_light_time(prob: float, lanes_released: int):
                '''Takes in the probability and the two lanes to be released.'''

                if lanes_released == NORTH:
                    n_vehicles_released = VPL[0] + VPL[1]
                elif lanes_released == SOUTH:
                    n_vehicles_released = VPL[3] + VPL[4]
                elif lanes_released == NS_TURNING:
                    n_vehicles_released = VPL[0] + VPL[3]
                elif lanes_released == NS_STRAIGHT:
                    n_vehicles_released = VPL[1] + VPL[4]
                elif lanes_released == EW:
                    n_vehicles_released = VPL[2] + VPL[5]
                else:
                    raise ValueError(f'Invalid lane idenitifer: {lanes_released}')

                for i, capacity_set in enumerate(two_lane_capacity_split):
                    if n_vehicles_released in capacity_set:
                        idx = i
                        break
                else:
                    raise ValueError('More than `2 * self.lane_capacity` or less than 0 vehicles released')

                if light_time == self.light_times[idx]:
                    return prob * 0.75
                return prob * 0.25 / (len(self.light_times) - 1)

            if current_light_config == EW:
                # We stay in EW if there are no waiting vehicles
                if all(VPL[n] == 0 for n in (0, 1, 3, 4)):
                    if new_light_config == EW:
                        return adjust_prob_for_light_time(1.0, EW)
                    return 0.0

                # When transitioning out of EW, we probabilistically transition to one of NORTH
                # SOUTH, NS_TURNING, or NS_STRAIGHT. With (70%, 20%, 10%, 0%) probability, we
                # transition to the configuration with the (most, 2nd most, 3rd most, 4th most)
                # waiting vehicles.
                n_north = VPL[0] + VPL[1]
                n_south = VPL[3] + VPL[4]
                n_turning = VPL[0] + VPL[3]
                n_straight = VPL[1] + VPL[4]
                lane_totals = [n_north, n_south, n_turning, n_straight]
                light_configs = [NORTH, SOUTH, NS_TURNING, NS_STRAIGHT]

                lane_totals, light_configs = zip(*sorted(zip(lane_totals, light_configs), key=itemgetter(0)))

                if new_light_config == light_configs[-1]:
                    return adjust_prob_for_light_time(0.7, light_configs[-1])
                if new_light_config == light_configs[-2]:
                    return adjust_prob_for_light_time(0.2, light_configs[-2])
                if new_light_config == light_configs[-3]:
                    return adjust_prob_for_light_time(0.1, light_configs[-3])
                return 0.0

            if current_light_config == NORTH:
                # We stay in NORTH if there are no waiting vehicles
                if all(VPL[n] == 0 for n in (2, 3, 4, 5)):
                    if new_light_config == NORTH:
                        return adjust_prob_for_light_time(1.0, NORTH)
                    return 0.0

                # If E is full or W is full, transition to EW
                if VPL[2] == self.lane_capacity or VPL[5] == self.lane_capacity:
                    if new_light_config == EW:
                        return adjust_prob_for_light_time(1.0, EW)
                    return 0.0

                # If more vehicles are waiting in EW than SOUTH, transition to EW
                if VPL[2] + VPL[5] > VPL[3] + VPL[4]:
                    if new_light_config == EW:
                        return adjust_prob_for_light_time(1.0, EW)
                    return 0.0

                # If SOUTH_STRAIGHT > SOUTH_TURNING, transition to NS_STRAIGHT w/ 80%
                # and NS_TURNING w/ 10% and SOUTH w/ 10%
                if VPL[4] > VPL[3]:
                    if new_light_config == NS_STRAIGHT:
                        return adjust_prob_for_light_time(0.8, NS_STRAIGHT)
                    if new_light_config in (NS_TURNING, SOUTH):
                        return adjust_prob_for_light_time(0.1, new_light_config)
                    return 0.0

                # If SOUTH_STRAIGHT < SOUTH_TURNING, transition to NS_TURNING w/ 80%
                # and NS_STRAIGHT w/ 10% and SOUTH w/ 10%
                if VPL[4] < VPL[3]:
                    if new_light_config == NS_TURNING:
                        return adjust_prob_for_light_time(0.8, NS_TURNING)
                    if new_light_config in (NS_STRAIGHT, SOUTH):
                        return adjust_prob_for_light_time(0.1, new_light_config)
                    return 0.0

                # If SOUTH_STRAIGHT = SOUTH_TURNING, transition to SOUTH w/ 50%
                # and NS_STRAIGHT w/ 25% and NS_TURNING w/ 25%
                if VPL[4] == VPL[3]:
                    if new_light_config == SOUTH:
                        return adjust_prob_for_light_time(0.5, SOUTH)
                    if new_light_config in (NS_STRAIGHT, NS_TURNING):
                        return adjust_prob_for_light_time(0.25, new_light_config)
                    return 0.0

            if current_light_config == SOUTH:
                # We stay in SOUTH if there are no waiting vehicles
                if all(VPL[n] == 0 for n in (0, 1, 2, 5)):
                    if new_light_config == SOUTH:
                        return adjust_prob_for_light_time(1.0, SOUTH)
                    return 0.0

                # If E is full or W is full, transition to EW
                if VPL[2] == self.lane_capacity or VPL[5] == self.lane_capacity:
                    if new_light_config == EW:
                        return adjust_prob_for_light_time(1.0, EW)
                    return 0.0

                # If more vehicles are waiting in EW than NORTH, transition to EW
                if VPL[2] + VPL[5] > VPL[0] + VPL[1]:
                    if new_light_config == EW:
                        return adjust_prob_for_light_time(1.0, EW)
                    return 0.0

                # If NORTH_STRAIGHT > NORTH_TURNING, transition to NS_STRAIGHT w/ 80%
                # and NS_TURNING w/ 10% and NORTH w/ 10%
                if VPL[1] > VPL[0]:
                    if new_light_config == NS_STRAIGHT:
                        return adjust_prob_for_light_time(0.8, NS_STRAIGHT)
                    if new_light_config in (NS_TURNING, NORTH):
                        return adjust_prob_for_light_time(0.1, new_light_config)
                    return 0.0

                # If NORTH_STRAIGHT < NORTH_TURNING, transition to NS_TURNING w/ 80%
                # and NS_STRAIGHT w/ 10% and NORTH w/ 10%
                if VPL[1] < VPL[0]:
                    if new_light_config == NS_TURNING:
                        return adjust_prob_for_light_time(0.8, NS_TURNING)
                    if new_light_config in (NS_STRAIGHT, NORTH):
                        return adjust_prob_for_light_time(0.1, new_light_config)
                    return 0.0

                # If NORTH_STRAIGHT = NORTH_TURNING, transition to NORTH w/ 50%
                # and NS_STRAIGHT w/ 25% and NS_TURNING w/ 25%
                if VPL[1] == VPL[0]:
                    if new_light_config == NORTH:
                        return adjust_prob_for_light_time(0.5, NORTH)
                    if new_light_config in (NS_STRAIGHT, NS_TURNING):
                        return adjust_prob_for_light_time(0.25, new_light_config)
                    return 0.0

            if current_light_config == NS_STRAIGHT:
                # We stay in NS_STRAIGHT if there are no waiting vehicles
                if all(VPL[n] == 0 for n in (0, 2, 3, 5)):
                    if new_light_config == NS_STRAIGHT:
                        return adjust_prob_for_light_time(1.0, NS_STRAIGHT)
                    return 0.0

                # If E is full or W is full, transition to EW
                if VPL[2] == self.lane_capacity or VPL[5] == self.lane_capacity:
                    if new_light_config == EW:
                        return adjust_prob_for_light_time(1.0, EW)
                    return 0.0

                # If more vehicles are waiting in EW than NS_TURNING, transition to EW
                if VPL[2] + VPL[5] > VPL[0] + VPL[3]:
                    if new_light_config == EW:
                        return adjust_prob_for_light_time(1.0, EW)
                    return 0.0

                # If SOUTH_TURNING > NORTH_TURNING, transition to SOUTH w/ 80%
                # and NS_TURNING w/ 10% and NORTH w/ 10%
                if VPL[3] > VPL[0]:
                    if new_light_config == SOUTH:
                        return adjust_prob_for_light_time(0.8, SOUTH)
                    if new_light_config in (NS_TURNING, NORTH):
                        return adjust_prob_for_light_time(0.1, new_light_config)
                    return 0.0

                # If SOUTH_TURNING < NORTH_TURNING, transition to NORTH w/ 80%
                # and NS_TURNING w/ 10% and SOUTH w/ 10%
                if VPL[3] < VPL[0]:
                    if new_light_config == NORTH:
                        return adjust_prob_for_light_time(0.8, NORTH)
                    if new_light_config in (NS_TURNING, SOUTH):
                        return adjust_prob_for_light_time(0.1, new_light_config)
                    return 0.0

                # If SOUTH_TURNING = NORTH_TURNING, transition to NS_TURNING w/ 50%
                # and NORTH w/ 25% and SOUTH w/ 25%
                if VPL[3] == VPL[0]:
                    if new_light_config == NS_TURNING:
                        return adjust_prob_for_light_time(0.5, NS_TURNING)
                    if new_light_config in (SOUTH, NORTH):
                        return adjust_prob_for_light_time(0.25, new_light_config)
                    return 0.0

            if current_light_config == NS_TURNING:
                # We stay in NS_TURNING if there are no waiting vehicles
                if all(VPL[n] == 0 for n in (1, 2, 4, 5)):
                    if new_light_config == NS_TURNING:
                        return adjust_prob_for_light_time(1.0, NS_TURNING)
                    return 0.0

                # If E is full or W is full, transition to EW
                if VPL[2] == self.lane_capacity or VPL[5] == self.lane_capacity:
                    if new_light_config == EW:
                        return adjust_prob_for_light_time(1.0, EW)
                    return 0.0

                # If more vehicles are waiting in EW than NS_STRAIGHT, transition to EW
                if VPL[2] + VPL[5] > VPL[1] + VPL[4]:
                    if new_light_config == EW:
                        return adjust_prob_for_light_time(1.0, EW)
                    return 0.0

                # If SOUTH_STRAIGHT > NORTH_STRAIGHT, transition to SOUTH w/ 80%
                # and NS_STRAIGHT w/ 10% and NORTH w/ 10%
                if VPL[4] > VPL[1]:
                    if new_light_config == SOUTH:
                        return adjust_prob_for_light_time(0.8, SOUTH)
                    if new_light_config in (NS_STRAIGHT, NORTH):
                        return adjust_prob_for_light_time(0.1, new_light_config)
                    return 0.0

                # If SOUTH_STRAIGHT < NORTH_STRAIGHT, transition to NORTH w/ 80%
                # and NS_STRAIGHT w/ 10% and SOUTH w/ 10%
                if VPL[4] < VPL[1]:
                    if new_light_config == NORTH:
                        return adjust_prob_for_light_time(0.8, NORTH)
                    if new_light_config in (NS_STRAIGHT, SOUTH):
                        return adjust_prob_for_light_time(0.1, new_light_config)
                    return 0.0

                # If SOUTH_STRAIGHT = NORTH_STRAIGHT, transition to NS_STRAIGHT w/ 50%
                # and NORTH w/ 25% and SOUTH w/ 25%
                if VPL[4] == VPL[1]:
                    if new_light_config == NS_STRAIGHT:
                        return adjust_prob_for_light_time(0.5, NS_STRAIGHT)
                    if new_light_config in (SOUTH, NORTH):
                        return adjust_prob_for_light_time(0.25, new_light_config)
                    return 0.0

            raise RuntimeError(f'No probability assigned for args: {x, y, s}')

        # # Check the transition probabilities for validity
        # for x in self.inputs:
        #     for s in self.causal_states:
        #         total_prob = sum(transition_probs(y, x, s) for y in self.outputs)
        #         assert abs(total_prob - 1) < 1e-15, 'Invalid transition probabilities'

        self.transition_probs = transition_probs

        def update_rule(x, y, s):
            '''
            Given the input `x`, output `y`, and causal state `s`, returns the new causal state.
            That is, given the number of vehicles in each lane, the light configuration, light times,
            and previous light configuration, returns the light configuration, which is just the
            first element of the output.
            '''
            return y[0]

        self.update_rule = update_rule

        # Start in the NS_STRAIGHT state
        self.current_causal_state = NS_STRAIGHT

    def step(self, input: tuple[int, ...]) -> tuple[str, int]:
        raise NotImplementedError


class QuantumTrafficLight(TrafficLightBase):
    '''A quantum input-output process representing a traffic light.'''

    def __init__(self, lane_capacity: int, light_times: tuple[int, ...]):
        super().__init__(lane_capacity, light_times)

        self.encoder = PartialQuantumAgentEncoder(
            causal_states=self.causal_states,
            inputs=self.inputs,
            outputs=self.outputs,
            transition_probs=self.transition_probs,
            update_rule=self.update_rule,
            tol=1e-11,
        )
        self.encoder.encode()

        # Variables necessary for statevector measurement
        self.output_state_qubits = range(self.encoder.n_qubits_output_tape)
        self.non_memory_state_qubits = range(self.encoder.n_qubits_output_tape + self.encoder.n_qubits_input_tape)
        self.memory_state_shape = next(iter(self.encoder.memory_state_map.values())).shape
        self.dense_memory_state_map = {k: v.toarray() for k, v in self.encoder.memory_state_map.items()}

    def step(self, input: tuple[int, ...], check_update_rule: bool = True) -> tuple[str, int]:
        '''
        At each step the traffic light acts as follows:
            - The traffic light receives inputs:
                - The number of vehicles in each lane
            - The traffic light outputs:
                - Green or red for each lane
                - The duration of the light before the next change
            - The traffic light needs to store:
                - The previous color of the light for each lane

        Inputs:
            tuple[int * 6] {number of vehicles in each lane}

        Outputs:
            tuple[str, int] ({red or green for each lane}, {time of light})

        Causal States:
            str {red or green for each lane}
        '''

        # Apply the unitary: U|s>|x>|0>|0>
        sv = self.encoder.run_compiled_evolution(input, self.current_causal_state)

        # Measure the output qubits
        outcome, sv = sv.measure(self.output_state_qubits)
        output = self.outputs[int(outcome, base=2)]

        # Trace out the junk, output, and input states to obtain the new memory state
        quantum_memory_state = (
            partial_trace(sv, self.non_memory_state_qubits).to_statevector()._data.reshape(self.memory_state_shape)
        )

        # Check that the new causal state agrees with the update rule
        if check_update_rule:
            assert np.allclose(
                quantum_memory_state,
                self.dense_memory_state_map[self.update_rule(input, output, self.current_causal_state)],
            ), 'Update rule not satisfied'

        self.current_causal_state = min(
            self.encoder.memory_state_map.keys(),
            key=lambda k: np.linalg.norm(quantum_memory_state - self.dense_memory_state_map[k], ord='fro'),
        )

        return output


class ClassicalTrafficLight(TrafficLightBase):
    '''A classical input-output process representing a traffic light.'''

    def step(self, input: tuple[int, ...]) -> tuple[str, int]:
        '''
        At each step the traffic light acts as follows:
            - The traffic light receives inputs:
                - The number of vehicles in each lane
            - The traffic light outputs:
                - Green or red for each lane
                - The duration of the light before the next change
            - The traffic light needs to store:
                - The previous color of the light for each lane

        Inputs:
            tuple[int * 6] {number of vehicles in each lane}

        Outputs:
            tuple[str, int] ({red or green for each lane}, {time of light})

        Causal States:
            str {red or green for each lane}
        '''

        # Use transition probabilities to sample an output
        probabilities = [self.transition_probs(y, input, self.current_causal_state) for y in self.outputs]
        output = random.choices(self.outputs, weights=probabilities)[0]

        # Update the casual state with the update rule
        self.current_causal_state = self.update_rule(input, output, self.current_causal_state)

        return output


class FixedTimeTrafficLight:
    def __init__(self, light_times: tuple[int, int, int]):
        # `light_times` is a tuple of three ints corrsponding to the average number of
        # vehicles that can leave, repsectively, the east-west, north-south turning,
        # and north-south straight lanes while the light is green
        self.light_times = light_times
        assert len(light_times) == 3, 'Must have exactly 3 light times for FixedTimeTrafficLight'
        self.state = NS_STRAIGHT

    def step(self, _):
        if self.state == NS_STRAIGHT:
            self.state = EW
            return EW, self.light_times[0]

        if self.state == EW:
            self.state = NS_TURNING
            return NS_TURNING, self.light_times[1]

        self.state = NS_STRAIGHT
        return NS_STRAIGHT, self.light_times[2]


class MaximalTrafficLight:

    def __init__(self, light_times: tuple[int]):
        self.light_times = light_times

    def step(self, input: tuple[int, ...]) -> tuple[str, int]:
        '''
        Inputs:
            tuple[int * 6] {number of vehicles in each lane}

        Outputs:
            tuple[str, int] ({red or green for each lane}, {time of light})

        Causal States:
            str {red or green for each lane}

        Sets the lanes to green that allow the most vehicles to exit the intersection
        '''

        VPL = input

        lane_counts = {
            NS_STRAIGHT: (VPL[1], VPL[4]),
            NS_TURNING: (VPL[0], VPL[3]),
            NORTH: (VPL[0], VPL[1]),
            SOUTH: (VPL[3], VPL[4]),
            EW: (VPL[2], VPL[5]),
        }

        light_config = max(lane_counts, key=lambda x: sum(lane_counts.get(x)))
        n_vehicles = lane_counts[light_config]

        # Choose the minimal duration such that the max queue length in the green lanes
        # is less than or equal to the duration, or the maximal duration if there is no
        # duration large enough.
        duration = self.light_times[0]
        for light_time in self.light_times[1:]:
            if max(n_vehicles) <= duration:
                break
            duration = light_time

        return light_config, duration


class LaneAgent:
    '''A quantum input-output process representing a traffic lane.'''

    def __init__(self, lane_capacity: int, light_times: tuple[int, ...], ed_typical: int):
        # The maximum number of vehicles that can be in the lane
        self.lane_capacity = lane_capacity
        assert self.lane_capacity >= 4, 'vehicle capacity >= 4 is required for the current transition probabilities'

        # The possible light dirations
        self.light_times = light_times

        # The typical number of vehicles entering the lane each step
        self.ed_typical = ed_typical

        inputs = list(product(('r', 'g'), light_times))
        outputs = list(range(lane_capacity + 1))
        causal_states = list(range(lane_capacity + 1))

        # This is a distribution over integers `n` in [0, self.lane_capacity] and denotes the probability
        # of `n` vehicles entering the lane during the duration of the light
        def get_entering_distribution(duration):
            # Poisson distribution centered on ed_effective = ed_typical * duration / max_duration
            ed_effective = ed_typical * duration / self.light_times[-1]
            dist = np.arange(self.lane_capacity + 1)
            dist = np.exp(-ed_effective) * ed_effective**dist / factorial(dist)
            dist /= dist.sum()
            return dist.tolist()

        # This is a distribution over integers `n` in [0, self.lane_capacity] and denotes the probability
        # of `n` vehicles exiting the lane during the duration of the light
        def get_exiting_distribution(duration, sigma=2):
            # `duration` gives the mean number of vehicles that can exit the lane during the light cycle.
            # This is a discretized normal distribution with mean `duration` and standard deviation `sigma`
            dist = np.arange(self.lane_capacity + 1)
            dist = np.exp(-((dist - duration) ** 2) / (2 * sigma**2))
            dist /= dist.sum()
            return dist.tolist()

        def transition_probs(y, x, s):
            '''
            Given the input `x` and causal state `s`, returns the probability of emmiting output `y`.
            That is, given the light color, light duration, and current number of vehicles in the lane,
            returns the probability of having `y` vehicles remaining in the lane at the end of the step.
            '''

            color, duration = x
            n_current = s

            entering_distribution = get_entering_distribution(duration)

            if color == 'r':
                # If the light is red, the output probabilities (for a fixed n_current) are goverened
                # solely by the entering distribution
                y_probs = [0.0] * (self.lane_capacity + 1)
                for i in range(n_current, self.lane_capacity):
                    y_probs[i] = entering_distribution[i - n_current]
                y_probs[-1] = sum(entering_distribution[self.lane_capacity - n_current :])
                return y_probs[y]

            # If the light is green, we account for the entering and exiting vehicles
            exiting_distribution = get_exiting_distribution(duration)
            y_probs = defaultdict(float)
            for n_entering in range(self.lane_capacity + 1):
                for n_exiting in range(self.lane_capacity + 1):
                    n_remaining = min(max(n_current + n_entering - n_exiting, 0), self.lane_capacity)
                    y_probs[n_remaining] += entering_distribution[n_entering] * exiting_distribution[n_exiting]

            return y_probs[y]

        # # Check the transition probabilities for validity
        # for x in inputs:
        #     for s in causal_states:
        #         total_prob = sum(transition_probs(y, x, s) for y in outputs)
        #         assert abs(total_prob - 1) < 1e-14, 'Invalid transition probabilities'

        def update_rule(x, y, s):
            '''
            Given the input, output, and causal state, returns the new causal state.
            This is just the number of vehicles remaining in the lane, which is the same as the output.
            '''
            return y

        self.encoder = QuantumAgentEncoder(
            causal_states=causal_states,
            inputs=inputs,
            outputs=outputs,
            transition_probs=transition_probs,
            update_rule=update_rule,
            tol=1e-11,
            compute_full_unitary=False,
            clean=True,
        )
        self.encoder.encode()

        # Lanes start out with no vehicles
        self.current_causal_state = 0

        # Variables necessary for statevector measurement
        self.output_state_qubits = range(
            self.encoder.n_qubits_junk_tape, self.encoder.n_qubits_junk_tape + self.encoder.n_qubits_output_tape
        )
        self.non_memory_state_qubits = range(
            self.encoder.n_qubits_junk_tape + self.encoder.n_qubits_output_tape + self.encoder.n_qubits_input_tape
        )
        self.memory_state_shape = next(iter(self.encoder.memory_state_map.values())).shape

        self.density_matrices = {k: state @ state.T.conj() for k, state in self.encoder.memory_state_map.items()}

    def step(self, input: tuple[str, int], check_update_rule: bool = False):
        '''
        At each step, the lane acts as follows:
            - The lane receives inputs:
                - Whether or not the light is green for the lane
                - The duration of the light
            - The lane outputs:
                - The number of vehicles left in the lane at the end of the step
                    - This data is sent to the traffic light
            - The lane needs to store:
                - The number of vehicles left in the lane at the end of the step
            - The lane decides internally:
                - The number of vehicles entering the lane during the step

        Inputs:
            tuple[str, int] ({red or green}, {time of light})

        Outputs:
            int {number of vehicles remaining in the lane}

        Causal states:
            int {number of vehicles remaining in the lane at end of previous step}
        '''

        # Apply the unitary: U|s>|x>|0>|0>
        sv = self.encoder.run_compiled_evolution(input, self.current_causal_state)

        # Measure the output qubits
        outcome, sv = sv.measure(self.output_state_qubits)
        # The outputs are just integers {0, ..., N} corresponding to states {|0>, ..., |N>}, so
        # we can just call `int` on the measurement outcome to convert the measurement to a
        # classical output state
        output = self.encoder.outputs[int(outcome, base=2)]

        # Trace out the junk, output, and input states to obtain the new memory state
        quantum_memory_state = (
            partial_trace(sv, self.non_memory_state_qubits).to_statevector()._data.reshape(self.memory_state_shape)
        )

        # Check that the new causal state agrees with the update rule
        if check_update_rule:
            assert np.allclose(
                quantum_memory_state,
                self.encoder.memory_state_map[self.encoder.update_rule(input, output, self.current_causal_state)],
            ), 'Update rule not satisfied'

        self.current_causal_state = min(
            self.encoder.memory_state_map.keys(),
            key=lambda k: np.linalg.norm(quantum_memory_state - self.encoder.memory_state_map[k], ord='fro'),
        )

        return output


class IntersectionSim:
    AGENT_CACHE = Path.cwd() / '_agent_cache'

    def __init__(self, lane_capacity: int, light_times: tuple[int, ...], arrival_rates: tuple[float, float, float]):
        '''Initialize a quantum ABM intersection simulation.'''

        self.lane_capacity = lane_capacity

        self.light = MaximalTrafficLight(light_times)

        # Arguments for each lane agent type
        ns_t_agent_args = (lane_capacity, light_times, arrival_rates[1])
        ns_s_agent_args = (lane_capacity, light_times, arrival_rates[2])
        ew_agent_args = (lane_capacity, light_times, arrival_rates[0])

        self.lanes: list[LaneAgent] = []
        for args in (ns_t_agent_args, ns_s_agent_args, ew_agent_args):
            hsh = hashlib.sha256(str(args).encode('utf-8'), usedforsecurity=False).hexdigest()
            agent_path = self.AGENT_CACHE / f'{hsh}.pkl'
            if agent_path.exists():
                with agent_path.open('rb') as f:
                    self.lanes.append(dill.load(f))
            else:
                self.lanes.append(LaneAgent(*args))
                with agent_path.open('wb') as f:
                    dill.dump(self.lanes[-1], f)
        for i in range(3):
            self.lanes.append(deepcopy(self.lanes[i]))

        self.queue_lengths = [[] for _ in range(6)]
        self.causal_state_occurrences = [defaultdict(int) for _ in range(6)]

    def run(self, iterations: int, display: bool = True, pause_time: float = 1.0):
        # Initially, no vehicles in each lane
        vehicle_counts = (0,) * 6
        for it in tqdm(range(iterations), desc='Simulating intersection', disable=display):
            # Update the traffic light
            light_config, light_time = self.light.step(vehicle_counts)

            if display:
                self.display(it, vehicle_counts, light_config)
                time.sleep(pause_time)

            # Update each lane
            vehicle_counts = tuple(lane.step((light_config[i], light_time)) for i, lane in enumerate(self.lanes))

            # Update stats from the iteration
            for i, lane in enumerate(self.lanes):
                self.causal_state_occurrences[i][lane.current_causal_state] += 1
                self.queue_lengths[i].append((light_config[i], light_time, vehicle_counts[i]))

            if display:
                self.display(it, vehicle_counts, light_config)
                time.sleep(pause_time)
        return self.compute_entropies()

    def compute_entropies(self):
        '''Compute the von Neumann and Shannon entorpies of the lane agents.'''

        quantum_entropies = []
        classical_entropies = []
        for i, lane in enumerate(self.lanes):
            total = sum(self.causal_state_occurrences[i].values())
            rho = sum(
                (count / total) * lane.density_matrices[s] for s, count in self.causal_state_occurrences[i].items()
            )
            # `rho` may not be full rank, and thus has not matrix log, so we have to use the eigenvalue
            # definition of the Von Neumann entropy
            # qx = -(rho @ log2m(rho)).trace()

            eigenvalues = eigvals(rho)
            # Any complex part of the eigenvalues should be a result of numerical errors as
            # a valid density matrix always has non-negative real eigenvalues that sum to 1.
            # If not, `rho` is not a valid density matrix.
            assert np.all(np.abs(l.imag) < 1e-15 for l in eigenvalues), 'Eigenvalues have significant imaginary part'
            assert np.all(l.real > -1e-15 for l in eigenvalues), 'Eigenvalues are all non-negative'
            assert np.abs(np.sum(eigenvalues) - 1) < 1e-14, 'Eigenvalues do not sum to 1'
            eigenvalues = (max(float(l.real), 0.0) for l in eigenvalues)
            qx = sum(-l * log2(l) if l != 0 else 0 for l in eigenvalues)
            quantum_entropies.append(qx)

            cx = 0
            for count in self.causal_state_occurrences[i].values():
                p = count / total
                if p > 0:
                    cx -= p * log2(p)
            classical_entropies.append(cx)
        return quantum_entropies, classical_entropies

    def display(self, it: int, vehicle_counts: tuple[int], light_config: str):
        '''Print the intersection to stdout.'''

        R = '\u001b[31mX\u001b[0m'
        G = '\u001b[32mO\u001b[0m'

        VPL = vehicle_counts

        north = lambda sb, s, t: (f'{" " * sb}||  {"↓" if s else ' '}  :  {"↳" if t else ' '}  |     :     ||\n')
        south = lambda sb, s, t: f'{" " * sb}||     :     |  {"↰" if s else ' '}  :  {"↑" if t else ' '}  ||\n'
        ew_n_edge = lambda c1, c2: (
            f'{"=" * (2 * self.lane_capacity + 1)}   {G if c1 == 'g' else R}     '
            f'{G if c2 == 'g' else R}{" " * 15}{"=" * (2 * self.lane_capacity + 1)}\n'
        )
        ew_s_edge = lambda c1, c2: (
            f'{"=" * (2 * self.lane_capacity + 1)}{" " * 15}{G if c1 == 'g' else R}     '
            f'{G if c2 == 'g' else R}   {"=" * (2 * self.lane_capacity + 1)}\n'
        )
        ew_cl = f'{"-" * (2 * self.lane_capacity + 1)}{" " * 25}{"-" * (2 * self.lane_capacity + 1)}\n'
        west = lambda n, c: f'{" " * (2 * self.lane_capacity - 2 * n)}{" →" * n}  {G if c == 'g' else R}\n'
        east = lambda n, c: f'{" " * (2 * self.lane_capacity + 24)}{G if c == 'g' else R}  {"← " * n}\n'

        s = f'Iteration: {it}\n'
        for l in range(self.lane_capacity, 0, -1):
            s += north(2 * self.lane_capacity, l <= VPL[4], l <= VPL[3])
        s += ew_n_edge(light_config[4], light_config[3])
        s += east(VPL[2], light_config[2])
        s += ew_cl
        s += west(VPL[5], light_config[5])
        s += ew_s_edge(light_config[0], light_config[1])
        for l in range(self.lane_capacity):
            s += south(2 * self.lane_capacity, l < VPL[0], l < VPL[1])

        print(s)


def plot_max_queue_dists(queue_lengths):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(
        {
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
        }
    )

    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
    for i in range(3):
        # Max queue length between green lights
        queue = queue_lengths[i] + queue_lengths[i + 3]
        max_queue_length_before_green = []
        for j in range(len(queue) - 1):
            if queue[j][0] == 'r' and queue[j + 1][0] == 'g':
                max_queue_length_before_green.append(queue[j][2])
        sns.kdeplot(
            max_queue_length_before_green,
            ax=axs[0],
            label=rf'Lanes {i + 1} \& {i + 4}',
            common_norm=True,
            bw_adjust=1.55,
            linewidth=1,
        )
        # plt.hist(max_queue_length_before_green, bins=range(31), alpha=0.5)

        # Wait times between green lights
        wait_times = []
        n = 0
        for cycle in queue:
            if cycle[0] == 'r':
                n += cycle[1]
            elif n:
                wait_times.append(n)
                n = 0

        # bincounts, _ = np.histogram(wait_times, bins=np.asarray(list(range(122))) - 0.5)
        # x = np.arange(len(bincounts))
        # msk = bincounts > 0
        # axs[1].scatter(x[msk], bincounts[msk])
        # axs[1].hist(wait_times, bins=np.asarray(list(range(122))) - 0.5)
        sns.kdeplot(
            wait_times,
            ax=axs[1],
            label=rf'Lanes {i + 1} \& {i + 4}',
            common_norm=True,
            bw_adjust=1.7,
            linewidth=1,
        )

    axs[0].set_xlabel('Queue length')
    axs[1].set_xlabel('Wait time')
    axs[0].set_ylabel('')
    axs[1].set_ylabel('')
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    # plt.savefig('queue_dists.png', dpi=400)
    plt.show()


if __name__ == '__main__':
    lane_capacity = 30
    # Light times are given as the number of vehicles than can exit lane in the lane while it is green
    # light_times = tuple(round(t * lane_capacity) for t in (0.4, 0.55, 0.7, 0.85, 1.0))
    light_times = tuple(round(t * 20) for t in (0.4, 0.55, 0.7, 0.85, 1.0))
    # Rates at which vehicles arrive at the EW, NS Turing, and NS Straight lanes, respectively.
    arrival_rates = (4, 6, 8)

    sim = IntersectionSim(lane_capacity, light_times, arrival_rates)
    n = 250_000
    ql_path = f'_data_cache/ql_{hashlib.sha256(str((n, lane_capacity, *light_times, *arrival_rates)).encode('utf-8')).hexdigest()}.pkl'
    ent = sim.run(n, False, 0)

    with open(ql_path, 'wb') as f:
        dill.dump(sim.queue_lengths, f)

    with open(ql_path, 'rb') as f:
        queue_lengths = dill.load(f)
    plot_max_queue_dists(queue_lengths)
