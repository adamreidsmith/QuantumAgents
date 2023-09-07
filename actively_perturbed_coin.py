from itertools import product
import random
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
from qiskit.quantum_info import Statevector, partial_trace
import matplotlib.pyplot as plt

from agent_encoding import QuantumAgentEncoder
from constants import *


def simulate_apc(its=10000, p=None, q=None, seed=None):
    '''Quantum agent based modelling simulation of the actively perturbed coin.  The agent represents 
    a single coin with states 0 and 1, receiving a binary input x ∈ {0, 1} at each time step. In 
    response, the agent flips the coin with probability p if x = 1 and with probability q if x = 0, 
    where 0 < p, q < 1. The agent then outputs the new state y ∈ {0, 1} of the coin.
    '''

    if seed is not None:
        random.seed(seed)
    p = p or random.random()
    q = q or random.random()

    print(f'p: {p}\nq: {q}')

    def transition_probabilities(y, x, s):
        '''Transition probabilities for the actively purturbed coin.
        Return the probability of outputting y given input x and causal state s.
        '''

        probs = {
            (0, 0, 0): 1 - q,
            (1, 0, 0): q,
            (0, 1, 0): 1 - p,
            (1, 1, 0): p,
            (0, 0, 1): q,
            (1, 0, 1): 1 - q,
            (0, 1, 1): p,
            (1, 1, 1): 1 - p,
        }
        return probs[(y, x, s)]

    def update_rule(x, y, s):
        '''Update rule for the actively purturbed coin'''
        return y

    causal_states = [0, 1]
    inputs = [0, 1]
    outputs = [0, 1]
    input_encodings = QuantumAgentEncoder.encode_vals(inputs)
    output_encodings = QuantumAgentEncoder.encode_vals(outputs)

    input_state_map = {input_val: input_encodings[i] for i, input_val in enumerate(inputs)}
    # output_state_map = {output_val: output_encodings[i] for i, output_val in enumerate(outputs)}

    encoder = QuantumAgentEncoder(
        causal_states=causal_states,
        inputs=inputs,
        outputs=outputs,
        input_encodings=input_encodings,
        output_encodings=output_encodings,
        transition_probs=transition_probabilities,
        update_rule=update_rule,
        numerical=False,
    )
    encoder.encode()

    input_state = 0
    memory_state = 0
    results = []
    for _ in tqdm(range(its), desc=f'Simulating agent for {its} iterations'):
        qc = encoder.create_quantum_circuit(encoder.memory_state_map[memory_state], input_state_map[input_state])

        # Obtain the statevector
        sv = Statevector(qc)
        # sv.seed(random.randint(0, 1e6))

        # Measure the output qubit and obtain the post-measurement state
        outcome, post_measurement_state = sv.measure([1])
        output_state = int(outcome)

        # Save the results
        results.append([input_state, output_state, memory_state])

        # Trace out the input, output, and junk states to obtain the updated memory state
        quantum_memory_state = np.asarray(partial_trace(post_measurement_state, [0, 1, 2]).to_statevector()).reshape(
            2, 1
        )

        # Make sure the new memory state agrees with the update rule
        assert np.allclose(
            quantum_memory_state, encoder.memory_state_map[update_rule(input_state, output_state, memory_state)]
        )

        # Get the value corresponding to the new memory state
        memory_state = min(
            encoder.memory_state_map.keys(), key=lambda k: norm(quantum_memory_state - encoder.memory_state_map[k])
        )

        # Get a new random input
        input_state = random.randint(0, 1)

    # Check to make sure we obtain the correct transition probabilities
    transition_probs = {
        (0, 0, 0): 0,
        (1, 0, 0): 0,
        (0, 1, 0): 0,
        (1, 1, 0): 0,
        (0, 0, 1): 0,
        (1, 0, 1): 0,
        (0, 1, 1): 0,
        (1, 1, 1): 0,
    }

    for input_state, output_state, memory_state in results:
        transition_probs[(output_state, input_state, memory_state)] += 1

    # Counts follow a Poisson distribution, of which the standard deviation is the square root of the mean
    tp_errors = transition_probs.copy()
    for k, n in tp_errors.items():
        tp_errors[k] = np.sqrt(n)

    states = []
    probs = []
    prob_errors = []
    probs_theoretical = []
    for x, s in product(range(2), range(2)):
        for y in range(2):
            states.append((y, x, s))

            # Compute the transition probability
            denom = transition_probs[(y, x, s)] + transition_probs[(1 - y, x, s)]
            probs.append(transition_probs[(y, x, s)] / denom)

            # Compute the error in the transition probability
            denom_error = np.sqrt(tp_errors[(y, x, s)] ** 2 + tp_errors[(1 - y, x, s)] ** 2)
            error = probs[-1] * np.sqrt(
                (tp_errors[(y, x, s)] / transition_probs[(y, x, s)]) ** 2 + (denom_error / denom) ** 2
            )
            prob_errors.append(error)

            # Get the theoretical transition probability
            probs_theoretical.append(transition_probabilities(y, x, s))

    states = [f'$\mathbb P(y={s[0]}|x={s[1]},s=s_{s[2]})$' for s in states]

    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 12, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})
    plt.errorbar(states, probs, yerr=prob_errors, label='Computed', c='r', alpha=0.5, fmt='.', elinewidth=1, zorder=2)
    plt.scatter(states, probs_theoretical, label='Theoretical', c='b', alpha=0.5, marker='.', zorder=1)
    plt.legend(loc='upper left')
    plt.ylabel('Transition probability')
    plt.title('Transition Probabilities of the Actively Perturbed Coin')
    plt.axhline(y=p, color='k', linestyle=':', linewidth=0.9, zorder=0)
    plt.axhline(y=1 - p, color='k', linestyle=':', linewidth=0.9, zorder=0)
    plt.axhline(y=q, color='k', linestyle=':', linewidth=0.9, zorder=0)
    plt.axhline(y=1 - q, color='k', linestyle=':', linewidth=0.9, zorder=0)
    plt.text(x=plt.xlim()[1] + 0.03, y=p - 0.008, s='$p$')
    plt.text(x=plt.xlim()[1] + 0.03, y=1 - p - 0.008, s='$1-p$')
    plt.text(x=plt.xlim()[1] + 0.03, y=q - 0.008, s='$q$')
    plt.text(x=plt.xlim()[1] + 0.03, y=1 - q - 0.008, s='$1-q$')
    plt.xticks(fontsize=8, rotation=20)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()
    plt.show()
    # plt.savefig('transition_probabilities.png', dpi=300)


if __name__ == '__main__':
    simulate_apc(its=100000, p=0.8, q=0.35, seed=SEED)
