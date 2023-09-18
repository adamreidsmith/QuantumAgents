from functools import reduce
from itertools import product
from typing import Callable, Any
from sympy import Symbol, solve, nsolve, sqrt as sympy_sqrt
import numpy as np
from numpy.linalg import cholesky, eigvals, norm, pinv, matrix_rank
from scipy.linalg import ishermitian
from qiskit import QuantumCircuit
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Statevector

from constants import *


def multi_kron(*v):
    # Compute the Kronecker product of several vectors

    return reduce(np.kron, v)


def kron_power(v, n):
    # Take the kronecher product of n copies of v

    return reduce(np.kron, [v] * n)


class QuantumAgentEncoder:
    def __init__(
        self,
        causal_states: list[Any],
        inputs: list[Any],
        outputs: list[Any],
        transition_probs: Callable[[Any, Any, Any], float],
        update_rule: Callable[[Any, Any, Any], Any],
        input_encodings: list[np.ndarray] | None = None,
        output_encodings: list[np.ndarray] | None = None,
        numerical: bool = True,
    ):
        '''
        Parameters
        ----------
        causal_states : list
            List of causal states of the adaptive agent.
        inputs : list
            List of inputs to the input-output process. If input_encodings is not specified,
            these should be non-negative integers.
        outputs : list
            List of outputs to the input-output process. If output_encodings is not specified,
            these should be non-negative integers.
        transition_probs : Callable[[Any, Any, Any], float]
            A function that takes an output value, an input value, and a causal state in that
            order as parameters and returns the transition probability, i.e.
            transition_probs(y, x, s) = P(y|x, s).
        update_rule : Callable[[Any, Any, Any], Any]
            A function that takes an input value, and output value, and a causal state in that
            order as parameters and returns a new causal state, i.e. update_rule(x, y, s) = s'.
        input_encodings : list[np.ndarray] | None
            A list of quantum encodings of the inputs as ndarrays. If None, encodings are
            deduced from the integer input values. Default: None.
        output_encodings : list[np.ndarray] | None
            A list of quantum encodings of the outputs as ndarrays. If None, encodings are
            deduced from the integer output values. Default: None.
        numerical : bool
            Whether to solve the multivariate polynomials numerically or symbolically.
            Default: True
        '''

        self.causal_states = causal_states
        self.causal_state_pairs = list(product(self.causal_states, self.causal_states))
        self.inputs = inputs
        self.outputs = outputs
        self.transition_probs = transition_probs
        self.update_rule = update_rule
        self.numerical = numerical

        # Encode the inputs, if necessary
        if input_encodings is None:
            input_encodings = self.encode_vals(self.inputs)
        self.input_encodings = input_encodings
        if output_encodings is None:
            output_encodings = self.encode_vals(self.inputs)
        self.output_encodings = output_encodings

        self.input_state_map = {x: x_encoded for x, x_encoded in zip(self.inputs, self.input_encodings)}

        # Number of qubits necessary to encode the input and output states
        self.n_qubits_input_tape = int(np.log2(len(self.input_encodings[0])))
        self.n_qubits_output_tape = int(np.log2(len(self.output_encodings[0])))

        # Initialize variables to store values to be computed when encode() is called
        self.unitary = None
        self.memory_state_map = None
        self.junk_state_map = None
        self.n_qubits_memory_tape = None
        self.n_qubits_junk_tape = None

    @staticmethod
    def find_valid_solution(solutions):
        '''Determine which solution of the multivariate polynomial we are looking for'''

        for sol in solutions:
            sol_vals = [complex(val) for val in sol.values()]
            if all(v.imag == 0.0 and v.real >= 0 and v.real <= 1 + 1e-8 for v in sol_vals) and not all(
                v.real < 1e-8 for v in sol_vals
            ):
                return sol
        raise ValueError('No valid solution found')

    @staticmethod
    def expand_with_identity(m, d):
        '''Add identity columns and rows to a square matrix m to expand it to shape (d, d)'''

        mshape = m.shape[0]
        if mshape == d:
            return m

        m = np.pad(m, ((0, d - mshape), (0, d - mshape)))
        for i in range(mshape, d):
            m[i, i] = 1.0
        return m

    @staticmethod
    def householder_vector(x):
        '''Generate the Householder vector'''

        e1 = np.eye(len(x))[0]
        v = x + np.sign(x[0]) * norm(x) * e1
        return v / norm(v)

    @staticmethod
    def householder_reflection(v):
        '''Construct the Householder reflection matrix'''

        I = np.eye(len(v))
        vvH = np.outer(v, v.conj())
        return I - 2 * vvH

    @staticmethod
    def reconstruct_matrix(x, b):
        '''Reconstruct columns of a matrix A from x and b satisfying Ax=b.
        x and b should be matrices with the vectors as columns.
        '''

        A = np.zeros((x.shape[0], x.shape[0]))
        x_pinv = pinv(x)
        b_x_pinv = b @ x_pinv
        x_x_pinv = x @ x_pinv

        for i in range(A.shape[0]):
            if np.allclose(x_x_pinv[:, i], np.eye(A.shape[0])[:, i]):
                A[:, i] = b_x_pinv[:, i]
            elif np.sum(np.abs(x_x_pinv[:, i])) > 1e-8:
                raise RuntimeError(
                    'Cannot reconstruct matrix, likely because the columns of matrix x are linearly dependent'
                )
        return A

    @staticmethod
    def gram_schmidt(vectors, start_index=0):
        '''Apply the Gram-Schmidt process to orthonormalize the set of vectors'''

        proj = lambda v, u: (np.dot(v, u.conj()) / np.dot(u, u.conj())) * u

        for i in range(start_index, len(vectors)):
            v = vectors[i]

            # Remove from v the projection of v onto each previous vector in the set
            for u in vectors[:i]:
                v -= proj(v, u)

            # Normalize v
            v /= norm(v)
            vectors[i] = v

        # Make sure the vectors are orthonormal
        for v1, v2 in product(vectors, vectors):
            if np.all(v1 == v2):
                assert np.abs(v1 @ v2.T.conj() - 1) < 1e-12, 'Vectors are not normalized'
            else:
                assert np.abs(v1 @ v2.T.conj()) < 1e-12, 'Vectors are not orthogonal'

        return vectors

    @classmethod
    def extend_columns_to_orthonormal_basis(cls, U):
        '''Replace the zero columns of U with non-zero unit vectors such that the columns form an orthonormal basis,
        making U a unitary matrix. The supplied non-zero columns of U must already be orthonormal.
        '''

        nonzero_col_indices = [i for i in range(U.shape[1]) if np.sum(np.abs(U[:, i])) > 1e-12]
        nonzero_cols = [U[:, i] for i in nonzero_col_indices]

        # Check for orthonormality of the supplied columns of U
        for i, j in product(nonzero_col_indices, nonzero_col_indices):
            overlap = U[:, i] @ U[:, j].T.conj()
            if i == j:
                assert np.abs(overlap - 1) < 1e-12, f'Column {i} is not normalized'
            else:
                assert np.abs(overlap) < 1e-12, f'Columns {i} and {j} are not orthogonal'

        zero_vector = np.zeros_like(nonzero_cols[0])
        dim_index = 0

        # Extend the set of columns to a basis
        while len(nonzero_cols) < U.shape[1]:
            basis_vector = zero_vector.copy()
            basis_vector[dim_index] = 1.0

            nonzero_cols.append(basis_vector)

            if matrix_rank(np.column_stack(nonzero_cols)) != len(nonzero_cols):
                nonzero_cols.pop()

            if dim_index >= U.shape[0]:
                raise RuntimeError('Failed to orthonormalize columns')

            dim_index += 1

        # Apply Gram-Schmidt to orthonormalize the columns
        nonzero_cols = cls.gram_schmidt(nonzero_cols, start_index=len(nonzero_col_indices))
        nonzero_cols = nonzero_cols[::-1]  # Reverse the list

        # Place the original non-zero columns of U into their original positions
        nonzero_cols_ordered = [None] * len(nonzero_cols)
        for i in nonzero_col_indices:
            nonzero_cols_ordered[i] = nonzero_cols.pop()

        # Place the remaining columns in the correct positions
        for i, vec in enumerate(nonzero_cols_ordered):
            if vec is None:
                nonzero_cols_ordered[i] = nonzero_cols.pop()

        return np.column_stack(nonzero_cols_ordered)

    @staticmethod
    def encode_vals(vals: list[int]):
        '''Encode values as quantum states'''

        encodings = []
        max_input = max(vals)
        n_qubits = int(np.ceil(np.log2(max_input + 1)))
        for x in vals:
            x_binary = bin(x)[2:].rjust(n_qubits, '0')
            encodings.append(multi_kron(*(KET_ZERO if d == '0' else KET_ONE for d in x_binary)))
        return encodings

    @classmethod
    def align_first_row(cls, L):
        '''Align the first row of matrix L with the canonical basis unit vector e1.
        This is done by reflecting the first row of L onto the canonical basis unit
        vector e1 using a Householder reflection.
        '''

        # Extract the first row
        l1 = L[0, :].copy()

        # Return if no transformation is necessary
        if abs(l1[0]) > 1e-8 and np.all(np.abs(l1[1:]) < 1e-8):
            return L

        # Compute the Householder vector for the first row and generate the Householder reflection matrix
        v = cls.householder_vector(l1)
        H = cls.householder_reflection(v)

        # Ensure the transformation is unitary
        assert np.allclose(np.eye(L.shape[0]), H.T.conj() @ H), 'Householder transformation not unitary'

        # Reflect the matrix
        L @= H

        # Ensure the first component is positive
        if L[0, 0] < 0:
            return -L
        return L

    def encode(self):
        '''Encode the agent in the quantum realm'''

        input_specialized_overlaps = self.compute_input_specialized_overlaps()
        memory_states, junk_states = self.construct_quantum_memory_junk_states(input_specialized_overlaps)
        self.unitary = self.construct_unitary(memory_states, junk_states)

    def compute_input_specialized_overlaps(self):
        '''Step 1'''

        # Define the variables of the multivariate polynomial equations
        # These are the overlaps of the input specialized memory states
        variables, variable_names = {}, {}
        for x in self.inputs:
            for s0, s1 in self.causal_state_pairs:
                var_name = f'c{x}{s0}{s1}'
                var = Symbol(var_name)
                variables[var_name] = var
                variable_names[var] = var_name

        # Formulate the multivariate polynomial equations relating input specialized overlaps
        # 0 = \sum_y\sqrt{P(y|x,s)P(y|x,s')}\prod_{x'}c^{x'}_{\lambda(z,s)\lambda(z,s')} - c^x_{ss'}
        equations = []
        for x in self.inputs:
            for s0, s1 in self.causal_state_pairs:
                equation = 0
                for y in self.outputs:
                    s0_updated = self.update_rule(x, y, s0)
                    s1_updated = self.update_rule(x, y, s1)
                    prod_vars = [variables[f'c{xp}{s0_updated}{s1_updated}'] for xp in self.inputs]

                    equation += sympy_sqrt(
                        self.transition_probs(y, x, s0) * self.transition_probs(y, x, s1)
                    ) * np.prod(prod_vars)
                equation -= variables[f'c{x}{s0}{s1}']
                equations.append(equation)

        # Solve the equations using sympy.solve or sympy.nsolve
        if self.numerical:
            solutions = nsolve(equations, list(variables.values()), [0.8] * len(variables), dict=True)
        else:
            solutions = solve(equations, *variables.values(), dict=True)
        solution = self.find_valid_solution(solutions)
        solution = {variable_names[var]: float(val) for var, val in solution.items()}

        return solution

    def construct_quantum_memory_junk_states(self, input_specialized_overlaps: dict[str, float]):
        '''Step 2'''

        # Compute the memory state overlaps from the input specialized memory state overlaps
        memory_state_overlaps = {}
        for s0, s1 in self.causal_state_pairs:
            memory_state_overlaps[f'c{s0}{s1}'] = np.prod(
                [input_specialized_overlaps[f'c{x}{s0}{s1}'] for x in self.inputs]
            )

        # Compute the junk state overlaps from the input specialized memory state overlaps
        junk_state_overlaps = {}
        for x in self.inputs:
            for s0, s1 in self.causal_state_pairs:
                junk_state_overlaps[f'd{x}{s0}{s1}'] = np.prod(
                    [input_specialized_overlaps[f'c{xp}{s0}{s1}'] for xp in self.inputs if xp != x]
                )

        # Compute the gram matrix for the memory states
        memory_gram_matrix = np.empty((len(self.causal_states), len(self.causal_states)))
        for (i, s0), (j, s1) in product(enumerate(self.causal_states), enumerate(self.causal_states)):
            memory_gram_matrix[i, j] = memory_state_overlaps[f'c{s0}{s1}']

        # Compute the gram matrix for the junk states
        junk_gram_matrices = [np.empty_like(memory_gram_matrix) for _ in range(len(self.inputs))]
        for k, x in enumerate(self.inputs):
            for (i, s0), (j, s1) in product(enumerate(self.causal_states), enumerate(self.causal_states)):
                junk_gram_matrices[k][i, j] = junk_state_overlaps[f'd{x}{s0}{s1}']

        # Expand the gram matrices so that their dimension is a power of 2
        # This ensures memory and junk states are vectors with a power of 2 length
        new_dim = 2 ** int(np.ceil(np.log2(memory_gram_matrix.shape[0])))
        memory_gram_matrix = self.expand_with_identity(memory_gram_matrix, new_dim)
        junk_gram_matrices = [self.expand_with_identity(m, new_dim) for m in junk_gram_matrices]

        # Some sanity checks on the Gram matrices
        assert ishermitian(memory_gram_matrix), 'Memory state Gram matrix not Hermitian'
        assert all(ishermitian(m) for m in junk_gram_matrices), 'Junk state gram matrix not Hermitian'
        assert np.all(eigvals(memory_gram_matrix) > 0), 'Memory state Gram matrix not positive definite'
        assert all(np.all(eigvals(m) > 0) for m in junk_gram_matrices), 'Junk state Gram matrix not positive definite'

        # The Cholesky decomposition gives the vectors from the set of inner products
        memory_cholesky_decomp = cholesky(memory_gram_matrix)
        junk_cholesky_decomps = [cholesky(m) for m in junk_gram_matrices]

        # The inner product is invariant under unitary transformations, so we apply a Householder reflection to
        # align the first vector with the first canonical basis unit vector
        memory_states = [v.reshape(-1, 1) for v in self.align_first_row(memory_cholesky_decomp)]
        memory_states = {f'm{s}': memory_states[i] for i, s in enumerate(self.causal_states)}
        junk_states = [[v.reshape(-1, 1) for v in self.align_first_row(m)] for m in junk_cholesky_decomps]
        junk_states = {
            f'j{x}{s}': junk_states[i][j] for i, x in enumerate(self.inputs) for j, s in enumerate(self.causal_states)
        }

        # Make sure overlaps are correct
        for s0, s1 in self.causal_state_pairs:
            overlap = memory_states[f'm{s0}'].T @ memory_states[f'm{s1}']
            assert (
                abs(memory_state_overlaps[f'c{s0}{s1}'] - overlap) < 1e-8
            ), f'Memory states from causal states ({s0}, {s1}) do not provide the correct overlap'
        for x in self.inputs:
            for s0, s1 in self.causal_state_pairs:
                overlap = junk_states[f'j{x}{s0}'].T @ junk_states[f'j{x}{s1}']
                assert (
                    abs(junk_state_overlaps[f'd{x}{s0}{s1}'] - overlap) < 1e-8
                ), f'Junk states from input ({x}) and causal states ({s0}, {s1}) do not provide the correct overlap'

        self.memory_state_map = {s: memory_states[f'm{s}'] for s in self.causal_states}
        self.junk_state_map = {(x, s): junk_states[f'j{x}{s}'] for x in self.inputs for s in self.causal_states}

        self.n_qubits_memory_tape = int(np.log2(len(list(memory_states.values())[0])))
        self.n_qubits_junk_tape = int(np.log2(len(list(junk_states.values())[0])))

        return memory_states, junk_states

    def construct_unitary(self, memory_states: dict[str, np.ndarray], junk_states: dict[str, np.ndarray]):
        '''Steps 3 and 4'''

        # Compute the initial and final states of the agents unitary transformation
        LHS_vectors, RHS_vectors = [], []
        for x_ind, x in enumerate(self.inputs):
            for s in self.causal_states:
                output_qubits = kron_power(KET_ZERO, int(np.log2(len(self.output_encodings[0]))))
                junk_qubits = kron_power(KET_ZERO, int(np.log2(len(list(junk_states.values())[0]))))
                LHS = multi_kron(memory_states[f'm{s}'], self.input_encodings[x_ind], output_qubits, junk_qubits)
                LHS_vectors.append(LHS)

                RHS = 0
                for y_ind, y in enumerate(self.outputs):
                    coeff = np.sqrt(self.transition_probs(y, x, s))
                    state = multi_kron(
                        memory_states[f'm{self.update_rule(x, y, s)}'],
                        self.input_encodings[x_ind],
                        self.output_encodings[y_ind],
                        junk_states[f'j{x}{s}'],
                    )
                    RHS += coeff * state
                RHS_vectors.append(RHS)

        LHS = np.hstack(LHS_vectors)
        RHS = np.hstack(RHS_vectors)

        # Reconstruct the unitary matrix U given by U @ LHS = RHS.
        U = self.reconstruct_matrix(LHS, RHS)

        # Ensure the reconstruction was successful
        assert np.allclose(U @ LHS - RHS, 0), 'Unitary reconstruction failed'

        full_unitary = self.extend_columns_to_orthonormal_basis(U)

        assert np.allclose(
            full_unitary.T.conj() @ full_unitary, np.eye(full_unitary.shape[0])
        ), 'Construction failed. Matrix not unitary'

        return full_unitary

    def create_quantum_circuit(self, causal_state, input_val):
        '''Create a quantum circuit representing the evolution of the quantum agent
        for particular memory and input states
        '''

        input_state = self.input_state_map[input_val]
        memory_state = self.memory_state_map[causal_state]

        if self.unitary is None:
            raise RuntimeError('You must encode the agent before a quantum circuit can be created')

        n_qubits = sum(
            [
                self.n_qubits_memory_tape,
                self.n_qubits_input_tape,
                self.n_qubits_output_tape,
                self.n_qubits_junk_tape,
            ]
        )

        qc = QuantumCircuit(n_qubits)  # , self.n_qubits_output_tape)

        # Initialize the circuit in the initial state and apply the unitary operator
        initial_state = multi_kron(
            memory_state,
            input_state,
            kron_power(KET_ZERO, self.n_qubits_output_tape),
            kron_power(KET_ZERO, self.n_qubits_junk_tape),
        )
        qc.initialize(Statevector(initial_state))
        qc.append(UnitaryGate(self.unitary), range(n_qubits))

        # # Measure the output tape
        # # Qiskit's backwards qubit ordering means we have to count backwards to get the right index
        # output_qubit_index = self.n_qubits_junk_tape
        # qc.measure(
        #     range(output_qubit_index, output_qubit_index + self.n_qubits_output_tape),
        #     range(self.n_qubits_output_tape),
        # )

        return qc
