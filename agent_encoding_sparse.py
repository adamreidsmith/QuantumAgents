import math
import warnings
from functools import reduce
from itertools import product, combinations_with_replacement
from typing import Any
from collections.abc import Hashable, Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse as sparse
import scipy.sparse.linalg as sparsela
from scipy.sparse import SparseEfficiencyWarning
from sympy import Symbol, solve, nsolve, sqrt as sympy_sqrt, I
from qiskit.quantum_info import Statevector, partial_trace

from constants import *


def multi_kron(*v: np.ndarray) -> np.ndarray:
    '''Compute the Kronecker product of several vectors.'''
    return reduce(np.kron, v)


def sparse_kron(*v: sparse.spmatrix) -> sparse.spmatrix:
    '''Compute the Kronecker product of several vectors.'''
    return reduce(sparse.kron, v).tocsc()


def kron_power(v: np.ndarray, n: int) -> np.ndarray:
    '''Take the kronecher product of n copies of v.'''
    return reduce(np.kron, [v] * n)


def prod(*vals: Any) -> Any:
    '''Computes the product of vals. If `vals` is a single non-iterable object, it is
    returned as is.
    '''
    if len(vals) == 1:
        vals = vals[0]
    if not isinstance(vals, Iterable):
        return vals
    return reduce(lambda a, b: a * b, vals)


class QuantumAgentEncoder:
    '''
    Encodes input-output processes in the quantum realm.

    References
    ----------
    [1] T. J. Elliott, M. Gu, A. J. P. Garner, and J. Thompson, “Quantum adaptive agents
        with efficient long-term memories,” Phys. Rev. X, vol. 12, p. 011007, Jan 2022.
    '''

    def __init__(
        self,
        causal_states: list[Hashable],
        inputs: list[Hashable],
        outputs: list[Hashable],
        transition_probs: Callable[[Hashable, Hashable, Hashable], float],
        update_rule: Callable[[Hashable, Hashable, Hashable], Hashable],
        method: str = 'broyden',
        initialize_jacobian: bool = True,
        processes: int = 8,
        tol: float = 1e-12,
        compute_full_unitary: bool = True,
        clean: bool = True,
    ) -> None:
        '''
        Parameters
        ----------
        causal_states : list[Hashable]
            List of causal states of the adaptive agent.
        inputs : list[Hashable]
            List of inputs to the input-output process. If `input_encodings` is not specified,
            these should be non-negative integers.
        outputs : list[Hashable]
            List of outputs to the input-output process. If `output_encodings` is not specified,
            these should be non-negative integers.
        transition_probs : Callable[[Hashable, Hashable, Hashable], float]
            A function that takes an output value, an input value, and a causal state in that
            order as parameters and returns the transition probability, i.e.
            transition_probs(y, x, s) = P(y|x, s).
        update_rule : Callable[[Hashable, Hashable, Hashable], Hashable]
            A function that takes an input value, and output value, and a causal state in that
            order as parameters and returns a new causal state, i.e. update_rule(x, y, s) = s'.
        method : str
            The method used for solving non-linear systems of equations. Options are ['broyden',
            'sympy_numeric', 'symbolic']. Default: 'broyden'
        initialize_jacobian : bool
            Whether or not to initialize the Jacobian matrix for Broyden's method using finite
            differences. Doing so is more computationally intensive, but generally provides
            faster convergence. Only applies when method='broyden'. Default: True
        processes : int
            The number of processes to use when computing the Jacobian for Broyden's method.
            Only applies when method='broyden' and initialize_jaobian=True. Default: 8
        tol : float
            Tolerance for Broyden's method, Hermitian/unitary checks, etc. Default: 1e-12
        compute_full_unitary : bool
            If True, the full agent evolution unitary will be computed. Otherwise, only the columns
            necessary to evolve the agent classically will be computed, which significantly reduces
            copmutation times. Default: True
        clean : bool
            If True, remove auxillary variables not used after encoding. Default: True
        '''

        # The user-defined causal states (any Hashable)
        self.causal_states = causal_states
        self.causal_state_pairs = list(product(self.causal_states, self.causal_states))
        # The user-defined inputs (any Hashable)
        self.inputs = inputs
        # The user-defined outputs (any Hashable)
        self.outputs = outputs
        # A function mapping tuples (output_state, input_state, causal_state) to the transition probability
        # P(output_state | input_state, causal_state)
        self.transition_probs = transition_probs
        # A function mapping tuples (input_state, output_state, causal_state) of states at time t to the updated
        # causal state at time t + 1
        self.update_rule = update_rule
        # The method used for solving non-linear systems of equations
        if method not in ['broyden', 'sympy_numeric', 'symbolic']:
            raise ValueError(f'Invalid method: {method}')
        self.method = method
        self.initialize_jacobian = initialize_jacobian
        self.processes = processes
        self.tol = tol
        self.compute_full_unitary = compute_full_unitary
        self.clean = clean

        # Quantum state encodings of the input states
        self.input_encodings = self.encode_vals(self.inputs)
        # Quantum state encodings of the output states
        self.output_encodings = self.encode_vals(self.outputs)

        # A map from the input states to their quantum encodings
        self.input_state_map = {x: x_encoded for x, x_encoded in zip(self.inputs, self.input_encodings)}

        # The unitary that defines the agent evolution
        self.unitary: sparse.csr_matrix
        # A map from the agents memory/causal states to their quantum state representations
        self.memory_state_map: dict[Hashable, np.ndarray]
        # A map from tuples (input, causal state) to the quantum representations of the junk states
        self.junk_state_map: dict[tuple[Hashable, Hashable], np.ndarray]
        # The number of qubits in the memory/causal states/tape
        self.n_qubits_memory_tape: int
        # The number of qubits in the input states/tape
        self.n_qubits_input_tape = int(math.log2(len(self.input_encodings[0])))
        # the number of qubits in the output states/tape
        self.n_qubits_output_tape = int(math.log2(len(self.output_encodings[0])))
        # The number of qubits in the junk states/tape
        self.n_qubits_junk_tape: int

        # A map from tuples (causal state, input) to the quantum state representation (as a numpy array)
        # of the **LHS** of the evolution equation (equation (3) from [1])
        self.initial_state_map: dict[tuple[Hashable, Hashable], np.ndarray]
        # A map from tuples (causal state, input) to the quantum state representation (as a Statevector)
        # of the **RHS** of the evolution equation (equation (3) from [1])
        self.compiled_initial_state_map: dict[tuple[Hashable, Hashable], Statevector] | None = None

    @staticmethod
    def expand_with_identity(m: np.ndarray, d: int) -> np.ndarray:
        '''Add identity columns and rows to a square matrix m to expand it to shape (d, d)'''

        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(f'`m` must be a square matrix: found dimensions {m.shape}')

        mshape = m.shape[0]
        if mshape == d:
            return m

        m = np.pad(m, ((0, d - mshape), (0, d - mshape)))
        for i in range(mshape, d):
            m[i, i] = 1.0
        return m

    @staticmethod
    def encode_vals(vals: list[Hashable]) -> list[np.ndarray]:
        '''Encode values as quantum states. This maps a list of hashables [h_0, ..., h_{n-1}]
        to computational basis states [|0>, ..., |n-1>].
        '''

        encodings = []
        n_qubits = math.ceil(math.log2(len(vals)))
        print(f'Encoding vals with {n_qubits} qubits')
        for x in range(len(vals)):
            x_binary = f'{x:0{n_qubits}b}'
            encodings.append(multi_kron(*(KET_ZERO if d == '0' else KET_ONE for d in x_binary)))
        return encodings

    def encode(self) -> None:
        '''Encode the agent in the quantum realm.'''

        print('Checking update rule dependence on causal states')
        if self.check_update_rule_independence():
            input_specialized_overlaps = self.compute_independent_input_specialized_overlaps()
        else:
            print('Solving for input-specialized overlaps')
            if self.method == 'broyden':
                input_specialized_overlaps = self.compute_input_specialized_overlaps_broyden()
            else:
                input_specialized_overlaps = self.compute_input_specialized_overlaps()
        print('Constructing memory and junk states')
        self.construct_quantum_memory_junk_states(input_specialized_overlaps)
        print('Constructing unitary')
        self.unitary: sparse.csr_matrix = self.construct_unitary()
        if self.clean:
            self.causal_states = None
            self.causal_state_pairs = None
            self.inputs = None
            self.input_encodings = None
            self.output_encodings = None
            self.input_state_map = None
            self.junk_state_map = None
            self.initial_state_map = None
            self.unitary = None

    def check_update_rule_independence(self):
        '''
        Check whether or not the update rule is independent of causal state.  If it is,
        a significant simplification in the computation of memory state overlaps is possible:

        c_{lambda(z, s)lambda(z, s')} = 1  =>  c^x_{ss'} = sum_y(sqrt[P(y|x,s)P(y|x,s')])
        '''

        for x in self.inputs:
            for y in self.outputs:
                s_updated = self.update_rule(x, y, self.causal_states[0])
                for s in self.causal_states:
                    if self.update_rule(x, y, s) != s_updated:
                        return False
        return True

    def compute_independent_input_specialized_overlaps(self):
        '''
        Compute input specialized overlaps in the case where the update rule is independent
        of the memory state.
        '''

        input_specialized_overlaps = {}
        for x in tqdm(self.inputs, desc='Computing independent input specialized overlaps'):
            for s0, s1 in self.causal_state_pairs:
                overlap = sum(
                    math.sqrt(self.transition_probs(y, x, s0) * self.transition_probs(y, x, s1)) for y in self.outputs
                )
                input_specialized_overlaps[f'c|{x}|{s0}|{s1}'] = overlap
        return input_specialized_overlaps

    @staticmethod
    def approx_jacobian_sequential(
        x: np.ndarray, f: Callable[[np.ndarray], np.ndarray], show_progress: bool
    ) -> np.ndarray:
        '''Approximate a Jacobian using finite differences.'''
        n = len(x)
        J = np.zeros((n, n), dtype=complex)
        eps = np.sqrt(np.finfo(float).eps)
        for j in tqdm(range(n), desc='Initializing Jacobian', disable=not show_progress):
            # Create a small perturbation
            dx = np.zeros_like(x, dtype=complex)
            dx[j] = eps
            # Compute finite difference approximation
            J[:, j] = (f(x + dx) - f(x)) / eps
        return J

    @staticmethod
    def jac_parallel_helper(args: tuple[np.ndarray, Callable, list]) -> np.ndarray:
        '''Helper for approx_jacobian_parallel.'''
        x, f, idx = args
        f.compile_equations()
        J = np.zeros((len(x), len(idx)), dtype=complex)
        eps = np.sqrt(np.finfo(float).eps)
        print(f'Computing approximate Jacobian for indices {idx[0]} to {idx[-1]}')
        for j in tqdm(idx, desc='Process 0', disable=idx[0] != 0):
            # Create a small perturbation
            dx = np.zeros_like(x, dtype=complex)
            dx[j] = eps
            # Compute finite difference approximation
            J[:, j - idx[0]] = (f(x + dx) - f(x)) / eps
        return J

    def approx_jacobian_parallel(self, x: np.ndarray, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        '''Approximate a Jacobian using finite differences.'''
        indices = list(range(len(x)))
        indices_split = np.array_split(indices, self.processes)
        args = ((x, f, idx) for idx in indices_split)
        with ProcessPoolExecutor(max_workers=self.processes) as executor:
            Js = list(executor.map(self.jac_parallel_helper, args))
        return np.hstack(Js)

    def broyden(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        max_iter: int = 100,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, bool]:
        '''
        Applies Broyden's method for finding the root of a nonlinear function f: R^n -> R^n
        in n variables.
        '''

        # Initial Jacobian approximation
        if self.initialize_jacobian and self.processes > 1:
            B = self.approx_jacobian_parallel(x0, f)
            # f is compiled in each subprocess, but not in the main process, so we do it here.
            # f cannot be compiled before calling self.approx_jacobian_sequential since compiling
            # generates code objects, which are not picklable.
            f.compile_equations()
        else:
            f.compile_equations()
            B = (
                self.approx_jacobian_sequential(x0, f, show_progress)
                if self.initialize_jacobian
                else np.eye(len(x0), dtype=complex)
            )

        print('Running Broyden\'s method')
        # Initial function evaluation
        fx = f(x0)
        x = x0
        for it in range(max_iter):
            # Compute search direction
            dx = -npla.solve(B, fx)

            # Update solution
            x_new = x + dx
            fx_new = f(x_new)

            # Convergence check
            # Use absolute value of the norm for complex numbers
            error = np.abs(npla.norm(fx_new))
            if show_progress:
                print(f'Iteration: {it + 1}  |  Error: {error:.4g}')
            if error < self.tol:
                return x_new, True

            # Broyden's update for Jacobian approximation
            # B_{k+1} = B_k + (dy - B_k * dx) / (dx^T * dx) * dx^T
            dy = fx_new - fx
            numerator = dy - B @ dx
            denominator = np.dot(dx.conj(), dx)
            B += np.outer(numerator, dx.conj()) / denominator

            # Update for next iteration
            x = x_new
            fx = fx_new
        return x, False

    def compute_input_specialized_overlaps_broyden(self) -> dict[str, float]:
        '''Step 1 from [1].'''

        vars_str = [f'c|{x}|{s0}|{s1}' for x in self.inputs for s0, s1 in self.causal_state_pairs]
        letter_iter = LetterIterator()
        vars_to_simple_name = {v: next(letter_iter) for v in vars_str}

        equations = []
        for x in tqdm(self.inputs, desc='Formulating equations'):
            for s0, s1 in self.causal_state_pairs:
                if s0 == s1:
                    equations.append(f'1-' + vars_to_simple_name[f'c|{x}|{s0}|{s1}'])
                    continue
                eqn = []
                for y in self.outputs:
                    s0_updated = self.update_rule(x, y, s0)
                    s1_updated = self.update_rule(x, y, s1)
                    coef = math.sqrt(self.transition_probs(y, x, s0) * self.transition_probs(y, x, s1))
                    if coef == 0:
                        continue
                    prod_vars = (vars_to_simple_name[f'c|{xp}|{s0_updated}|{s1_updated}'] for xp in self.inputs)
                    eqn.append(f'{coef}*{"*".join(prod_vars)}')
                equations.append('+'.join(eqn) + '-' + vars_to_simple_name[f'c|{x}|{s0}|{s1}'])

        evaluation_function = Funcify(equations, [vars_to_simple_name[v] for v in vars_str])

        x0 = np.empty(len(vars_str), dtype=complex)
        for i, var in enumerate(vars_str):
            s0, s1 = var.split('|')[-2:]
            x0[i] = (1.0 + 0.0j) if s0 == s1 else 0.0j

        solution, converged = self.broyden(evaluation_function, x0)
        if not converged:
            raise RuntimeError('Failed to solve for input-specialized overlaps')
        solution = dict(zip(vars_str, solution))

        return solution

    def find_valid_solution(self, solutions: list[dict[Symbol, float | complex]]) -> dict[Symbol, float | complex]:
        '''Determine which solution of the multivariate polynomial we are looking for.'''

        sol_data = []
        for sol in solutions:
            sol_variables, sol_vals = zip(*sol.items())
            sol_vals = list(map(complex, sol_vals))

            # Reject zero solutions
            if all(abs(v) < self.tol for v in sol_vals):
                continue

            # Reject solutions which cannot result from quantum state overlaps due to asymmetry
            sol_var_as_str = {str(k): v for k, v in sol.items()}
            for var, val in zip(sol_variables, sol_vals):
                x, s0, s1 = str(var).split('|')[-3:]
                if abs(val - sol_var_as_str[f'c|{x}|{s1}|{s0}'].conjugate()) > self.tol:
                    continue

            unit_overlaps = True
            for var, val in zip(sol_variables, sol_vals):
                s0, s1 = str(var).split('|')[-2:]
                if s0 == s1 and abs(val - 1) > self.tol:
                    unit_overlaps = False
                    break

            n_real = 0
            for val in sol_vals:
                if abs(val.imag) < self.tol:
                    n_real += 1

            sol_data.append((unit_overlaps, n_real, sol))

        if len(sol_data) == 0:
            raise ValueError('No valid solution found')

        # Prefer solutions with unit overlaps, then solutions with more real values
        sol_data.sort(key=lambda sol: 1_000_000 * sol[0] + sol[1])
        return sol_data[-1][2]

    def compute_input_specialized_overlaps(self) -> dict[str, float]:
        '''Step 1 from [1].'''

        # Define the variables of the multivariate polynomial equations
        # These are the overlaps of the input specialized memory states
        var_list = []
        variables: dict[str, Symbol] = {}
        variable_names: dict[Symbol, str] = {}
        for x in self.inputs:
            for s0, s1 in self.causal_state_pairs:
                var_name = f'c|{x}|{s0}|{s1}'
                var = Symbol(var_name, complex=True)
                var_list.append(var)
                variables[var_name] = var  # Maps a variable name (str) to a Sympy variable
                variable_names[var] = var_name  # Maps a Sympy variable to a variable name (str)

        # Formulate the multivariate polynomial equations relating input specialized overlaps
        # 0 = \sum_y \sqrt{P(y|x,s)P(y|x,s')} \prod_{x'} c^{x'}_{\lambda(z,s)\lambda(z,s')} - c^x_{ss'}
        equations = []
        for x in tqdm(self.inputs, desc='Compiling Sympy equations'):
            for s0, s1 in self.causal_state_pairs:
                if s0 == s1:
                    equations.append(1 - variables[f'c|{x}|{s0}|{s1}'])
                    continue
                equation = 0
                for y in self.outputs:
                    s0_updated = self.update_rule(x, y, s0)
                    s1_updated = self.update_rule(x, y, s1)
                    prod_vars = (variables[f'c|{xp}|{s0_updated}|{s1_updated}'] for xp in self.inputs)
                    equation += sympy_sqrt(self.transition_probs(y, x, s0) * self.transition_probs(y, x, s1)) * prod(
                        prod_vars
                    )
                equation -= variables[f'c|{x}|{s0}|{s1}']
                equations.append(equation)

        # Solve the equations using sympy.solve or sympy.nsolve
        if self.method == 'sympy_numeric':
            var_names, _vars = zip(*variables.items())
            x0 = []
            for var_name in var_names:
                s0, s1 = var_name.split('|')[-2:]
                if s0 == s1:
                    # States should be pure, so overlaps of the same state should be 1
                    x0.append(1.0 + 0.0j)
                else:
                    x0.append(0.1 + 0.1 * I)
            solutions = nsolve(equations, _vars, x0, dict=True, solver=6)
        else:
            solutions = solve(equations, *variables.values(), dict=True)
        solution = self.find_valid_solution(solutions)
        solution = {variable_names[var]: float(val) for var, val in solution.items()}

        return solution

    def construct_quantum_memory_junk_states(self, input_specialized_overlaps: dict[str, float]):
        '''Step 2 from [1]'''

        print('Computing Gram matrices')
        # Compute the memory state overlaps from the input-specialized memory state overlaps
        memory_state_overlaps = {}
        for s0, s1 in self.causal_state_pairs:
            memory_state_overlaps[(s0, s1)] = prod(
                (input_specialized_overlaps[f'c|{x}|{s0}|{s1}'] for x in self.inputs)
            )

        # Compute the junk state overlaps from the input-specialized memory state overlaps
        junk_state_overlaps = {}
        for x in self.inputs:
            for s0, s1 in self.causal_state_pairs:
                junk_state_overlaps[(x, s0, s1)] = prod(
                    (input_specialized_overlaps[f'c|{xp}|{s0}|{s1}'] for xp in self.inputs if xp != x)
                )

        # Compute the Gram matrix for the memory states
        memory_gram_matrix = np.empty((len(self.causal_states), len(self.causal_states)), dtype=complex)
        for (i, s0), (j, s1) in product(enumerate(self.causal_states), enumerate(self.causal_states)):
            memory_gram_matrix[i, j] = memory_state_overlaps[(s0, s1)]

        # Compute the Gram matrices for the junk states
        junk_gram_matrices = [np.empty_like(memory_gram_matrix) for _ in range(len(self.inputs))]
        for k, x in enumerate(self.inputs):
            for (i, s0), (j, s1) in product(enumerate(self.causal_states), enumerate(self.causal_states)):
                junk_gram_matrices[k][i, j] = junk_state_overlaps[(x, s0, s1)]

        # Expand the Gram matrices so that their dimension is a power of 2
        # This ensures memory and junk states are vectors with a power of 2 length
        new_dim = 2 ** math.ceil(math.log2(memory_gram_matrix.shape[0]))
        memory_gram_matrix = self.expand_with_identity(memory_gram_matrix, new_dim)
        junk_gram_matrices = [self.expand_with_identity(m, new_dim) for m in junk_gram_matrices]

        # Some sanity checks on the Gram matrices
        assert spla.ishermitian(memory_gram_matrix, atol=self.tol), 'Memory state Gram matrix not Hermitian'
        assert all(
            spla.ishermitian(m, atol=self.tol) for m in junk_gram_matrices
        ), 'Junk state Gram matrix not Hermitian'
        assert np.all(npla.eigvals(memory_gram_matrix) > 0), 'Memory state Gram matrix not positive definite'
        assert all(
            np.all(npla.eigvals(m) > 0) for m in junk_gram_matrices
        ), 'Junk state Gram matrix not positive definite'
        print('Gram matrix checks passed')

        # The Cholesky decomposition gives the vectors from the set of inner products
        print('Computing Cholesky decompositions')
        memory_cholesky_decomp = npla.cholesky(memory_gram_matrix)
        junk_cholesky_decomps = [npla.cholesky(m) for m in junk_gram_matrices]

        # Check that the row vectors reproduce the Gram matrix
        assert (
            npla.norm(memory_cholesky_decomp @ memory_cholesky_decomp.T.conj() - memory_gram_matrix, ord='fro')
            < self.tol
        ), 'Gram matrix decomposition failed'
        assert all(
            npla.norm(jm @ jm.T.conj() - junk_gram_matrices[i], ord='fro') < self.tol
            for i, jm in enumerate(junk_cholesky_decomps)
        ), 'Gram matrix decomposition failed'

        print('Formulating memory and junk states')
        memory_states = [v.reshape(-1, 1) for v in memory_cholesky_decomp]
        self.memory_state_map = {s: memory_states[i] for i, s in enumerate(self.causal_states)}
        junk_states = [[v.reshape(-1, 1) for v in m] for m in junk_cholesky_decomps]
        self.junk_state_map = {
            (x, s): junk_states[i][j] for i, x in enumerate(self.inputs) for j, s in enumerate(self.causal_states)
        }

        # Make sure overlaps are correct
        for s0, s1 in self.causal_state_pairs:
            overlap = self.memory_state_map[s0].T @ self.memory_state_map[s1]
            assert (
                abs(memory_state_overlaps[(s0, s1)] - overlap) < self.tol
            ), f'Memory states from causal states ({s0}, {s1}) do not provide the correct overlap'
        for x in self.inputs:
            for s0, s1 in self.causal_state_pairs:
                overlap = self.junk_state_map[(x, s0)].T @ self.junk_state_map[(x, s1)]
                assert (
                    abs(junk_state_overlaps[(x, s0, s1)] - overlap) < self.tol
                ), f'Junk states from input ({x}) and causal states ({s0}, {s1}) do not provide the correct overlap'
        print('Memory and junk state checks passed')

        self.n_qubits_memory_tape = int(math.log2(len(list(self.memory_state_map.values())[0])))
        self.n_qubits_junk_tape = int(math.log2(len(list(self.junk_state_map.values())[0])))

        print('Computing initial states')
        self.initial_state_map = {}
        # Compute all possible input_states
        for s, memory_state in self.memory_state_map.items():
            for x, input_state in self.input_state_map.items():
                self.initial_state_map[(x, s)] = multi_kron(
                    memory_state,
                    input_state,
                    kron_power(KET_ZERO, self.n_qubits_output_tape),
                    kron_power(KET_ZERO, self.n_qubits_junk_tape),
                )

    def reconstruct_matrix(self, x: sparse.spmatrix, b: sparse.spmatrix) -> sparse.spmatrix:
        '''
        Reconstruct columns of a matrix A from x and b satisfying Ax=b.
        x and b should be matrices with the vectors as columns.
        '''

        # When ⁠A has linearly independent columns (as is the case here), pinv(A)⁠
        # can be computed as pinv(A) = (A^* @ A)^-1 @ A^*
        print(f'Computing pseudoinverse of {x.shape} matrix')
        with warnings.catch_warnings(action='ignore', category=SparseEfficiencyWarning):
            x_pinv = sparsela.inv(x.T.conj() @ x) @ x.T.conj()

        # Compute b @ x_pinv and x @ x_pinv
        b_x_pinv = (b @ x_pinv).tocsc()
        x_x_pinv = (x @ x_pinv).tocsc()

        # Prepare the result matrix as a sparse matrix
        A = sparse.csc_matrix((x.shape[0], x.shape[0]), dtype=complex)

        # Pre-subtract the identity and compute the norms to make the loop faster
        identity = sparse.dok_matrix(x_x_pinv.shape, dtype=complex)
        for i in range(identity.shape[0]):
            identity[i, i] = 1
        x_x_pinv_mi = (x_x_pinv - identity).tocsr()  # norm converts to csr anyway
        x_x_pinv_mi_norm: np.ndarray = sparsela.norm(x_x_pinv_mi, ord=2, axis=0)
        del x_x_pinv_mi
        x_x_pinv_norm: np.ndarray = sparsela.norm(x_x_pinv, ord=2, axis=0)

        col_mask = x_x_pinv_mi_norm < self.tol
        del x_x_pinv_mi_norm
        if np.any(x_x_pinv_norm[~col_mask] > self.tol):
            raise RuntimeError(
                'Cannot reconstruct matrix, likely because the columns of matrix x are linearly dependent'
            )
        indices = np.arange(A.shape[1])[col_mask]
        del col_mask, x_x_pinv_norm
        n = 50
        with warnings.catch_warnings(action='ignore', category=SparseEfficiencyWarning):
            for i in tqdm(range(0, len(indices), n), desc='Reconstructing known unitary columns'):
                A[:, indices[i : i + n]] = b_x_pinv[:, indices[i : i + n]]

        # with warnings.catch_warnings(action='ignore', category=SparseEfficiencyWarning):
        #     for i in tqdm(range(A.shape[0]), desc='Reconstructing known unitary columns'):
        #         # Create identity column vector for comparison
        #         identity_col = sparse.csc_matrix((A.shape[0], 1))
        #         identity_col[i] = 1.0

        #         # Check if x_x_pinv column is close to the identity column
        #         if sparsela.norm(x_x_pinv[:, i] - identity_col) < self.tol:
        #             A[:, i] = b_x_pinv[:, i]
        #         elif sparsela.norm(x_x_pinv[:, i]) > self.tol:
        #             raise RuntimeError(
        #                 'Cannot reconstruct matrix, likely because the columns of matrix x are linearly dependent'
        #             )

        # Convert to CSR format for efficiency
        return A.tocsr()

    def extend_columns_to_orthonormal_basis(self, U: sparse.spmatrix) -> sparse.spmatrix:
        '''Replace the zero columns of U with non-zero unit vectors such that the columns form an orthonormal basis,
        making U a unitary matrix. The supplied non-zero columns of U must already be orthonormal.
        '''

        Ucp = U.tocsc(copy=True)

        print('Identifying zero columns')
        zero_cols = []
        nonzero_cols = []
        for i in range(Ucp.shape[1]):
            if np.sum(np.abs(Ucp[:, i].data)) == 0:
                zero_cols.append(i)
            else:
                nonzero_cols.append(i)
        zero_cols = zero_cols[::-1]

        # Check for orthonormality of the supplied columns of Ucp
        for i, j in tqdm(
            combinations_with_replacement(nonzero_cols, 2),
            desc='Confirming orthonormality of nonzero columns',
            total=len(nonzero_cols) * (len(nonzero_cols) + 1) // 2,
        ):
            overlap = (Ucp[:, i].T.conj() @ Ucp[:, j])[0, 0]
            if i == j:
                assert np.abs(overlap - 1) < self.tol, f'Column {i} is not normalized'
            else:
                assert np.abs(overlap) < self.tol, f'Columns {i} and {j} are not orthogonal'
        print('Initial column orthonormality checks passed')

        dim_index = 0
        with tqdm(desc='Extending matrix columns to orthonormal basis', total=len(zero_cols)) as pbar:
            while zero_cols:
                j = zero_cols[-1]

                # Generate a candidate basis vector
                candidate = sparse.random(1, Ucp.shape[0], density=0.02, dtype=Ucp.dtype).tocsr()
                if candidate.count_nonzero() < 2:
                    candidate = sparse.random(1, Ucp.shape[0], density=0.5, dtype=Ucp.dtype).tocsr()

                # Orthogonalize against existing non-zero columns
                with warnings.catch_warnings(action='ignore', category=SparseEfficiencyWarning):
                    for k in nonzero_cols:
                        # Get k-th column
                        existing_col = Ucp[:, k]

                        # Subtract the projection or candidate onto existing_col
                        candidate -= candidate.dot(existing_col.conj())[0, 0] * existing_col.T

                        # Break if candidate is all zero
                        if not candidate.count_nonzero():
                            break
                    else:
                        # Normalize the vector
                        candidate /= sparsela.norm(candidate, ord='fro')

                        # Replace the zero column with the new orthonormal vector
                        Ucp[:, j] = candidate.T
                        zero_cols.pop()
                        nonzero_cols.append(j)
                        pbar.update()

                dim_index += 1

        # Final orthonormality check
        print('Checking orthonormality')
        for i in range(Ucp.shape[1]):
            for j in range(i, Ucp.shape[1]):
                overlap = (Ucp[:, i].T.conj() @ Ucp[:, j])[0, 0]
                if i == j:
                    if np.abs(overlap - 1) > self.tol:
                        print(f'WARNING: Column {i} is not normalized. Retrying orthonormalization.')
                        return self.extend_columns_to_orthonormal_basis(U)
                else:
                    if np.abs(overlap) > self.tol:
                        print(f'WARNING: Columns {i} and {j} are not orthogonal. Retrying orthonormalization.')
                        return self.extend_columns_to_orthonormal_basis(U)
        print('Column orthonormality checks passed')
        return Ucp

    def construct_unitary(self) -> sparse.csr_matrix:
        '''Steps 3 and 4 from [1].'''

        # Compute the initial and final states of the agents unitary transformation
        LHS_vectors: list[sparse.csc_matrix] = []
        RHS_vectors: list[sparse.csc_matrix] = []
        for x_ind, x in tqdm(
            enumerate(self.inputs), desc='Formulating underdetermined system', total=len(self.inputs)
        ):
            for s in self.causal_states:
                LHS_vectors.append(sparse.csc_matrix(self.initial_state_map[(x, s)]))

                RHS = 0
                for y_ind, y in enumerate(self.outputs):
                    coeff = np.sqrt(self.transition_probs(y, x, s))
                    state = multi_kron(
                        self.memory_state_map[self.update_rule(x, y, s)],
                        self.input_encodings[x_ind],
                        self.output_encodings[y_ind],
                        self.junk_state_map[(x, s)],
                    )
                    RHS += coeff * state
                RHS_vectors.append(sparse.csc_matrix(RHS))

        LHS = sparse.hstack(LHS_vectors)
        RHS = sparse.hstack(RHS_vectors)

        # Reconstruct the (not yet) unitary matrix U given by U @ LHS = RHS.
        U = self.reconstruct_matrix(LHS, RHS)

        print('Computing compiled states')
        self.compiled_initial_state_map = {k: Statevector(U @ v) for k, v in self.initial_state_map.items()}

        # Ensure the reconstruction was successful
        assert np.allclose((U @ LHS - RHS).data, 0, atol=self.tol), 'Unitary reconstruction failed'

        if not self.compute_full_unitary:
            return U.tocsr()

        full_unitary = self.extend_columns_to_orthonormal_basis(U)
        print('Unitary computed')

        eye = sparse.lil_matrix(full_unitary.shape)
        eye[range(full_unitary.shape[0]), range(full_unitary.shape[0])] = 1
        hopefully_zero = full_unitary.T.conj() @ full_unitary - eye
        assert (
            sparsela.norm(hopefully_zero) < self.tol
        ), f'Unitary construction failed. Matrix not unitary within tolerance: {sparsela.norm(hopefully_zero)} > {self.tol}'
        print('Unitary checks passed')

        return full_unitary.tocsr()

    def run_compiled_evolution(self, input_val: Hashable, causal_state: Hashable) -> Statevector:
        '''Given the causal state `s` and input `x` return the result of applying the evolution
        circuit as a Statevector.
        '''

        return self.compiled_initial_state_map[(input_val, causal_state)]


class PartialQuantumAgentEncoder:
    '''
    Encodes input-output processes in the quantum realm.

    References
    ----------
    [1] T. J. Elliott, M. Gu, A. J. P. Garner, and J. Thompson, “Quantum adaptive agents
        with efficient long-term memories,” Phys. Rev. X, vol. 12, p. 011007, Jan 2022.
    '''

    dtype = float

    def __init__(
        self,
        causal_states: list[Hashable],
        inputs: list[Hashable],
        outputs: list[Hashable],
        transition_probs: Callable[[Hashable, Hashable, Hashable], float],
        update_rule: Callable[[Hashable, Hashable, Hashable], Hashable],
        tol: float = 1e-12,
    ) -> None:
        '''
        Parameters
        ----------
        causal_states : list[Hashable]
            List of causal states of the adaptive agent.
        inputs : list[Hashable]
            List of inputs to the input-output process. If `input_encodings` is not specified,
            these should be non-negative integers.
        outputs : list[Hashable]
            List of outputs to the input-output process. If `output_encodings` is not specified,
            these should be non-negative integers.
        transition_probs : Callable[[Hashable, Hashable, Hashable], float]
            A function that takes an output value, an input value, and a causal state in that
            order as parameters and returns the transition probability, i.e.
            transition_probs(y, x, s) = P(y|x, s).
        update_rule : Callable[[Hashable, Hashable, Hashable], Hashable]
            A function that takes an input value, and output value, and a causal state in that
            order as parameters and returns a new causal state, i.e. update_rule(x, y, s) = s'.
        tol : float
            Tolerance for Hermitian/unitary checks, etc. Default: 1e-12
        '''

        # The user-defined causal states (any Hashable)
        self.causal_states = causal_states
        self.causal_state_pairs = list(product(self.causal_states, self.causal_states))
        # The user-defined inputs (any Hashable)
        self.inputs = inputs
        # The user-defined outputs (any Hashable)
        self.outputs = outputs
        # A function mapping tuples (output_state, input_state, causal_state) to the transition probability
        # P(output_state | input_state, causal_state)
        self.transition_probs = transition_probs
        # A function mapping tuples (input_state, output_state, causal_state) of states at time t to the updated
        # causal state at time t + 1
        self.update_rule = update_rule

        # Quantum state encodings of the input states
        self.input_encodings = self.encode_vals(self.inputs)
        # Quantum state encodings of the output states
        self.output_encodings = self.encode_vals(self.outputs)

        self.tol = tol

        # A map from the input states to their quantum encodings
        self.input_state_map = {x: x_encoded for x, x_encoded in zip(self.inputs, self.input_encodings)}

        # The unitary that defines the agent evolution
        self.unitary: sparse.csr_matrix
        # A map from the agents memory/causal states to their quantum state representations
        self.memory_state_map: dict[Hashable, sparse.spmatrix]

        # The number of qubits in the memory/causal states/tape
        self.n_qubits_memory_tape: int
        # The number of qubits in the input states/tape
        self.n_qubits_input_tape = int(math.log2(self.input_encodings[0].shape[0]))
        # the number of qubits in the output states/tape
        self.n_qubits_output_tape = int(math.log2(self.output_encodings[0].shape[0]))

        # A map from tuples (causal state, input) to the quantum state representation (as a numpy array)
        # of the **LHS** of the evolution equation (equation (3) from [1])
        self.initial_state_map: dict[tuple[Hashable, Hashable], sparse.spmatrix]

    @classmethod
    def expand_with_identity(cls, m: sparse.dok_matrix, d: int) -> sparse.dok_matrix:
        '''Add identity columns and rows to a square matrix m to expand it to shape (d, d)'''

        m = m.todok()

        if m.shape[0] != m.shape[1]:
            raise ValueError(f'`m` must be a square matrix: found dimensions {m.shape}')

        if m.shape[0] == d:
            return m

        result = sparse.dok_matrix((d, d), dtype=cls.dtype)
        result[: m.shape[0], : m.shape[1]] = m
        for i in range(m.shape[0], d):
            result[i, i] = 1.0
        return result

    @classmethod
    def encode_vals(cls, vals: list[Hashable]) -> list[sparse.csc_matrix]:
        '''Encode values as quantum states. This maps a list of hashables [h_0, ..., h_{n-1}]
        to computational basis states [|0>, ..., |n-1>].
        '''

        encodings = []
        n_qubits = math.ceil(math.log2(len(vals)))
        with warnings.catch_warnings(action='ignore', category=SparseEfficiencyWarning):
            for x in tqdm(range(len(vals)), desc=f'Encoding vals with {n_qubits} qubits'):
                encoding = sparse.csc_matrix((2**n_qubits, 1), dtype=cls.dtype)
                encoding[x, 0] = 1.0
                encodings.append(encoding)
        return encodings

    def encode(self) -> None:
        '''Encode the agent in the quantum realm.'''

        assert (
            self.check_update_rule_independence()
        ), f'Cannot use {self.__class__.__name__} when update rule is dependent on s'
        input_specialized_overlaps = self.compute_independent_input_specialized_overlaps()
        print('Constructing memory states')
        self.construct_quantum_memory_states(input_specialized_overlaps)
        print('Constructing unitary')
        self.unitary = self.construct_unitary()

    def check_update_rule_independence(self):
        '''
        Check whether or not the update rule is independent of causal state.  If it is,
        a significant simplification in the computation of memory state overlaps is possible:

        c_{lambda(z, s)lambda(z, s')} = 1  =>  c^x_{ss'} = sum_y(sqrt[P(y|x,s)P(y|x,s')])
        '''

        for x in tqdm(self.inputs, desc='Checking update rule dependence on causal states'):
            for y in self.outputs:
                s_updated = self.update_rule(x, y, self.causal_states[0])
                for s in self.causal_states:
                    if self.update_rule(x, y, s) != s_updated:
                        return False
        return True

    def compute_independent_input_specialized_overlaps(self):
        '''
        Step 1 from [1].
        Compute input specialized overlaps in the case where the update rule is independent
        of the memory state.
        '''

        input_specialized_overlaps = {}
        for x in tqdm(self.inputs, desc='Computing independent input specialized overlaps'):
            for s0, s1 in self.causal_state_pairs:
                overlap = sum(
                    math.sqrt(self.transition_probs(y, x, s0) * self.transition_probs(y, x, s1)) for y in self.outputs
                )
                input_specialized_overlaps[f'c|{x}|{s0}|{s1}'] = overlap
        return input_specialized_overlaps

    def construct_quantum_memory_states(self, input_specialized_overlaps: dict[str, float]):
        '''Step 2 from [1], but only for memory states.'''

        # Compute the memory state overlaps from the input-specialized memory state overlaps
        memory_state_overlaps = {}
        for s0, s1 in tqdm(self.causal_state_pairs, desc='Computing Gram matrix'):
            memory_state_overlaps[(s0, s1)] = prod(
                (input_specialized_overlaps[f'c|{x}|{s0}|{s1}'] for x in self.inputs)
            )

        # Compute the Gram matrix for the memory states
        memory_gram_matrix = sparse.dok_matrix((len(self.causal_states), len(self.causal_states)), dtype=self.dtype)
        for (i, s0), (j, s1) in product(enumerate(self.causal_states), enumerate(self.causal_states)):
            memory_gram_matrix[i, j] = memory_state_overlaps[(s0, s1)]

        # Expand the Gram matrix so that the dimension is a power of 2
        # This ensures memory states are vectors with a power of 2 length
        new_dim = 2 ** math.ceil(math.log2(memory_gram_matrix.shape[0]))
        memory_gram_matrix = self.expand_with_identity(memory_gram_matrix, new_dim).toarray()

        # Some sanity checks on the Gram matrix
        assert spla.ishermitian(memory_gram_matrix, atol=self.tol), 'Memory state Gram matrix not Hermitian'
        assert np.all(npla.eigvals(memory_gram_matrix) > 0), 'Memory state Gram matrix not positive definite'
        print('Gram matrix checks passed')

        # The Cholesky decomposition gives the vectors from the set of inner products
        print('Computing Cholesky decompositions')
        memory_cholesky_decomp = npla.cholesky(memory_gram_matrix)

        # Check that the row vectors reproduce the Gram matrix
        assert (
            npla.norm(memory_cholesky_decomp @ memory_cholesky_decomp.T.conj() - memory_gram_matrix, ord='fro')
            < self.tol
        ), 'Gram matrix decomposition failed'

        print('Formulating memory states')
        memory_states = [v.reshape(-1, 1) for v in memory_cholesky_decomp]
        self.memory_state_map = {s: sparse.csc_matrix(memory_states[i]) for i, s in enumerate(self.causal_states)}

        # Make sure overlaps are correct
        for s0, s1 in self.causal_state_pairs:
            overlap = (self.memory_state_map[s0].T @ self.memory_state_map[s1])[0, 0]
            assert (
                abs(memory_state_overlaps[(s0, s1)] - overlap) < self.tol
            ), f'Memory states from causal states ({s0}, {s1}) do not provide the correct overlap'
        print('Memory state checks passed')

        self.n_qubits_memory_tape = int(math.log2(next(iter(self.memory_state_map.values())).shape[0]))

        output = sparse.csc_matrix((2**self.n_qubits_output_tape, 1), dtype=self.dtype)
        with warnings.catch_warnings(action='ignore', category=SparseEfficiencyWarning):
            output[0, 0] = 1.0

        self.initial_state_map = {}
        # Compute all possible input_states
        for x, input_state in tqdm(self.input_state_map.items(), desc='Computing initial states'):
            for s, memory_state in self.memory_state_map.items():
                self.initial_state_map[(x, s)] = sparse_kron(
                    memory_state,
                    input_state,
                    output,
                )

    def reconstruct_matrix(self, x: sparse.spmatrix, b: sparse.spmatrix) -> sparse.csr_array:
        '''
        Reconstruct columns of a matrix A from x and b satisfying Ax=b.
        x and b should be matrices with the vectors as columns.
        '''

        # When ⁠A has linearly independent columns (as is the case here), pinv(A)⁠
        # can be computed as pinv(A) = (A^* @ A)^-1 @ A^*
        print(f'Computing pseudoinverse of {x.shape} matrix')
        with warnings.catch_warnings(action='ignore', category=SparseEfficiencyWarning):
            # This is a relatively inefficient method to compute the pseudoinverse. Could be
            # improved with a good SVD implementation
            x_pinv = sparsela.inv(x.T.conj() @ x) @ x.T.conj()

        # Compute b @ x_pinv and x @ x_pinv
        b_x_pinv = (b @ x_pinv).tocsc()
        x_x_pinv = (x @ x_pinv).todok()

        # Prepare the result matrix as a sparse matrix
        A = sparse.csc_matrix((x.shape[0], x.shape[0]), dtype=self.dtype)

        # Pre-subtract the identity and compute the norms to make the loop faster
        identity = sparse.dok_matrix(x_x_pinv.shape, dtype=self.dtype)
        for i in range(identity.shape[0]):
            identity[i, i] = 1
        x_x_pinv_mi = (x_x_pinv - identity).tocsr()  # norm converts to csr anyway
        x_x_pinv_mi_norm: np.ndarray = sparsela.norm(x_x_pinv_mi, ord=2, axis=0)
        del x_x_pinv_mi
        x_x_pinv_norm: np.ndarray = sparsela.norm(x_x_pinv, ord=2, axis=0)

        # for i in tqdm(range(A.shape[0]), desc='Reconstructing known unitary columns'):
        #     # Check if x_x_pinv column is close to the identity column
        #     if x_x_pinv_mi_norm[i] < self.tol:
        #         A[:, i] = b_x_pinv[:, i]
        #     # elif sparsela.norm(x_x_pinv[:, i]) > self.tol:
        #     #     raise RuntimeError(
        #     #         'Cannot reconstruct matrix, likely because the columns of matrix x are linearly dependent'
        #     #     )

        col_mask = x_x_pinv_mi_norm < self.tol
        del x_x_pinv_mi_norm
        if np.any(x_x_pinv_norm[~col_mask] > self.tol):
            raise RuntimeError(
                'Cannot reconstruct matrix, likely because the columns of matrix x are linearly dependent'
            )
        indices = np.arange(A.shape[1])[col_mask]
        del col_mask, x_x_pinv_norm
        n = 50
        with warnings.catch_warnings(action='ignore', category=SparseEfficiencyWarning):
            for i in tqdm(range(0, len(indices), n), desc='Reconstructing known unitary columns'):
                A[:, indices[i : i + n]] = b_x_pinv[:, indices[i : i + n]]

        # Convert to CSR format for efficiency
        return A.tocsr()

    def construct_unitary(self) -> sparse.csr_matrix:
        '''Steps 3 and 4 from [1].'''

        # Compute the initial and final states of the agents unitary transformation
        LHS_vectors: list[sparse.csc_matrix] = []
        RHS_vectors: list[sparse.csc_matrix] = []
        for x_ind, x in tqdm(
            enumerate(self.inputs), desc='Formulating underdetermined system', total=len(self.inputs)
        ):
            for s in self.causal_states:
                LHS_vectors.append(sparse.csc_matrix(self.initial_state_map[(x, s)]))

                RHS = 0
                for y_ind, y in enumerate(self.outputs):
                    coeff = np.sqrt(self.transition_probs(y, x, s))
                    state = sparse_kron(
                        self.memory_state_map[self.update_rule(x, y, s)],
                        self.input_encodings[x_ind],
                        coeff * self.output_encodings[y_ind],
                    )
                    # RHS += coeff * state
                    RHS += state
                RHS_vectors.append(sparse.csc_matrix(RHS))

        LHS = sparse.hstack(LHS_vectors)
        RHS = sparse.hstack(RHS_vectors)

        # Reconstruct the (not yet) unitary matrix U given by U @ LHS = RHS.
        U = self.reconstruct_matrix(LHS, RHS)

        # Ensure the reconstruction was successful
        assert np.allclose((U @ LHS - RHS).data, 0, atol=self.tol), 'Unitary reconstruction failed'

        return U.tocsr()

    def run_compiled_evolution(self, input_val: Hashable, causal_state: Hashable) -> Statevector:
        '''Given the causal state `s` and input `x` return the result of applying the evolution
        circuit as a Statevector.
        '''

        return Statevector((self.unitary @ self.initial_state_map[(input_val, causal_state)]).toarray())


class QuantumAgent:
    def __init__(self, encoder: QuantumAgentEncoder, initial_causal_state: Hashable):
        self.encoder = encoder

        # We have to reverse the qubit indices since Qiskit's qubit ordering is backwards
        self.output_tape_qubits = list(
            range(self.encoder.n_qubits_junk_tape, self.encoder.n_qubits_junk_tape + self.encoder.n_qubits_output_tape)
        )
        self.non_memory_state_qubits = list(
            range(
                self.encoder.n_qubits_junk_tape + self.encoder.n_qubits_output_tape + self.encoder.n_qubits_input_tape
            )
        )

        self.memory_state_shape = next(iter(self.encoder.memory_state_map.values())).shape

        self.classical_causal_states, quantum_causal_states = zip(*self.encoder.memory_state_map.items())
        self.stacked_quantum_causal_states = np.stack(quantum_causal_states)

        # The current classical causal state
        self.current_causal_state = initial_causal_state

    def step(self, input: Hashable, check_update_rule: bool = False):
        # Get the initial statevector with the unitary applied
        statevector = self.encoder.compiled_initial_state_map[(input, self.current_causal_state)]

        # Measure the output qubits
        outcome, statevector = statevector.measure(self.output_tape_qubits)
        output = self.encoder.outputs[int(outcome, base=2)]

        # Trace out the junk, output, and input states to obtain the new memory state
        quantum_causal_state = (
            partial_trace(statevector, self.non_memory_state_qubits)
            .to_statevector()
            ._data.reshape(self.memory_state_shape)
        )

        # Check that the new causal state agrees with the update rule
        if check_update_rule:
            assert np.allclose(
                quantum_causal_state,
                self.encoder.memory_state_map[self.encoder.update_rule(input, output, self.current_causal_state)],
            ), 'Update rule not satisfied'

        self.current_causal_state = self.get_classical_memory_state_from_quantum(quantum_causal_state)

        return output

    def get_classical_memory_state_from_quantum(self, quantum_memory_state):
        return min(
            self.encoder.memory_state_map.keys(),
            key=lambda k: npla.norm(quantum_memory_state - self.encoder.memory_state_map[k]),
        )


class LetterIterator:
    '''Counts in base 52 using lowercase and uppercase letters while skipping python keywords.'''

    import keyword

    def __init__(self):
        self.current = [0]

    def __iter__(self):
        return self

    def __next__(self):
        result = ''.join(chr((97 if x < 26 else 39) + x) for x in self.current)
        for i in range(len(self.current) - 1, -1, -1):
            self.current[i] = (self.current[i] + 1) % 52
            if self.current[i]:
                break
        else:
            self.current.insert(0, 0)
        if self.keyword.iskeyword(result):
            return next(self)
        return result


class Funcify:
    def __init__(self, equations: list[str], variables: list[str]):
        '''Convert a list of mathematical expressions into a function using `eval`.'''

        self.equations = equations
        self.variables = variables
        self.n_vars = len(self.variables)

        if len(equations) != self.n_vars:
            raise ValueError('Must have the same number of equations as variables')

    def __call__(self, x: list | np.ndarray) -> Callable:
        '''Evaluate the equations for a given input.'''

        assert len(x) == self.n_vars, f'func takes an input of length {self.n_vars}: found length {len(x)}'

        # Note x is converted to a list as it provides a significant performance improvement,
        # even with the call to `np.ndarray.tolist`.
        if isinstance(x, np.ndarray):
            x = x.tolist()

        local_vars = dict(zip(self.variables, x))
        global_vars = {'__builtins__': None}

        return np.fromiter((eval(eqn, global_vars, local_vars) for eqn in self.equations), dtype=complex)

    def compile_equations(self):
        print(f'Compiling {len(self.equations)} equations')
        self.equations = [compile(eqn, '<string>', 'eval') for eqn in self.equations]
