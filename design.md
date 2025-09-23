# GaussED: Architecture and Design

**GaussED** is a Python package implemented in **JAX** for general-purpose Gaussian Process (GP) inference and Sequential Experimental Design (SED).

Its design balances **generality** (supporting arbitrary domains, operators, and likelihoods) with **efficiency** (vectorised implementations and flexible linear solvers).

---

## Core Capabilities

GaussED provides two complementary pillars of functionality:

### 1. General Gaussian Process Inference

* **Flexible observations**: can condition on *general linear functionals* of Gaussian processes (e.g. function evaluations, derivatives, line-integrals).
* **Transformations**: supports transformations of GPs via linear operators.
* **Non-Gaussian likelihoods**: handled automatically using latent-variable augmentation.
* **Hyperparameter inference**: optimises kernel/likelihood parameters automatically, with support for gradient-based optimisation or sampling (e.g. MCMC).

### 2. Automatic Sequential Experimental Design (SED)

* **Acquisition-driven measurements**: select optimal measurements by defining linear functionals and maximising an acquisition function.
* **Multi-step design**: stack multiple functionals for adaptive experimental design across multiple rounds.

---

## Modes of Operation

Two main computational modes are supported:

1. **`matvec` mode**

   * All linear operators are represented via their `matvec` actions.
   * Efficient for large-scale problems where explicit storage is infeasible.

2. **`matrix` mode**

   * Linear operators are explicitly stored as matrices (via thin wrappers around `LinearOperator`).
   * Useful for small- to medium-scale problems and debugging.

---

## Project Structure

```
gaussed/
│
├── engines/                # Inference engines, solvers, SED
│   ├── latent/             # Non-Gaussian likelihood inference
│   ├── sed/                # Sequential Experimental Design
│   └── solver/             # Linear solvers
│
├── gp/                     # GP backends and user API
│   ├── backends/           # Kernel, inducing point, operator backends
│   ├── gp_ops/             # Operators acting on GPs
│   ├──── base.py           # Base operator
│   ├── kernels/            # Kernel definitions
│   ├──── base.py           # Base Kernel class
│   ├── base.py             # Base Gaussian Process definition
│   └── means.py            # Mean function base and other mean functions
├── domains/                # Domain types (input space definitions)
├── codomains/              # Codomain definitions (output spaces)
├── utils/                  # Utility functions
│   ├── constraints.py      # Handles parameter constraints.
│   └── geometry.py         # Handles safe norm.
└── render.py               # Graph rendering for computational pipelines
```

Design note:
Custom **PyTrees** are used for data structures (rather than `eqx.Module`) to avoid extra dependencies and to maintain a lightweight codebase.

---

## Gaussian Processes

Gaussian Processes in GaussED are defined with explicit separation of components:

```python
GP(
    mean: MeanFunction,
    kernel: Kernel
    domain: Domain,
    codomain: Codomain,
    backend: Backend
)
```

* **`output_shape`**: shape of the GP output, e.g. `(d1, d2, ...)`.
* **`batch_shape`**: broadcasting dimensions, e.g. `(B, d1, d2, ...)`.

This modular design makes it possible to extend GPs with new operators, domains, or codomains without altering inference logic.

### GP Representations

GaussED currently supports three types of GP inference:

1. **Kernel-based Representation**

   * Uses the full covariance kernel for exact GP inference.
   * Suitable for small- to medium-scale problems.

2. **Inducing points Representation**

   * Sparse GP approximation via inducing points.
   * Enables scaling to large datasets.

3. **Operator / Basis-function Representation**

   * Interprets the GP as Bayesian linear regression in a basis-function space.
   * Useful for specialised applications where an operator structure is known.

These representations are automatically inferred from the choice of `Backend`.


---

## Linear Operators and Conditioning

Applying linear functionals to Gaussian processes is handled via the `Probe` interface. A `Probe` object interfaces with a GP to compute a linear functional $L$ applied to a GP ($L f$). For each probe, only an `apply(f: Func, ctx) -> LinearOp` method must be provided.

Depending on the GP representation, we:

1. **Kernel-based and Inducing Points Representation:** Any cross covariance $K(A,B)$ is computed via nested applications $$ K(A,B) = A\cdot (X \mapsto B\cdot(Y\mapsto k(X,Y))) $$
2. **Basis-Function:** A cross covariance is computed as $\Phi^\top \Phi$ which is achieved via an `apply` to `BasisMap.phi`.

---
## Backends

A `Backend` is the *glue* that connects explicit numerical computations, the choice of GP numerics and computing Posterior GPs.

 A `Backend` contains a `Solver` object/

At the **user level**, the API remains simple:

* specify a mean and kernel,
* optionally choose a backend,
* the inference engine internally selects the correct routines.

---

## Design Philosophy

1. **Generality**: any linear functional of a GP is valid for inference.
2. **Composability**: inference engines, operators, and solvers are modular.
3. **Performance**: JAX transformations (`jit`, `vmap`, `grad`) are used throughout for efficiency.
4. **Minimal dependencies**: avoids unnecessary external packages (e.g. using PyTrees over Equinox).
5. **Transparency**: computational pipelines can be visualised via `render.py`.

---

## Future Directions

* Advanced acquisition functions for **Bayesian optimal experimental design**.
* Improved linear solvers (e.g. multi-grid methods).
* Integration with probabilistic programming frameworks for end-to-end workflows.

---
