# GaussED

## TODO:

My immediate goals:

- Enforce shape conventions via Domain/Codomain as "input_shape" and "output_shape" respectively. (done)
- Test and fix operators (start with partial derivative). I probably should implement a very basic manual computation or something just to check (done partial - will leave quad later)
- Implement and test one basis function approach and make sure the API glue holds strong.
- Implement a GP sampling function (need to think about how this interacts with the LinearSolver)

After the basic API is fully working, it will be easy to implement new methods like new linear solvers or different GP approximations.

The system that has not been designed at all yet is experimental design. And how to wrap everything into a nice automatic SED bundle. I think it makes sense to leave this till last, since it is a  different system. My rough thinking was:

- Design function protocol to get observations
- Define "measurements" or "experiment" protocol to define what experiments we want to perform.
- Be able to stack measurements into blocks (e.g. take three measurements at one instance (a block) and then subsequent blocks will be in time - future experiments)
- Build a nice convenient wrapper to perform SED
- Need to implement optimisation algorithms for optimising acquisition functions (need to standardise acquisition protocol as well)


## Shape Conventions

- **Mean functions**: `MeanFunc: (n, input_shape) -> (n, output_shape)`
- **Kernel functions**: `Kernel: (nF, input_shape), (nG, input_shape) -> (nF, nG, output_shape, output_shape)`

### `LinOp`:

`LinOp` objects are always matrices (shape `(n,m)`)

### `input_shape`, `output_shape` and `n`

Mean functions: `MeanFunc: (n, input_shape) -> (n, output_shape)`