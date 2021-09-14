import torch

from GaussED.distribution.base import Distribution


class Design(Distribution):

    def __init__(self, basis_matrix, design_sampler, initial_design=None, sample_domain=None, matheron_sampler=None):
        self.basis_matrix = basis_matrix
        self.sample = design_sampler
        self.matheron_sample = matheron_sampler

        self.initial_design = initial_design
        self.sample_domain = sample_domain

        self.transform = None
        self.inverse_transform = None

    def update_basis_matrix(self, basis_matrix, design_point, m):
        return torch.cat([basis_matrix, self.basis_matrix(design_point.unsqueeze(0), m)])

    def set_sample(self, sampling_func):
        self.sample = sampling_func

    def set_basis_matrix(self, basis_matrix):
        self.basis_matrix = basis_matrix

    def set_sample_domain(self, sample_domain):
        self.sample_domain = sample_domain

    def set_transform(self, transform, inverse_transform):
        self.transform = transform
        self.inverse_transform = inverse_transform


class EvaluationDesign(Design):

    def __init__(self, observables, initial_design=None):

        if type(observables) is list:
            self.observables = observables
        else:
            self.observables = [observables]

        self.dim = self.observables[0].dim

        self.initial_design = initial_design
        self.d_lower, self.d_upper = self.observables[0].domain.T

    def set_domain(self, domain):
        self.d_lower, self.d_upper = domain.T

    def basis_matrix(self, design, m):
        basis_mats = []
        for i in self.observables:
            basis_mats.append(i.basis_matrix(self.transform(design), m))  #used to return torch.cat(basis_mats)

        out = torch.stack(basis_mats, dim=1)  # this interleaves the basis matrices
        return out.flatten(0, 1)

    def sample(self, d, mean, cov, n, random_sample=None, solver=None, sqrt=None):
        x = self.transform(d)
        samples = []
        for i in self.observables:
            samples.append(i.sample_mesh(mean, cov, x.unsqueeze(0), n, random_sample=random_sample, solver=solver, sqrt=sqrt))
        return torch.cat(samples).T

    def matheron_sample(self, d, phi_mat, y, n, random_sample=None, nugget=None, solver=None, inverse=None):
        x = self.transform(d)
        samples = []
        for i in self.observables:
            samples.append(i.matheron_sample_mesh(phi_mat, y, x.unsqueeze(0), n, random_sample=random_sample,
                                                  nugget=nugget, solver=solver, inverse=inverse))
        return torch.cat(samples).T

    def transform(self, d):
        return self.d_lower + (self.d_upper - self.d_lower) * torch.sigmoid(d)

    def sample_domain(self, N=None):
        sample_domain_dist = torch.distributions.Uniform(self.d_lower, self.d_upper)
        if N is None:
            return self.inverse_transform(sample_domain_dist.sample((1,))).flatten()
        else:
            return self.inverse_transform(sample_domain_dist.sample((N,)))

    def inverse_transform(self, x):
        return torch.log((x - self.d_lower) / (self.d_upper - self.d_lower)) - \
               torch.log(1 - (x - self.d_lower) / (self.d_upper - self.d_lower))

    def set_initial_design(self, initial_design):
        self.initial_design = initial_design
