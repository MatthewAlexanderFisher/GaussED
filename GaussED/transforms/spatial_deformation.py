import torch, math


class WarpClass:

    def __init__(self, mean, dist_cdfs, params, eps, weights, domain=torch.Tensor([[-1,1],[-1,1]])):

        self.mean = mean
        self.params = params
        self.eps = eps
        self.weights = weights
        self.param_length = len(params)
        self.n_components = len(weights)

        self.dist_cdfs = dist_cdfs

        if self.n_components != len(self.dist_cdfs):
            raise ValueError("Number of mixture components doesn't match initial weights.")

        self.d_lower, self.d_upper = domain.T

    def forward(self, x):
        x_t = self.inverse_transform(x)
        mean_t = self.transform(self.mean)
        eps_t = -torch.exp(self.eps)
        dists = torch.cdist(x_t, mean_t.unsqueeze(0))

        warping  = self.rad_basis(dists, self.params,  self.weights)
        out = x_t + eps_t * (mean_t - x_t) * warping
        return self.transform(out)

    def rad_basis(self, d, params, weights):
        weights_t = torch.exp(weights) / torch.sum(torch.exp(weights))
        out = 0
        for i in range(self.n_components):
            out = out + weights_t[i] * self.dist_cdfs[i](d, params[i])
        return out

    def transform(self, d):
        return self.d_lower + (self.d_upper - self.d_lower) * torch.sigmoid(d)

    def inverse_transform(self, x):
        return torch.log((x - self.d_lower) / (self.d_upper - self.d_lower)) - \
               torch.log(1 - (x - self.d_lower) / (self.d_upper - self.d_lower))

    def set_domain(self, domain):
        self.d_lower, self.d_upper = domain.T


def uniform_cdf(x, params):
    return x

def gauss_cdf(x, params):
    xt = torch.log(x) - torch.log(1 - x)
    mean, var = params
    return 1/2 * (1 + torch.erf((xt - mean) / (torch.exp(var) * math.sqrt(2)))) # normal.cdf(t_x)


class MixtureInputWarp:

    def __init__(self, dist_cdfs, params1, params2, weights1, weights2, domain=torch.Tensor([[-1,1],[-1,1]])):

        self.n_components = len(weights1)

        self.dist_cdfs = dist_cdfs
        if self.n_components != len(self.dist_cdfs):
            raise ValueError("Number of mixture components doesn't match initial weights.")

        self.params1 = params1
        self.weights1 = weights1

        self.params2 = params2
        self.weights2 = weights2

        self.d_lower, self.d_upper = domain.T
        self.a0, self.b0 = domain[0]
        self.a1, self.b1 = domain[1]

    def forward(self, x):
        x_t = ((x - self.d_lower) / (self.d_upper - self.d_lower)).T

        out1 = 0
        out2 = 0

        w1_t = torch.exp(self.weights1) / torch.sum(torch.exp(self.weights1))
        w2_t = torch.exp(self.weights2) / torch.sum(torch.exp(self.weights2))

        for i in range(self.n_components):
            out1 += w1_t[i] * self.dist_cdfs[i](x_t[0], self.params1[i])
            out2 += w2_t[i] * self.dist_cdfs[i](x_t[1], self.params2[i])

        out = torch.stack([out1, out2]).T
        out_t = out * (self.d_upper - self.d_lower) + self.d_lower

        return out_t

    def warp(self, x, dim=0):
        if dim == 0:
            weights = torch.exp(self.weights1) / torch.sum(torch.exp(self.weights1))
            params = self.params1
            a, b = self.a0, self.b0
        else:
            weights = torch.exp(self.weights2) / torch.sum(torch.exp(self.weights2))
            params = self.params2
            a, b = self.a1, self.b1

        x_t = (x - a) / (b - a)
        out = 0

        for i in range(self.n_components):
            out += weights[i] * self.dist_cdfs[i](x_t, params[i])

        out *= (b - a)
        out += a
        return out

    def transform(self, d):
        return self.d_lower + (self.d_upper - self.d_lower) * torch.sigmoid(d)

    def inverse_transform(self, x):
        return torch.log((x - self.d_lower) / (self.d_upper - self.d_lower)) - \
               torch.log(1 - (x - self.d_lower) / (self.d_upper - self.d_lower))

    def inverse_transform2(self, x):
        return torch.log((x - self.d_lower[0]) / (self.d_upper[0] - self.d_lower[0])) - \
               torch.log(1 - (x - self.d_lower[0]) / (self.d_upper[0] - self.d_lower[0]))

    def set_domain(self, domain):
        self.d_lower, self.d_upper = domain.T

def uniform_cdf(x, params):
    return x

def half_normal_cdf(x, scale):
    scale_t = torch.exp(scale)
    normal = torch.distributions.HalfNormal(scale_t)
    return normal.cdf(x)

def log_normal_cdf(x, params):
    mean, scale = params
    scale_t = torch.exp(scale)
    normal = 1/2 + 1/2 * torch.erf((torch.log(x) - mean) / (math.sqrt(2) * scale_t))#torch.distributions.LogNormal(mean, scale_t)
    return normal


def warpy(x, mean=torch.Tensor([[0.5,0.5],[0.7,0.7]]), eps=torch.Tensor([-1,-1]), l=torch.Tensor([0.1,0.1]), weights=torch.Tensor([0,0])):
    x_t = torch.log(x) - torch.log(1 - x)
    mean_t = torch.log(mean) - torch.log(1 - mean)
    out = 0
    weights_t = torch.exp(weights) / torch.sum(torch.exp(weights))
    for i in range(len(mean)):
        dists = torch.cdist(x_t, mean_t[i].unsqueeze(0))
        out = out + weights_t[i] * (x_t + eps[i] * (mean_t[i] - x_t) * torch.exp(-(dists.pow(2) / l[i])))
    return torch.sigmoid(out)

def warpy2(x, mean=torch.Tensor([[0.1,0.5],[0.7,0.7],[0.9,.2]]), eps=torch.Tensor([1,1,-1]), covs=torch.Tensor([[10,10,-6],[3,3,8],[1,1,0]]), weights=torch.Tensor([4,4,4])):
    x_t = torch.log(x) - torch.log(1 - x)
    x_t_T = x_t.T
    mean_t = torch.log(mean) - torch.log(1 - mean)
    out = 0
    weights_t = torch.exp(weights) / torch.sum(torch.exp(weights))
    w_sum = 0
    for i in range(len(mean)):
        dist1 = (x_t_T[0] - mean_t[i][0]).pow(2)
        dist2 = (x_t_T[1] - mean_t[i][1]).pow(2)
        a, b, rho = covs[i]
        rho = 2 * torch.sigmoid(rho) - 1
        matrix = torch.Tensor([[a.pow(2), rho * a * b], [rho * a * b, b.pow(2)]])
        weight_func = weights_t[i] * torch.exp(-(dist1 / 2) - (dist2 / 2)).unsqueeze(1)
        w_sum = w_sum + weight_func
        out = out +  weight_func * (x_t + eps[i] * (mean_t[i] - x_t) * torch.diag(torch.exp(-torch.matmul(x_t - mean_t[i], torch.matmul(matrix , (x_t - mean_t[i]).T)))).unsqueeze(1))
    out = out / w_sum
    return torch.sigmoid(out)


def warpy3(x, mean=torch.Tensor([[0.1,0.5],[0.7,0.7],[0.9,.2]]), eps=torch.Tensor([1,1,-1]), covs=torch.Tensor([[10,10,-6],[3,3,8],[1,1,0]]), weights=torch.Tensor([4,4,4])):
    x_t = torch.log(x) - torch.log(1 - x)
    mean_t = torch.log(mean) - torch.log(1 - mean)
    weights_t = torch.exp(weights) / torch.sum(torch.exp(weights))
    out = 0
    for i in range(len(mean)):
        a, b, rho = covs[i]
        rho = 2 * torch.sigmoid(rho) - 1
        matrix = torch.Tensor([[a.pow(2), rho * a * b], [rho * a * b, b.pow(2)]])
        out = out +  weights_t[i] * (x_t + eps[i] * (mean_t[i] - x_t) * torch.exp(-torch.sum((x_t - mean_t[i]) * torch.matmul(matrix , (x_t - mean_t[i]).T).T, dim=1)).unsqueeze(1))
    out = out
    return torch.sigmoid(out)

class WarpClass:

    def __init__(self, mean, dist_cdfs, params, eps, weights, domain=torch.Tensor([[-1,1],[-1,1]])):

        self.mean = mean
        self.params = params
        self.eps = eps
        self.weights = weights
        self.param_length = len(params)
        self.n_components = len(weights)

        self.dist_cdfs = dist_cdfs

        if self.n_components != len(self.dist_cdfs):
            raise ValueError("Number of mixture components doesn't match initial weights.")

        self.d_lower, self.d_upper = domain.T

    def forward(self, x):
        x_t = self.inverse_transform(x)
        mean_t = self.transform(self.mean)
        eps_t = -torch.exp(self.eps)
        dists = torch.cdist(x_t, mean_t.unsqueeze(0))

        warping  = self.rad_basis(dists, self.params,  self.weights)
        out = x_t + eps_t * (mean_t - x_t) * warping
        return self.transform(out)

    def rad_basis(self, d, params, weights):
        weights_t = torch.exp(weights) / torch.sum(torch.exp(weights))
        out = 0
        for i in range(self.n_components):
            out = out + weights_t[i] * self.dist_cdfs[i](d, params[i])
        return out

    def transform(self, d):
        return self.d_lower + (self.d_upper - self.d_lower) * torch.sigmoid(d)

    def inverse_transform(self, x):
        return torch.log((x - self.d_lower) / (self.d_upper - self.d_lower)) - \
               torch.log(1 - (x - self.d_lower) / (self.d_upper - self.d_lower))

    def set_domain(self, domain):
        self.d_lower, self.d_upper = domain.T

# from scipy import stats
# x_2D = torch.linspace(-1,0.99,80)
# y_2D = torch.linspace(-0.99,0.99,80)
# X_2D,Y_2D = torch.meshgrid(x_2D,y_2D)
# mesh = torch.stack([X_2D,Y_2D]).T.reshape(X_2D.shape[0]**2,2)
# t_mesh = warper.forward(mesh).detach()
#
# kernel1 = stats.gaussian_kde(mesh.numpy().T, 0.05) # using bandwidth 0.05
# kernel2 = stats.gaussian_kde(t_mesh.numpy().T)
#
# fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 6))
# ax1.contourf(Y_2D.numpy(), X_2D.numpy(), kernel1(torch.stack([X_2D,Y_2D]).T.reshape(X_2D.shape[0]**2,2).T.numpy()).reshape(80,80), levels=100)
# ax2.contourf(Y_2D.numpy(), X_2D.numpy(), kernel2(torch.stack([X_2D,Y_2D]).T.reshape(X_2D.shape[0]**2,2).T.numpy()).reshape(80,80), levels=100)
# plt.show()


# import torch


# class ExperimentWarp:
#
#     def __init__(self, gp, black_box, design, acquisition, m, data=None):
#         self.gp = gp
#         self.black_box = black_box
#         self.design = design
#
#         self.acquisition = acquisition
#         self.acquisition.eval_params["m"] = m
#
#         self.m = m
#
#         self.domain = gp.domain
#
#         self.optimise_method = torch.optim.Adam
#         self.optimiser_params = {"lr": 1e-1}
#         self.optimiser_steps = 1000
#
#
#         self.parameter_optimiser_params = {"lr": 1e-3}
#         self.parameter_optimiser_steps = 1000
#         self.parameter_optimiser = None
#
#         self.lam = 150
#
#         self.current_design = design.initial_design
#         self.basis_mat = self.design.basis_matrix(self.current_design, self.m)
#
#         if data is None:
#             self.data = self.black_box(self.current_design).flatten()
#         else:
#             self.data = data
#
#     def run(self, n, optimise_loss=True, optimise_kernel_params=True, retain_graph=True, print_step=False, debug=False):
#
#         current_design = self.current_design
#         current_data = self.data
#         current_phi_mat = self.design.basis_matrix(current_design, self.m)
#
#         for i in range(n):
#
#             x_2D = torch.linspace(-1.2,1.2,61)
#             y_2D = torch.linspace(-1.2,1.2,61)
#             X_2D,Y_2D = torch.meshgrid(x_2D,y_2D)
#             mesh = torch.stack([X_2D,Y_2D]).T.reshape(X_2D.shape[0]**2,2)
#             t_mesh = mixture_input_warp.forward(mesh).detach() # input_warp(mesh).detach()
#
#             fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 6))
#             ax1.scatter(mesh.T[0],mesh.T[1], s=5)
#             ax2.scatter(t_mesh.T[0],t_mesh.T[1],s=5)
#             plt.show()
#
#             ax = circ.plot_experiment(design_transform1(current_design[0]).unsqueeze(1),1000)
#             for k in current_design:
#                 param = design_transform1(k).unsqueeze(1)
#                 circ.plot_experiment_lines(ax,param)
#             circ.plot_experiment_lines(ax,design_transform1(current_design[-1]).unsqueeze(1),color="blue")
#             plt.show()
#
#
#             if i > 20:
#                 parameter_optimiser_steps = 200
#             else:
#                 parameter_optimiser_steps = self.parameter_optimiser_steps
#
#             if print_step is True:
#                 print("Step ", i+1, "/", n)
#
#             d = self.design.sample_domain().detach().requires_grad_(True)
#
#             optimiser = self.optimise_method([d], **self.optimiser_params)
#
#             if optimise_kernel_params is True and i > 7:
#                 warp_optimiser = self.optimise_method(self.gp.kernel.parameters + [warper.mean, warper.eps, warper.weights] + warper.params, **self.parameter_optimiser_params)
#
#                 for j in range(parameter_optimiser_steps): #radial_mixture_deformation
#                     warp_optimiser.zero_grad()
#                     L = - self.gp.log_likelihood(current_phi_mat, current_data, solver=self.acquisition.solver, nugget=self.acquisition.nugget)  #+ self.lam * warp_regulariser(mesh_loss, warping_func=radial_mixture_deformation.forward)
#                     L.backward(retain_graph=retain_graph)
#                     if debug is True and j % 20 == 0:
#                         print("Step: ", j, ", Loss: ", L)
#                     warp_optimiser.step()
#                     current_phi_mat = self.design.basis_matrix(current_design, self.m)
#
#             if optimise_loss is True:
#                 for j in range(self.optimiser_steps):
#                     optimiser.zero_grad()
#
#                     out = self.acquisition.eval(d, current_phi_mat.detach(), current_data, **self.acquisition.eval_params)
#                     out.backward()
#                     #out.backward(retain_graph=retain_graph)
#                     optimiser.step()
#
#             current_design = torch.cat([current_design, d.detach().unsqueeze(0)])
#             current_data = torch.cat([current_data, self.black_box(d.detach())])
#             current_phi_mat = self.design.basis_matrix(current_design, self.m)
#
#             if debug:
#                 print("start: ", starting_point, ". end: ", self.design.design_transform(d.detach()))
#
#             self.set_current_design(current_design)
#             self.set_data(current_data)
#             self.set_basis_mat(current_phi_mat)
#
#     def set_basis_mat(self, basis_mat):
#         self.basis_mat = basis_mat
#
#     def set_current_design(self, current_design):
#         self.current_design = current_design
#
#     def set_data(self, data):
#         self.data = data
#
#     def set_optimiser_method(self, method):
#         self.optimise_method = method
#
#     def set_optimiser_steps(self, optimiser_steps):
#         self.optimiser_steps = optimiser_steps
#
#     def set_parameter_optimiser(self, parameter_optimiser):
#         self.parameter_optimiser = parameter_optimiser
#
#     def set_optimiser_params(self, optimiser_params):
#         """
#         Sets the optimiser parameters
#         :param optimiser_params: a dictionary of input parameters to torch.optimiser
#         """
#         self.optimiser_params = optimiser_params
#
#     def set_acquisition_params(self, acquisition_params):
#         self.acquisition.eval_params = acquisition_params
#         self.m = acquisition_params.get("m")
