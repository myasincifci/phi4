import torch

class ReparamKL(torch.nn.Module):
    """
    Calculates the loss

    .. math::

        L = 1/N \sum_{i=1}^N ( S(g(z_i)) - log det dg/dz(z_i) + q_Z(z_i) )

    where :math:`z_i \sim p_Z`.
    Minimizing this loss is equivalent to minimizing KL(q, p) from the
    variational distribution q to the target Boltzmann distribution p=1/Z exp(-S).
    """
    def __init__(self, model, action, lat_shape, batch_size):
        super().__init__()
        self.model = model
        self.action = action
        self.lat_shape = lat_shape
        self.batch_size = batch_size

    def forward(self):
        try:
            samples, log_probs = self.model.sample_with_logprob(self.batch_size)
        except:
            samples, log_probs = self.model.sample_and_log_prob(self.batch_size)

        actions = self.action(samples.view(-1, *self.lat_shape))

        loss = actions + log_probs

        loss_mean = loss.mean()

        return loss_mean, loss, actions, samples
