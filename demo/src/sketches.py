from .imports import *
from omnilearn import *
from omnilearn.op import *



class AE(Model):
    latent_dim = hparam()
    
    encoder = part.function(inherit=True)(input='observation', output='latent')
    decoder = part.function(inherit=True)(input='latent', output='reconstruction')

    criterion = part.criterion('mse')(input='reconstruction', target='observation', output='loss')

    @space('latent')
    def latent_space(self) -> spaces.Vector:
        return spaces.Vector(self.latent_dim)
    @space('observation')
    def observation_space(self, reconstruction = None) -> spaces.Vector:
        if reconstruction is None:
            raise GearFailed
        return reconstruction
    @space('reconstruction')
    def reconstruction_space(self, observation = None) -> spaces.Vector:
        if observation is None:
            raise GearFailed
        return observation


class VAE(AE):
    encoder = submodule(inherit=True)(input='observation', output='posterior_params')

    @space('posterior_params')
    def posterior_param_space(self, latent: spaces.Vector) -> spaces.Normal:
        return spaces.Vector(2 * latent.size)

    @tool('posterior')
    def get_posterior(self, posterior_params):
        mean, logvar = posterior_params.chunk(2, dim=-1)
        return torch.distributions.Normal(mean, logvar.exp().sqrt())
    
    @tool('latent')
    def sample_latent(self, posterior):
        return posterior.rsample()
    
    @tool('mean')
    def get_mean(self, posterior):
        return posterior.mean()
    
    @indicator('kl_divergence')
    def compute_kl_divergence(self, posterior):
        return torch.distributions.kl_divergence(posterior, torch.distributions.Normal(0, 1)).sum(dim=-1).mean()

class StaticVarianceVAE(VAE):
    encoder = submodule(inherit=True)(input='observation', output='posterior_mean')

    @submodule(inherit=True)
    def posterior_std(self):
        return nn.Parameter(torch.zeros(self.latent_dim))
    
    @space('posterior_mean')
    def posterior_param_space(self, latent: spaces.Vector) -> spaces.Normal:
        return spaces.Vector(latent.size)

    @tool('posterior')
    def get_posterior(self, posterior_mean):
        std = self.posterior_std.unsqueeze(0).expand(posterior_mean.size(0), -1)
        return torch.distributions.Normal(posterior_mean, std)
    



class Classification(Machine):
    classifier = submodule(inherit=True)(input='observation', output='prediction')
    
    loss = part.criterion('bce')(input='prediction', target='label', output='loss')

    @space('prediction')
    def prediction_space(self, label: spaces.Categorical) -> spaces.Logits:
        return spaces.Logits(label.classes)



class NormAE(AE):
    p = hparam(2)

    @tool('latent')
    def encode(self, latent):
        return latent / latent.norm(dim=-1, p=self.p, keepdim=True)




