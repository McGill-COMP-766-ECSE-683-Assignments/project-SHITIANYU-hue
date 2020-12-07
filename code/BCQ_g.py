import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import stable_dynamics as dynmod
from torch.utils.tensorboard import SummaryWriter
from torch.nn import init, Parameter
import math
from torch.autograd import Variable


# Noisy linear layer with independent Gaussian noise
class NoisyLinear(nn.Linear):
	def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
		super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
		# µ^w and µ^b reuse self.weight and self.bias
		self.sigma_init = sigma_init
		self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
		self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
		self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
		self.register_buffer('epsilon_bias', torch.zeros(out_features))
		self.reset_parameters()

	def reset_parameters(self):
		if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
			init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
			init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
			init.constant(self.sigma_weight, self.sigma_init)
			init.constant(self.sigma_bias, self.sigma_init)

	def forward(self, input):
		return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

	def sample_noise(self):
		self.epsilon_weight = torch.randn(self.out_features, self.in_features)
		self.epsilon_bias = torch.randn(self.out_features)

	def remove_noise(self):
		self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
		self.epsilon_bias = torch.zeros(self.out_features)



class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05,no_noise=False,sigma_init=0.017):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		if no_noise:
			self.l3 = nn.Linear(300, action_dim)
		else:
			self.l3=NoisyLinear(300, action_dim, sigma_init=sigma_init)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a))
		return (a + action).clamp(-self.max_action, self.max_action)

	def sample_noise(self):
		self.l3.sample_noise()

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim,no_noise=False,sigma_init=0.017):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		if no_noise:
			self.l3 = nn.Linear(300, 1)
			self.l6 = nn.Linear(300, 1)
		else:
			self.l3=NoisyLinear(300, 1, sigma_init=sigma_init)
			self.l6=NoisyLinear(300,1,sigma_init=sigma_init)

	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

	def sample_noise(self):
		self.l3.sample_noise()
		self.l6.sample_noise()

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std *torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
		


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		state_dim=state_dim*2
		latent_dim = action_dim * 2

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device


	def select_action(self, state, goal):
		state=np.hstack((state,goal))
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100,no_noise=False):
		risks=[]
		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done, goal= replay_buffer.sample(batch_size)
			risk=dynmod.Dynamics(lsd=state.size()[1]).to(self.device)(state,goal)
			risks.append(risk.item())
			# print('risk',risk.item())
			state=torch.cat([state,goal],dim=1)
			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss
			# print('vae_loss',vae_loss)
			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()
			if not no_noise:
				self.critic.sample_noise()
				self.actor.sample_noise()

			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state_new=torch.cat([next_state,goal],dim=1)
				next_state = torch.repeat_interleave(next_state_new, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * target_Q
			# print('target Q',target_Q.sum())
			current_Q1, current_Q2 = self.critic(state, action)
			##
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)+5*risk
			# print('critic loss',critic_loss)
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()


			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
			# print('actor_loss',actor_loss)
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()


			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return risks