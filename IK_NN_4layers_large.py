import numpy as np
import torch
import torch.nn.functional as F
import argparse, os
from gazebo import *
from euler_convert import euler2quat,quat2euler
device = "cuda"

class IK_NN(torch.nn.Module):
	def __init__(self, n_feature, n_neuron, n_output):
		super(IK_NN,self).__init__()
		self.l1 = torch.nn.Linear(n_feature,n_neuron)
		self.l2 = torch.nn.Linear(n_neuron,600)
		self.l3 = torch.nn.Linear(600,600)
		self.l4 = torch.nn.Linear(600,n_neuron)
		self.out = torch.nn.Linear(n_neuron,n_output)
    
	def forward(self,x):
		x = F.tanh(self.l1(x))
		x = F.tanh(self.l2(x))
		x = F.tanh(self.l3(x))
		x = F.tanh(self.l4(x))
		x = self.out(x)
		return x


def load_model():
	model_save_path = "/home/ccchang/TD3/saved_models/4layers_large"
	resume_model = "IK_model0.016130905598402023"
	model = IK_NN(n_feature=6, n_neuron=300, n_output=6)
	device = "cuda"
	if device == "cuda":
		###Load on GPU
		model = model.cuda()
		checkpoint = torch.load('{0}/{1}'.format(model_save_path, resume_model),map_location=device)
		model.load_state_dict(checkpoint)

	else:
		###Load on CPU      
		device = torch.device('cpu')
		checkpoint = torch.load('{0}/{1}'.format(model_save_path, resume_model),map_location=device)
		model.load_state_dict(checkpoint)
	return model


def get_random_joint_angle():
	# pose_range = [[-1.57,1.57],[-0.75,2.1],[-0.75,2.1],[-1.57,1.57],[-1.57,1.57],[-1.57,1.57]]
	act = [np.random.uniform(-1.57,1.57),np.random.uniform(-0.75,2.1),np.random.uniform(-0.75,2.1),
	       np.random.uniform(-1.57,1.57),np.random.uniform(-1.57,1.57),np.random.uniform(-1.57,1.57)]
	return act


def get_batch_tensor(batch_num):
	x = []
	target = []
	# tmp1_list = []
	# tmp2_list = []
	for i in range(batch_num):
		act = get_random_joint_angle()
		next_state, reward, done, _ = env.step(act)
		# env.interface.GAZEBO_Step(1)
		# time.sleep(0.1)
		# input('stop')
		# continue

		tcp_pq = pq_dict_to_list(env.interface.GAZEBO_GetLinkPQByName("vs060", "J6"))
		q = [tcp_pq[3], tcp_pq[4],tcp_pq[5],tcp_pq[6]]
		euler = quat2euler(q) #W,X,Y,Z

		print(euler)
		print(np.max(euler),np.min(euler))
		
		tcp_euler = [tcp_pq[0]/1.8,tcp_pq[1]/1.8,tcp_pq[2]/1.8,
		                euler[0],euler[1],euler[2]]
		x.append(tcp_euler)
		act = [act[0]/1.57,act[1]/2.1,act[2]/2.1,act[3]/1.57,act[4]/1.57,act[5]/1.57]
		target.append(act)

	# print(np.max(x),np.min(x))
	# input("stop")
	x_tensor = torch.from_numpy(np.array(x)).float().to(device)
	target_tensor = torch.from_numpy(np.array(target)).float().to(device)
	# print("time:",time.clock()-start_t)
	return x_tensor, target_tensor


if __name__ == "__main__":
	model = IK_NN(n_feature=6, n_neuron=300, n_output=6)
	# model = load_model()
	model.cuda()
	# # model.cuda().double()
	# # print(model)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	env = GAZEBO("17004")
	print("after create gazebo")
	batch_size = 256
	epochs = 10000000
	min_loss = 20000000

	for i in range(epochs):
		# print(i)
		x,target = get_batch_tensor(batch_size)
		y = model(x)
		loss = criterion(y,target)
		if i%50==0:
			print("loss:", loss.item())

		if loss < min_loss:
			if i>1000:
				min_loss = loss
				print("============model saved============")
				print("best loss:", loss.item())
				torch.save(model.state_dict(),"./saved_models/4layers_large/IK_model"+str(loss.item()))				

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Perform action
		# next_state, reward, done, _ = env.step(action)

		# Train agent after collecting sufficient data
		# if t >= args.start_timesteps:
		# 	policy.train(replay_buffer, args.batch_size)

