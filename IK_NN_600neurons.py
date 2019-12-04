import numpy as np
import torch
import torch.nn.functional as F
import argparse, os, utils
from gazebo import *
from euler_convert import euler2quat,quat2euler
device = "cuda"

class IK_NN(torch.nn.Module):
	def __init__(self, n_feature, n_neuron, n_output):
		super(IK_NN,self).__init__()
		self.l1 = torch.nn.Linear(n_feature,n_neuron)
		self.l2 = torch.nn.Linear(n_neuron,n_neuron)
		self.l3 = torch.nn.Linear(n_neuron,n_neuron)
		self.out = torch.nn.Linear(n_neuron,n_output)
    
	def forward(self,x):
		x = F.tanh(self.l1(x))
		x = F.tanh(self.l2(x))
		x = F.tanh(self.l3(x))
		x = self.out(x)
		return x

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
		tcp_pq = pq_dict_to_list(env.interface.GAZEBO_GetLinkPQByName("vs060", "J6"))
		# print(tcp_pq)
		q = [tcp_pq[3], tcp_pq[4],tcp_pq[5],tcp_pq[6]]
		euler = quat2euler(q) #W,X,Y,Z
		tcp_euler = [tcp_pq[0]/1.8,tcp_pq[1]/1.8,tcp_pq[2]/1.8,euler[0]/np.pi,euler[1]/np.pi,euler[2]/np.pi]
		# tmp1 = [tcp_pq[0],tcp_pq[1],tcp_pq[2]]
		# tmp2 = [euler[0],euler[1],euler[2]]
		# print("euler:",tmp_euler) #yaw(z), pitch(y), roll(x)
		# tmp_qu = euler2quat(tmp_euler[0],tmp_euler[1],tmp_euler[2])  #yaw(z), pitch(y), roll(x)
		# print("qu:",tmp_qu) #W,X,Y,Z
		# print(tcp_euler)
		# input("stop")
		x.append(tcp_euler)
		# tmp1_list.append(tmp1)
		# tmp2_list.append(tmp2)
		act = [act[0]/1.57,act[1]/2.1,act[2]/2.1,act[3]/1.57,act[4]/1.57,act[5]/1.57]
		target.append(act)
		# env.interface.GAZEBO_Step(1)
		# time.sleep(.01)

	# print(np.shape(x),np.shape(target))
	# print(np.max(x),np.min(x))
	# print(np.max(tmp1_list),np.min(tmp1_list))
	# print(np.max(tmp2_list),np.min(tmp2_list))
	# print(np.max(target),np.min(target))
	# input('stop')

	x_tensor = torch.from_numpy(np.array(x)).float().to(device)
	target_tensor = torch.from_numpy(np.array(target)).float().to(device)
	# print("time:",time.clock()-start_t)
	return x_tensor, target_tensor


if __name__ == "__main__":

	model = IK_NN(n_feature=6, n_neuron=600, n_output=6)
	model.cuda()
	# # model.cuda().double()
	# # print(model)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	env = GAZEBO("15000")
	print("after create gazebo")
	batch_size = 128
	epochs = 1000000
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
				torch.save(model.state_dict(),"./saved_models/neurons600/IK_model_600neurons"+str(loss.item()))				

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Perform action
		# next_state, reward, done, _ = env.step(action)

		# Train agent after collecting sufficient data
		# if t >= args.start_timesteps:
		# 	policy.train(replay_buffer, args.batch_size)

