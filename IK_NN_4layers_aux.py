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
		self.b1 = torch.nn.BatchNorm1d(n_neuron)

		self.l2 = torch.nn.Linear(n_neuron,n_neuron)
		self.b2 = torch.nn.BatchNorm1d(n_neuron)
		self.d1 = torch.nn.Dropout(0.2)

		self.l3 = torch.nn.Linear(n_neuron,n_neuron)
		self.b3 = torch.nn.BatchNorm1d(n_neuron)

		self.l4 = torch.nn.Linear(n_neuron,n_neuron)
		self.b4 = torch.nn.BatchNorm1d(n_neuron)
		self.d2 = torch.nn.Dropout(0.2)

		self.out = torch.nn.Linear(n_neuron,n_output)
    
	def forward(self,x):
		x = self.b1(torch.tanh(self.l1(x)))
		x = self.b2(torch.tanh(self.l2(x)))
		x = self.d1(x)

		x = self.b3(torch.tanh(self.l3(x)))
		x = self.b4(torch.tanh(self.l4(x)))
		x = self.d2(x)
		x = self.out(x)

		return x


def load_model():
	model_save_path = "/home/ccchang/TD3/saved_models/aux_model"
	resume_model = "IK_model_4layers0.009619119577109814"
	model = IK_NN(n_feature=10, n_neuron=300, n_output=6)
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
		tcp_pq_euler = [tcp_pq[0]/1.8,tcp_pq[1]/1.8,tcp_pq[2]/1.8,
		                tcp_pq[3],tcp_pq[4],tcp_pq[5],tcp_pq[6],
						euler[0]/np.pi,euler[1]/np.pi,euler[2]/np.pi]
		x.append(tcp_pq_euler)
		act = [act[0]/1.57,act[1]/2.1,act[2]/2.1,act[3]/1.57,act[4]/1.57,act[5]/1.57]
		target.append(act)

	# print(np.max(x),np.min(x))
	# input("stop")
	x_tensor = torch.from_numpy(np.array(x)).float().to(device)
	target_tensor = torch.from_numpy(np.array(target)).float().to(device)
	# print("time:",time.clock()-start_t)
	return x_tensor, target_tensor


def get_learning_rate(optimizer):
    for param in optimizer.param_groups:
        return param['lr']

def decay_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.2 * lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("=======================decay learning rate to ",param_group['lr'], "=======================")
		

if __name__ == "__main__":
	model = IK_NN(n_feature=10, n_neuron=300, n_output=6)
	# model = load_model()
	model.cuda()
	# # model.cuda().double()
	# # print(model)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
	lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15,factor=0.1)
	env = GAZEBO("18006")
	print("after create gazebo")
	batch_size = 256
	epochs = 10000000
	min_loss = 20000000
	loss_avg = 0
	val_frequency = 10

	for i in range(1,epochs+1):
		# print(i)
		x,target = get_batch_tensor(batch_size)
		y = model(x)
		loss = criterion(y,target)
		loss_avg += loss.item()
		
		if i%val_frequency==0:
			loss_avg /= val_frequency
			print("Epoch:{},loss:{}".format(i,loss_avg))
			lr_scheduler.step(loss_avg)
			if loss_avg < min_loss:
				if i>5000:
					min_loss = loss_avg
					# print("============model saved============")
					print("best loss:", loss_avg)
					# torch.save(model.state_dict(),"./saved_models/aux_model/IK_model_4layers"+str(loss.item()))							
			loss_avg = 0

		# if i!=0 and i % 20000 == 0:
		# 	lr = get_learning_rate(optimizer)
		# 	decay_learning_rate(optimizer,lr)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

