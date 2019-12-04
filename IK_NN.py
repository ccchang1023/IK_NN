import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse, os
from gazebo import *
from euler_convert import euler2quat,quat2euler
device = "cpu"
env = GAZEBO("18004")

pos_limit = [[-1.57,1.57],[-0.75,2.1],[-0.75,2.1],[-1.57,1.57],[-1.57,1.57],[-1.57,1.57]]

class IK_NN(torch.nn.Module):
	def __init__(self, n_feature, n_neuron, n_output):
		super(IK_NN,self).__init__()
		self.l1 = torch.nn.Linear(n_feature,n_neuron)
		self.l2 = torch.nn.Linear(n_neuron,n_neuron)
		self.l3 = torch.nn.Linear(n_neuron,n_neuron)
		self.l4 = torch.nn.Linear(n_neuron,n_neuron)
		self.out = torch.nn.Linear(n_neuron,n_output)
    
	def forward(self,x):
		x = F.tanh(self.l1(x))
		x = F.tanh(self.l2(x))
		x = F.tanh(self.l3(x))
		x = F.tanh(self.l4(x))
		x = self.out(x)
		return x


def load_model(resume_model):
	model_save_path = "/home/ccchang/TD3/saved_models/aux_model"
	model = IK_NN(n_feature=10, n_neuron=300, n_output=6)
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
		# print(tcp_pq, act)
		# input("stop")

	# print(np.max(x),np.min(x))
	# input("stop")
	x_tensor = torch.from_numpy(np.array(x)).float().to(device)   #(batch_num,10)
	target_tensor = torch.from_numpy(np.array(target)).float().to(device) #(batch_num,6)
	# print("shape:",x_tensor.size(),target_tensor.size())
	return x_tensor, target_tensor


def train():
	model = IK_NN(n_feature=10, n_neuron=300, n_output=6)
	# resume_model = "IK_model_4layers0.009619119577109814"
	# model = load_model(resume_model)
	if device =="cuda":
		model.cuda()
	# # model.cuda().double()
	# # print(model)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	batch_size = 1
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
				torch.save(model.state_dict(),"./saved_models/aux_model/IK_model_4layers"+str(loss.item()))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Perform action
		# next_state, reward, done, _ = env.step(action)

		# Train agent after collecting sufficient data
		# if t >= args.start_timesteps:
		# 	policy.train(replay_buffer, args.batch_size)


def test():
	resume_model = "IK_model_4layers0.009619119577109814"
	model = load_model(resume_model)
	model.cuda()
	model.eval()
	# env = GAZEBO("16004")
	criterion = torch.nn.MSELoss()

	first_act = get_random_joint_angle()
	next_state, reward, done, _ = env.step(first_act)
	env.interface.GAZEBO_Step(1)
	tcp_pq_list = []
	act_list = []
	act = first_act.copy()

	with torch.no_grad():
		print("collect...")
		for i in range(100):
			# act = get_random_joint_angle()
			next_state, reward, done, _ = env.step(act)
			env.interface.GAZEBO_Step(1)
			tcp_pq = pq_dict_to_list(env.interface.GAZEBO_GetLinkPQByName("vs060", "J6"))
			tcp_pq_list.append(tcp_pq)
			act_list.append(act)
			act[0] -= 0.02
			for j in range(len(pos_limit)):
				act[j] = np.clip(act[j],pos_limit[j][0],pos_limit[j][1])
			print(i)
			input()

		next_state, reward, done, _ = env.step(first_act)
		env.interface.GAZEBO_Step(1)
		print(np.shape(tcp_pq_list),np.shape(act_list))
		print("predict...")
		for i in range(len(tcp_pq_list)):
			tcp_pq = tcp_pq_list[i]
			q = [tcp_pq[3], tcp_pq[4],tcp_pq[5],tcp_pq[6]]
			euler = quat2euler(q) #W,X,Y,Z
			tcp_pq_euler = [tcp_pq[0]/1.8,tcp_pq[1]/1.8,tcp_pq[2]/1.8,
							tcp_pq[3],tcp_pq[4],tcp_pq[5],tcp_pq[6],
							euler[0]/np.pi,euler[1]/np.pi,euler[2]/np.pi]

			act = act_list[i]
			act = [act[0]/1.57,act[1]/2.1,act[2]/2.1,act[3]/1.57,act[4]/1.57,act[5]/1.57]

			for i in range(len(pos_limit)):
				act = np.clip(act,pos_limit[i][0],pos_limit[i][1])

			input_tensor = torch.from_numpy(np.array(tcp_pq_euler)).float().to(device).unsqueeze(0)
			target_tensor = torch.from_numpy(np.array(act)).float().to(device).unsqueeze(0)

			# print("input shape:",input_tensor.size())
			# print("target shape:",target_tensor.size())
			predict_act = model(input_tensor).detach()

			loss = criterion(predict_act,target_tensor)
			print("loss:",loss.item())

			# print("before multiply:",predict_act)
			scalar = torch.tensor([1.57,2.1,2.1,1.57,1.57,1.57]).to(device)
			predict_act *= scalar
			# print("after multiply:",predict_act)

			predict_act = predict_act.cpu().numpy().reshape(-1)

			print("tcp:",tcp_pq)
			print("target_joint:",act_list[i])
			print("predict_joint:",predict_act)
			next_state, reward, done, _ = env.step(predict_act)
			env.interface.GAZEBO_Step(1)
			input("stop")


def test_2():
	resume_model = "IK_model_4layers0.009619119577109814"
	model = load_model(resume_model)
	model.cuda()
	model.eval()
	# env = GAZEBO("16004")
	criterion = torch.nn.MSELoss()
	first_act = get_random_joint_angle()
	next_state, reward, done, _ = env.step(first_act)
	env.interface.GAZEBO_Step(1)
	tcp_pq_list = []
	act_list = []
	act = first_act.copy()
	time_count = 0
	with torch.no_grad():
		for i in range(10000000):
			env.interface.GAZEBO_Step(1)
			time.sleep(0.1)
			tcp_pq = pq_dict_to_list(env.interface.GAZEBO_GetLinkPQByName("vs060", "J6"))
			tcp_pq[0] = -0.3
			# tcp_pq[0] = np.random.uniform(-0.7,-0.3)
			# tcp_pq[1] = np.random.uniform(-0.5,0.6)
			tcp_pq[1] = -0.1
			tcp_pq[2] = np.random.uniform(0.6,1.2)
			rx = 0
			ry = np.pi
			rz = 0
			q = euler2quat(rz,ry,rx)
			tcp_pq[3],tcp_pq[4],tcp_pq[5],tcp_pq[6] = q[0],q[1],q[2],q[3]
			# q = [tcp_pq[3], tcp_pq[4],tcp_pq[5],tcp_pq[6]]

			# euler = quat2euler(q) #W,X,Y,Z
			
			# print(tcp_pq,euler)
			# env.reset()

			tcp_pq_euler = [tcp_pq[0]/1.8,tcp_pq[1]/1.8,tcp_pq[2]/1.8,
							tcp_pq[3],tcp_pq[4],tcp_pq[5],tcp_pq[6],
							rz/np.pi,ry/np.pi,rx/np.pi]

			# tcp_pq_euler = [tcp_pq[0]/1.8,tcp_pq[1]/1.8,tcp_pq[2]/1.8,
			# 				tcp_pq[3],tcp_pq[4],tcp_pq[5],tcp_pq[6],
			# 				euler[0]/np.pi,euler[1]/np.pi,euler[2]/np.pi]

			input_tensor = torch.from_numpy(np.array(tcp_pq_euler)).float().to(device).unsqueeze(0)
			# target_tensor = torch.from_numpy(np.array(act)).float().to(device).unsqueeze(0)

			start_t = time.clock()
			predict_act = model(input_tensor).detach()
			time_count += time.clock()-start_t
			# loss = criterion(predict_act,target_tensor)
			# print("loss:",loss.item())

			scalar = torch.tensor([1.57,2.1,2.1,1.57,1.57,1.57]).to(device)
			predict_act *= scalar

			predict_act = predict_act.cpu().numpy().reshape(-1)
			for i in range(len(pos_limit)):
				predict_act = np.clip(predict_act,pos_limit[i][0],pos_limit[i][1])

			print("tcp:",tcp_pq)
			print("predict_joint:",predict_act)
			next_state, reward, done, _ = env.step(predict_act)
			env.interface.GAZEBO_Step(1)
			input("stop")

		print("Time:",time_count/100)		#0.0022171399999999686

if __name__ == "__main__":
	train()
	# test_2()
	# for i in range(10000000):
	# 	env.interface.GAZEBO_Step(1)
	# 	time.sleep(0.1)	

