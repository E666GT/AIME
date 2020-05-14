"""
This Py give the PPO solution to the Robot task in GYM.
"""
import gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(3,15)
        self.fc2=nn.Linear(15,15)
        self.fc3=nn.Linear(15,1)

        # code_size=10
        # self.Extractor_s = Extractor(3, code_size)
        # self.Extractor_r = Extractor(1,code_size)
        # self.Extractor_a = Extractor(1,code_size)
        # self.Extractor_steps=Extractor(1,code_size)
        # self.Extractor_vhat = Extractor(1,code_size)
        # self.Extractor_rhat = Extractor(1,code_size)

    def forward(self,x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        # x = torch.clamp(x,-2,2)
        return x

# class Extractor(nn.Module):
#     def __init__(self,input_size,output_size):
#         super(Extractor, self).__init__()
#         self.l=[15,64,32]
#         self.net=nn.Sequential(
#             nn.Linear(input_size,self.l[0]),
#             nn.Tanh(),
#             nn.Linear(self.l[0],self.l[1]),
#             nn.Tanh(),
#             nn.Linear(self.l[1],self.l[2]),
#             nn.Tanh(),
#             nn.Linear(self.l[2],self.l[output_size]),
#         )

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss,self).__init__()
    def forward(self,output,reward):
        # target=(output+(reward+6)).detach()
        # loss=target-output
        loss=criterion(output,reward)
        return loss


if __name__=="__main__":

    """Env"""
    # env = gym.make('FetchPickAndPlace-v1')
    env = gym.make('Pendulum-v0')
    print(env.action_space.high)
    print(env.action_space.low)
    # [2.]
    # [-2.]
    # print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    # [1. 1. 8.]
    # [-1. -1. -8.]


    """Net"""
    net=Net().to(device)
    myloss=MyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    """Start"""
    env.reset()
    action=env.action_space.sample()
    losses=[]
    tic=0
    for _ in range(100000):
        env.render()
        observation, reward, done, info = env.step(action)


        inputs=torch.tensor(observation).float().to(device)
        output=net(inputs)

        loss=criterion(output,torch.tensor(reward).float().to(device))

        net.zero_grad()
        loss.backward()
        optimizer.step()

        # action=output.detach().numpy()
        # action = [2 if input("")=="1" else -2]
        action=[random.random()*random.randint(-2,2)]
        if(len(losses)<5):
            losses.append(loss)
        else:
            print(np.array(losses,dtype=np.float).mean())
            losses=[]
            print("time,",time.time()-tic)
            tic=time.time()
        # print("loss:",loss)
        # print("reward:",reward)
        # print("action:",action,"\n")
        pass
        # time.sleep(0.3)

    env.close()