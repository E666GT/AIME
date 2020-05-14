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
import yaml
# device=torch.device("cpu")
global args
import time
import os
from tempfile import TemporaryFile
from torch.distributions import MultivariateNormal
torch.autograd.set_detect_anomaly(True)
with open("args.yaml") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cuda")


class History:
    def __init__(self):
        # self.actions = []

        self.logprobs = []
        self.is_terminals = []
        self.steps=[]

        self.actions_hat=[]
        self.actions_hat_pre=[]

        self.states = []
        self.states_pre=[]

        self.rewards = []
        self.rewards_pre=[]

        self.v_hat=[]
        self.v_hat_pre=[]

        self.r_hat=[]
        self.r_hat_pre=[]



        self.all={

            "logprobs": self.logprobs,
            "is_terminals": self.is_terminals,
            "steps": self.steps,

            "actions_hat":self.actions_hat,
            "actions_hat_pre": self.actions_hat_pre,

            "states":self.states,
            "states_pre": self.states_pre,

            "v_hat": self.v_hat,
            "v_hat_pre":self.v_hat_pre,

            "r_hat": self.r_hat,
            "r_hat_pre":self.r_hat_pre,

            "rewards":self.rewards,
            "rewards_pre": self.rewards_pre,


        }

    def clear_memory(self):
        for _ in self.all.keys():
            del self.all[_][:]
        # del self.states[:]
        # del self.logprobs[:]
        # del self.rewards[:]
        # del self.is_terminals[:]
        # del self.steps[:]
        # del self.v_hat[:]
        # del self.r_hat[:]
        # del self.actions_hat[:]
        # del self.v_hat_pre[:]
        # del self.r_hat_pre[:]
        # del self.actions_hat_pre[:]
        # del self.states_pre[:]
        # del self.rewards_pre[:]

    def trans_to_torch(self):
        # self.all["steps"]=torch.tensor(self.all["steps"])

        torch_all={}
        shapes={}
        for k in self.all.keys():
            # # print(k)
            if(k=="is_terminals"):
                continue
            his=self.all[k]

            if(type(his[0])==np.ndarray):
                stack_res=torch.Tensor(his).to(device)
            else:
                try:
                    stack_res=torch.stack(his).to(device)
                except Exception as err:
                    pass
            if(stack_res.dim()==0):
                stack_res=stack_res.unsqueeze(0)


            stack_res=stack_res.squeeze(1)

            torch_all[k]=stack_res.detach()


            shape_=torch_all[k].shape
            shapes[k]=shape_
            # if(stack_res.dim()>=2):
            #     try:
            #         torch_all[k]=torch.squeeze(stack_res, 1).detach()
            #     except:
            #         pass
            # except:
            #     torch_all[k] = torch.squeeze(torch.stack(his), 1).to(device).detach()
        return torch_all

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.interact_with_env=1

        action_dim=args["action_dim"]
        action_std=args['action_std']
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)
        # self.action_var=args[action_var]

        memory_dim=args["memory_size"]# Hyper

        self.Global_Memory=torch.zeros(memory_dim)
        self.Temp_Memory=torch.zeros(memory_dim)

        code_size=10
        self.Extractor_s = Extractor(3, code_size).to(device)
        self.Extractor_r = Extractor(1,code_size).to(device)
        self.Extractor_a = Extractor(1,code_size).to(device)
        self.Extractor_steps=Extractor(1,code_size).to(device)
        self.Extractor_vhat = Extractor(1,code_size).to(device)
        self.Extractor_rhat = Extractor(1,code_size).to(device)

        codes_pre=[torch.ones([10]) for i in range(6)]
        self.Fuser_A_pre = Fuser(codes_pre,args["Fuser_A_pre_out_dim"]).to(device) # Hyper
        codes_after = [torch.ones([10]) for i in range(7)]
        self.Fuser_A_after = Fuser(codes_after,memory_dim).to(device)  # Hyper


        dummy_mem=[torch.ones(memory_dim) for i in range(2)]
        self.Fuser_Alpha = MLP_alpha(dummy_mem,[1]).to(device)


        dummy_mem=[torch.ones(memory_dim) for i in range(2)]
        self.FuserDecoder_A= Fuser(dummy_mem,memory_dim).to(device)
        dummy_mem_codes=[torch.ones(memory_dim) for i in range(3)]
        self.FuserDecoder_Action = Fuser(dummy_mem_codes,[args["action_dim"]]).to(device)
        self.FuserDecoder_Value = Fuser(dummy_mem_codes,[1]).to(device)
        self.FuserDecoder_Reward = Fuser(dummy_mem_codes,[1]).to(device)

    def forward(self,st_pre,rt_pre,at_pre,stepst,vt_hat_pre,rt_hat_pre,action_after):

        code_s=self.Extractor_s(st_pre)
        code_r=self.Extractor_r(rt_pre)
        code_a=self.Extractor_a(at_pre)
        code_steps=self.Extractor_steps(stepst)
        code_v_1_hat=self.Extractor_vhat(vt_hat_pre)
        code_rt_hat=self.Extractor_rhat(rt_hat_pre)

        sons_A_pre=[code_s,code_r,code_a,code_steps,code_v_1_hat,code_rt_hat]
        # try:
        Code_A_pre=self.Fuser_A_pre(sons_A_pre)
        # except:
        #     pass
        sons_A_after=sons_A_pre
        sons_A_after.append(Code_A_pre)

        # Temp_Mem=self.Fuser_A_after(sons_A_after)
        # Temp_Mem = Temp_Mem.mean(dim=0).unsqueeze(0)
        # Global_Mem=torch.unsqueeze(self.Global_Memory.to(device),0)

        self.Temp_Memory=self.Fuser_A_after(sons_A_after)
        # self.Temp_Memory = self.Temp_Memory.mean(dim=0).unsqueeze(0)

        self.Global_Memory=torch.unsqueeze(self.Global_Memory.to(device),0)
        self.Global_Memory=self.Global_Memory.repeat(self.Temp_Memory.shape[0],1,1)
        # alpha=self.Fuser_Alpha([self.Global_Memory,self.Temp_Memory])
        # alpha=0.1
        code_decode_A=self.FuserDecoder_A([self.Global_Memory,self.Temp_Memory])
        sons_final_decode=[self.Global_Memory,self.Temp_Memory,code_decode_A]
        rt_plus_1_hat=self.FuserDecoder_Reward(sons_final_decode)
        vt_hat=self.FuserDecoder_Value(sons_final_decode)

        # cal action
        action_mean=self.FuserDecoder_Action(sons_final_decode)
        action_var = self.action_var.expand_as(action_mean)
        action_cov=torch.diag_embed(action_var)
        # rch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, action_cov)

        if (self.interact_with_env):
            action = dist.sample()# for interact with env
            action = action.clamp(-2, 2).detach()
        else:
            action = action_after # train mode
        action_logprob = dist.log_prob(action).unsqueeze(0).t()
        if(action_logprob.dim()==1):
            action_logprob=action_logprob.unsqueeze(0)
        dist_entropy = dist.entropy().t()
        if (dist_entropy.dim() == 1):
            dist_entropy = dist_entropy.unsqueeze(0).t()
        # state_value = vt_hat
        self.Global_Memory=self.Global_Memory.mean(dim=0).squeeze(0)
        return rt_plus_1_hat,vt_hat,action,action_logprob,dist_entropy

    def enable_interact_mode(self,bl):
        self.interact_with_env=bl
class GRL:
    def __init__(self):
        global args
        self.gamma=args['gamma']
        self.eps_clip=args["eps_clip"]
        self.args=args
        self.net=Net().to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args["lr"], betas=eval(args["betas"]))
        self.MseLoss=nn.MSELoss()
        self.net_old=Net().to(device)
        self.net_old.load_state_dict(self.net.state_dict())

    def save(self,name):
        torch.save(self.net_old,os.path.join("trained",name))
        # outfile = TemporaryFile(os.path.join("trained",name+"_memory"))
        np.save(os.path.join("trained",name+"_mem.npy"),self.net.Global_Memory.cpu().numpy())
        # outfile.close()
        print("Saved!",name)
    def load(self,name):
        self.net_old=torch.load(os.path.join("trained",name))
        self.net_new=torch.load(os.path.join("trained",name))
        self.net.Global_Memory=torch.Tensor(np.load(os.path.join("trained",name+"_mem.npy"))).to(device)
        # pass

    @staticmethod
    def deal_input(observation,reward,action,_,vt_hat,rt_hat):
        try:
            st = torch.tensor(observation).float().to(device).unsqueeze(0)
        except Exception as err:
            print(err)
            pass
        rt = torch.tensor([reward]).float().to(device).unsqueeze(0)
        at = action.float().to(device)
        stepst = torch.tensor([_]).float().to(device).unsqueeze(0)
        vt_hat = vt_hat.float().to(device)
        rt_hat = rt_hat.float().to(device)
        # vals=[st,rt,at,stepst,vt_hat,rt_hat].unsqueeze(0)
        # for i in range(len(vals)):
        #     while(vals[i].dim()<=1):
        #         vals[i].unsqueeze(0)
        shapes=[st.shape,rt.shape,at.shape,stepst.shape,vt_hat.shape,rt.shape]
        dims = [st.dim(), rt.dim(), at.dim(), stepst.dim(), vt_hat.dim(), rt.dim()]
        return st,rt,at,stepst,vt_hat,rt_hat

    # def select_action(self, st, rt, at, stepst, vt_hat, rt_hat):
    #     rt_plus_1_hat, vt_hat, action, action_logprob, dist_entropy = grl.net(st, rt, at, stepst, vt_hat, rt_hat)
    #     action = action.clamp(-2, 2).detach()

        # state = torch.FloatTensor(state.reshape(1, -1)).to(device).double()
        # return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    def update(self,history):

        discounted_rewards=[]
        discounted_reward=0
        for reward, is_terminal in zip(reversed(history.rewards), reversed(history.is_terminals)):
            if is_terminal:
                discounted_reward=0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
            pass
        # normalizing the discounted rowards:
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = ((discounted_rewards-discounted_rewards.mean())/(discounted_rewards.std()+1e-5)).unsqueeze(1).to(device)

        #convert to tensor
        hs=history.trans_to_torch()
        total_loss=[0,0,0,0]
        K=self.args["K_iters_train"]
        for _ in range(K):
            # Evaluating old actions and values :
            # logprobs_new, state_values_new, dist_entropy_new = self.net(old_states.double(), old_actions.double())

            # st,rt,at,stepst,vt_hat,rt_hat=grl.deal_input(observation_pre,reward_pre,at_pre,_,vt_hat_pre,r_hat_pre)

            grl.net.enable_interact_mode(False)
            rt_hat, vt_hat, action_hat, action_logprob, dist_entropy\
                =grl.net(hs["states_pre"],hs["rewards_pre"],hs["actions_hat_pre"],
                         hs["steps"],hs["v_hat_pre"],hs["r_hat_pre"],action_after=hs["actions_hat"])


            ratios = torch.exp(action_logprob - hs["logprobs"].detach())

            advantages=discounted_rewards.to(device)-vt_hat
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            Loss_action=-torch.min(surr1, surr2).mean()
            Loss_V=self.args["cf_loss_V"] * self.MseLoss(vt_hat, discounted_rewards)
            Loss_r=self.args["cf_loss_r"] * self.MseLoss(rt_hat, hs["rewards"])
            Loss_dist=- 0.01*dist_entropy.mean()

            total_loss=[total_loss[0]+Loss_action,total_loss[1]+Loss_V,total_loss[2]+Loss_r,total_loss[3]+Loss_dist]

            loss = Loss_action+Loss_V+Loss_r+Loss_dist
            # loss = Loss_V
            # loss = Loss_r

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            # update Global Mem
            self.net.Global_Memory = (torch.squeeze(self.net.Global_Memory) + torch.squeeze(self.net.Temp_Memory.mean(0) * args["mem_alpha"])).detach()
            pass
        print("        LossA,LossV,LossR,LossD   ",total_loss[0].item()/K,"  ",total_loss[1].item()/K,"  ",total_loss[2].item()/K,"  ",total_loss[3].item()/K)
        print("        AVG Award  ",hs["rewards"].mean())
        self.net_old.load_state_dict(self.net.state_dict())

class Extractor(nn.Module):
    def __init__(self,input_size,output_size):
        super(Extractor, self).__init__()
        self.l=args["Extractor_L"]
        self.net=nn.Sequential(
            nn.Linear(input_size,self.l[0]),
            nn.Tanh(),
            nn.Linear(self.l[0],self.l[1]),
            nn.Tanh(),
            nn.Linear(self.l[1],self.l[2]),
            nn.Tanh(),
            nn.Linear(self.l[2],output_size),
        )

    def forward(self,x):
        try:
            return self.net(x)
        except:
            pass
class Fuser(nn.Module):
    def __init__(self,sons,out_dim):
        super(Fuser,self).__init__()
        global args
        self.args=args
        self.f=torch.cat(sons).flatten()
        input_size=self.f.size()[0]
        output_size=1
        for sz in out_dim:
            output_size*=sz
        self.out_dim=out_dim

        self.l=args["Fuser_L"]
        self.net=nn.Sequential(
            nn.Linear(input_size,self.l[0]),
            nn.Tanh(),
            nn.Linear(self.l[0],self.l[1]),
            nn.Tanh(),
            nn.Linear(self.l[1],self.l[2]),
            nn.Tanh(),
            nn.Linear(self.l[2],output_size),
        )

    def forward(self,sons):
        try:
            sons_flat=[torch.flatten(sons[i],1,-1) for i in range(len(sons))]
        except Exception as err:
            pass

        cat_sons_flat = torch.cat(sons_flat,1)
        out=self.net(cat_sons_flat)

        out_dim=self.out_dim[:]
        out_dim.insert(0,-1)
        out=out.view(out_dim)
        # f=cat_sons.flatten()
        # try:
        #     out=self.net(f).reshape(self.out_dim)
        # except:
        #     pass
        return out
class MLP_alpha(Fuser):
    def __init__(self,sons,out_dim):
        super(MLP_alpha,self).__init__(sons,out_dim)
        limitation=args["temp_alpha_clamp"]
    def forward(self,sons):
        # f = torch.cat(sons).flatten()
        # f = self.net(f).reshape(self.out_dim)
        # alpha = f.clamp(-self.args["temp_alpha_clamp"],self.args["temp_alpha_clamp"])


        sons_flat=[torch.flatten(sons[i],1,-1) for i in range(len(sons))]

        cat_sons_flat = torch.cat(sons_flat,1)

        out=self.net(cat_sons_flat)

        out_dim=self.out_dim[:]
        out_dim.insert(0,-1)
        out=out.view(out_dim)
        out=out.clamp(-self.args["temp_alpha_clamp"],self.args["temp_alpha_clamp"])

        return out
# class MyLoss_step(nn.Module):
#     def __init__(self):
#         super(MyLoss_step,self).__init__()
#     def forward(self,output,reward):
#         # target=(output+(reward+6)).detach()
#         # loss=target-output
#         loss=criterion(output,reward)
#         return loss
# class MyLoss_longstep(nn.Module):
#     def __init__(self):
#         super(MyLoss_longstep,self).__init__()
#     def forward(self,output,reward):
#         # target=(output+(reward+6)).detach()
#         # loss=target-output
#         loss=criterion(output,reward)
#         return loss


if __name__=="__main__":

    """Env"""
    # env = gym.make('FetchPickAndPlace-v1')
    env = gym.make(args["env_name"])
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
    grl=GRL()
    # net=Net().to(device)
    # myloss_step=MyLoss_step()
    # myloss_episode=MyLoss_episode()
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # criterion = nn.MSELoss()
    history=History()
    if(args["load"]):
        try:
            grl.load(args["save_and_load_path"])
            print("Successful load{}",format(args["save_and_load_path"]))
        except Exception as err:
            print(err,"\n","Load Error, Init the net")
    """Start"""
    total_steps=0
    history.clear_memory()
    tic = time.time()
    running_reward=0
    for epi in range(args["max_episodes"]):
    # only 1 epi

        env.reset()
        action=env.action_space.sample()
        losses=[]
        observation, reward, done, info = env.step(action)
        vt_hat_pre = torch.Tensor([[0]])
        r_hat_pre = torch.Tensor([[0]])
        at_pre = torch.Tensor([action])
        reward_pre = reward
        states_pre = observation


        for _ in range(args["steps_per_episode"]):
            total_steps+=1
            # if(total_steps%args["steps_per_episode"]==0):

            if(args["render"]):
                env.render()

            # input to net

            states_pre,reward_pre,at_pre,stepst,vt_hat_pre,r_hat_pre=grl.deal_input(states_pre,
                                                                                    reward_pre,
                                                                                    at_pre,
                                                                                    _,
                                                                                    vt_hat_pre,
                                                                                    r_hat_pre)
            grl.net.enable_interact_mode(True)
            rt_hat, vt_hat, action, action_logprob, dist_entropy=grl.net_old(states_pre,reward_pre,at_pre,stepst,vt_hat_pre,r_hat_pre,action_after=None)

            # rt_hat=torch.Tensor([[1]])
            # vt_hat=torch.Tensor([[1]])
            # action=torch.Tensor([[2]])
            # action_logprob=torch.Tensor([[1]])
            # dist_entropy=torch.Tensor([[1]])



            observation, reward, done, info = env.step(action.cpu())
            reward=(reward+8.1)/8.1
            running_reward += reward
            if(_==(args["steps_per_episode"]-1)):
                done=1

            # Add history,to be learned later
            history.states.append(observation)
            history.actions_hat.append(action)
            history.logprobs.append(action_logprob)
            history.steps.append(stepst)
            history.rewards.append(torch.Tensor([[reward]]).to(device))
            history.is_terminals.append(done)
            history.v_hat.append(vt_hat)
            history.r_hat.append(rt_hat)
            history.v_hat_pre.append(vt_hat_pre)
            history.r_hat_pre.append(r_hat_pre)
            history.actions_hat_pre.append(at_pre)
            history.states_pre.append(states_pre)
            history.rewards_pre.append(reward_pre)

            vt_hat_pre=vt_hat
            r_hat_pre=rt_hat
            at_pre=action
            states_pre=observation
            reward_pre=reward

            # off-policy update
            if total_steps % args["update_per_steps"]==0 and total_steps>1:
                # print("    Steps:",total_steps)
                print("    Updating...,Total_steps,",total_steps)
                grl.update(history)
                history.clear_memory()
                grl.save(args["save_and_load_path"])
                toc = time.time()
                print("    steps:", total_steps, "time,", toc - tic , " Runing Reward:",running_reward)
                running_reward=0
                tic = toc
            if(done):
                # print("Done"," timestamp=",total_steps)
                break
                """deprecated"""
                # observation, reward, done, info = env.step(action)
                # new_history=np.array([st,rt,at,stepst,vt_1_hat,rt_hat]).reshape(6,1)
                # History=np.concatenate((History,new_history),0)
                # rt_plus_1_hat,vt_hat,action=net(st,rt,at,stepst,vt_1_hat,rt_hat)
                # loss_step=myloss_step(rt_plus_1_hat,reward)
                # net.zero_grad()
                # loss_step.backward()
                # optimizer.step()
                # if(len(losses)<5):
                #     losses.append(loss_step)
                # else:
                #     print(np.array(losses,dtype=np.float).mean())
                #     losses=[]
                #     print("time,",time.time()-tic)
                #     tic=time.time()
        # if(epi%50==0):


        # # Learn History
        # Return=[]
        # gamma=args["gamma"]
        # grl.net.enable_interact_mode(False)
        # # Return=np.power(gamma,)
        # for t in range(History.shape(0)):
        #     return_=np.power(gamma,History.shape(0)-t)*
        #     Return.append(History[t:,1])
        # for t in range(History.shape(0)):
        #     if t==0:
        #         continue
        #
        #
        #     old_history=History[t-1,:]
        #     new_history=History[t,:]
        #     for k in range(args["K_iters_train"]):
        #         st = torch.tensor(old_history[0]).float().to(device)
        #         rt = torch.tensor(old_history[1]).float().to(device)
        #         at = torch.tensor(old_history[2]).float().to(device)
        #         stepst = torch.tensor(old_history[3]).float().to(device)
        #         vt_1_hat = torch.tensor(old_history[4]).float().to(device)
        #         rt_hat = torch.tensor(old_history[5]).float().to(device)
        #
        #         rt_plus_1_hat, vt_hat, action = net(st, rt, at, stepst, vt_1_hat, rt_hat)
        #         action = action.clamp(-2, 2)
        #
        #         reward=torch.tensor(new_history[1]).float().to(device)
        #         loss_step = myloss_step(rt_plus_1_hat, reward)
        #
        #
        #
        #
        #         net.zero_grad()
        #         loss_step.backward()
        #         optimizer.step()
        #


        # print("loss:",loss)
        # print("reward:",reward)
        # print("action:",action,"\n")
        # time.sleep(0.3)

    env.close()
    # Timenow=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # grl.save(Timenow+"_"+str(total_steps))

