# encoding=utf-8
from github.DDPG import DDPG
import random
import numpy as np
import time
from collections import deque



from github.Examples import get_parameters_12_dim
from github.Examples import compare_all

ddpg = True


#------Meta-Learning parameters setting---------------
support_num = 2
query_num = 2
epochs = 2

trajectory_num = 50
step_num = 2000


outputText = './parameter_test/para_uni_12.txt'
#----step1----training initialization parameters by Meta-Learning-------------
def get_parameter():
    # ----------Initialize the meta-network-----------------------------------------
    env_meta = get_parameters_12_dim(random.randint(0, 49))
    # Define a meta network model of DDPG type according to the selected system
    model_meta = DDPG(1, a_bound=env_meta.u, s_dim=env_meta.n_obs, is_train=True, path=env_meta.path,
                      units=env_meta.units, dense=env_meta.dense, activation=env_meta.activation)


    tot = 0
    safe_time = 0
    mxlen = 2000
    dq = deque(maxlen=mxlen)
    var = 3


    for t in range(epochs):
       #-----------Start meta-training------------------------

        for i in range(support_num):
            # randomly select a training task from the dataset
            env = get_parameters_12_dim(random.randint(0, 49))
            agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path=env.path,
                             units=env.units, dense=env.dense, activation=env.activation)

            # ------Assign the initial meta-network model parameters to each training task----
            out1, out2, out3, out4 = model_meta.get_apra()
            agent.assign(out1, out2, out3, out4)



            for k in range(trajectory_num):

                s = env.reset()
                reward = 0
                for p in range(step_num):
                    dq.append(sum(env.s ** 2))
                    a = agent.choose_action(s)
                    a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                    s_, r, done = env.step(a_v)
                    agent.store_transition(s, a_v, r, s_, done)
                    reward += r
                    if done:
                        print('unsafe')
                        safe_time = 0
                        tot = tot // 2000 * 2000 + 1010
                    if tot % 2000 == 0:
                        done = 1
                    tot += 1
                    if tot > 1000:
                        if tot % 10 == 0:
                            var *= .9995

                        agent.learn()

                    if done or (len(dq) == mxlen and np.var(dq) < 1e-10):
                        print('reward:', reward, 'if_unsafe:', done,' Explore:', var)
                        break
                    s = s_.copy()


        #-----------Start meta-updating------------------------

        for i in range(query_num):
            env = get_parameters_12_dim(random.randint(0, 49))
            reward = 0

            start = time.time()
            for k in range(trajectory_num):

                s = env.reset()
                for p in range(step_num):
                    a = agent.choose_action(s)
                    a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                    s_, r, done = env.step(a_v)
                    model_meta.store_transition(s, a_v, r, s_, done)
                    reward += r
                    if done:
                        print('unsafe')
                        safe_time = 0
                        tot = tot // 2000 * 2000 + 1010
                    if tot % 2000 == 0:
                        done = 1
                    tot += 1
                    if tot > 1000:
                        if tot % 10 == 0:
                            var *= .9995
                        model_meta.learn()

                    if done or (len(dq) == mxlen and np.var(dq) < 1e-10):
                        print('reward:', reward, 'if_unsafe:', done,' Explore:', var)
                        break
                    s = s_.copy()
                safe_time+= 1
                model_meta.save_model()

    return model_meta

def train():

    # ----------------parameters definition-----------------
    l = 0
    safe_track = 0 #the number of safe trajectories
    mxlen = 2000
    var = 3

    #-----------------starting training---------------------

    #~~~~~~~~~~~~~~construct training example~~~~~~~~~~~~~~~
    env = compare_all(1)
    agent = agent_0
    # ~~~~~~~Randomly select initial points for training~~~~~~
    num=50 # define the number of initial points
    points = []
    for _ in range(num):
        s = env.reset()
        points.append(s)
    for t in range(len(points)):
        s = points[t]
        reward = 0 # from RL
        dq = deque(maxlen=mxlen)
        while True:
            dq.append(sum(env.s ** 2))
            a = agent.choose_action(s)
            a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[
                0]  # add randomness to action selection for exploration
            s_, r, done = env.step(a_v)
            agent.store_transition(s, a_v, r, s_, done)
            reward += r
            if done:
                print('unsafe')
                safe_time = 0
                l = l // 2000 * 2000 + 1010
            if l % 2000 == 0:
                done = 1
            l += 1
            # ~~~~The track of the store does not start learning until a certain amount is reached~~~~
            if l > 1000:
                if l % 10 == 0:
                    var *= .9995
                # ~~~~starting learning~~~~
                agent.learn()

            if done or (len(dq) == mxlen and np.var(dq) < 1e-10) :
                print('reward:', reward, 'if_unsafe:', done, ' Explore:', var)
                break
            s = s_.copy()
        safe_track += 1
    agent.save_model() #save model

    return agent, safe_track


def test():
    #-----------------starting testing----------------
    env = compare_all(1)
    agent, safe_time = train()
    # ---------------parameters definition-----------
    l = 0
    var = 3
    mxlen = 50
    epoch = 1
    # ~~~~~~~Randomly select initial points for training~~~~~~
    num_p=50
    s_sum = []
    for _ in range(num_p):
        s = env.reset()
        s_sum.append(s)
    s_unsafe = [] #unsafe trajectories
    unsafe_time = 0 #the number of unsafe trajectories
    for i in range(len(s_sum)):

        s = s_sum[i]
        reward = 0
        dq = deque(maxlen=mxlen)
        while True:
            dq.append(sum(env.s ** 2))
            a = agent.choose_action(s)
            a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
            s_, r, done = env.step(a_v)
            reward += r
            if done:
                print('unsafe')
                print("=======test failed=======")
                unsafe_time += 1
                s_unsafe.append(s)

            if l % 2000 == 0:
                done = 1
            l += 1

            if done or (len(dq) == mxlen and np.var(dq) < 1e-10):
                # if done or (len(dq) == mxlen and np.var(dq) < 1e-10) :

                print('reward:', reward, 'if_unsafe:', done, ' Explore:', var)
                dq.clear()
                break
            s = s_.copy()


    return unsafe_time, s_unsafe,agent,safe_time


def re():
    #--------------retrain and test if there are unsafe trajectories-------------
    unsafe_time, s_unsafe, agent,safe_time = test()
    #-----------------start retraining and testing----------------
    epoch = 1

    while unsafe_time != 0:

        epoch += 1
        if epoch == 50:
            print('Maximum epoch has been reached, failed to train controller')
            break



if __name__ == '__main__':
    agent_0=get_parameter()
    unsafe_time, s_unsafe,agent,safe_time=test()
    if unsafe_time != 0:
        re()