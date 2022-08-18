# encoding=utf-8
#------------import package-----------
import numpy as np
from collections import deque
from github.DDPG import DDPG

#------------import systems-----------
from github.Examples import compare_DDPG_all

ddpg = True

def train():

    # ----------------parameters definition-----------------
    l = 0
    safe_track = 0 #the number of safe trajectories
    mxlen = 2000
    var = 3

    #-----------------starting training---------------------

    #~~~~~~~~~~~~~~construct training example~~~~~~~~~~~~~~~
    env = compare_DDPG_all(1)
    agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path='ddpg_rl' + env.path, units=env.units,
                   dense=env.dense,
                   activation=env.activation)
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
    env = compare_DDPG_all(1)
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

    unsafe_time, s_unsafe,agent,safe_time=test()
    if unsafe_time != 0:
        re()


