# encoding=utf-8
import numpy as np
from sympy import *
import math
import random


pi = math.pi




class Zones():
    def __init__(self, shape, center=None, r=0.0, low=None, up=None):
        self.shape = shape
        if shape == 'ball':
            self.low = np.array(low)
            self.up = np.array(up)

            self.center = np.array(center)
            self.r = r
        elif shape == 'box':
            self.low = np.array(low)
            self.up = np.array(up)
            self.center = (self.low + self.up) / 2
            self.r = sum(((self.up - self.low) / 2) ** 2)
        else:
            raise ValueError('没有形状为{}的区域'.format(shape))

    def __str__(self):
        return 'Zones:{' + 'shape:{}, center:{}, r:{}, low:{}, up:{}'.format(self.shape, self.center, self.r, self.low,
                                                                             self.up) + '}'


class Example():
    def __init__(self, n_obs, D_zones, I_zones, U_zones, f, u, degree, path, dense, units, activation, id, k, old_id=-1,B=None):
        self.n_obs = n_obs  # number of variables
        self.D_zones = D_zones  # invariant region
        self.I_zones = I_zones  # initial region
        self.U_zones = U_zones  # unsafe region
        self.f = f  # differential equations
        self.B = B  # barrier function
        self.u = u  # output range is [-u,u]
        self.degree = degree  # degree of fitting this polynomial
        self.path = path  # save path
        self.dense = dense  # the number of layers in the network
        self.units = units  # the number of nodes per layer
        self.activation = activation  # activation function
        self.k = k  #
        self.id = id  #
        self.old_id=old_id


class Env():
    def __init__(self, example):
        self.n_obs = example.n_obs
        self.D_zones = example.D_zones
        self.I_zones = example.I_zones
        self.U_zones = example.U_zones
        self.f = example.f
        self.B = example.B
        self.path = example.path
        self.u = example.u
        self.degree = example.degree
        self.dense = example.dense
        self.units = example.units
        self.activation = example.activation
        self.old_id = example.old_id
        self.dt = 0.01
        self.k = example.k
        self.is_lidao = False if self.B == None else True


    # def unisample(self, s):
    #     self.s = s

    # def reset(self):
    #     self.s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])  ##边长为1，中心在原点的正方体的内部，产生-0.5~0.5的随机数，组成
    #     if self.I_zones.shape == 'ball':
    #         ## 在超球内进行采样：将正方体进行归一化，变成对单位球的表面采样，再对其半径进行采样。
    #         self.s *= 2  ## 变成半径为1
    #         self.s = self.s / np.sqrt(sum(self.s ** 2)) * self.I_zones.r * np.random.random() ** (
    #                 1 / self.n_obs)  ##此时球心在原点
    #         ## random()^(1/d) 是为了均匀采样d维球体
    #         self.s += self.I_zones.center
    #
    #     else:
    #         self.s = self.s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
    #     return self.s
    def reset(self):
        self.s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])  ##边长为1，中心在原点的正方体的内部，产生-0.5~0.5的随机数，组成
        if self.I_zones.shape == 'ball':
            ## 在超球内进行采样：将正方体进行归一化，变成对单位球的表面采样，再对其半径进行采样。
            self.s *= 2  ## 变成半径为1
            self.s = self.s / np.sqrt(sum(self.s ** 2)) * self.I_zones.r * np.random.random() ** (
                    1 / self.n_obs)  ##此时球心在原点
            ## random()^(1/d) 是为了均匀采样d维球体
            self.s += self.I_zones.center

        else:
            self.s = self.s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
            # self.s = random.uniform(self.I_zones.up , self.I_zones.low)

        return self.s


    def step(self, u):
        self.ds = np.array([F(self.s, u) for F in self.f])
        self.s = self.s + self.ds * self.dt

        if self.D_zones.shape == 'box':
            self.s = np.array(
                [min(max(self.s[i], self.D_zones.low[i]), self.D_zones.up[i]) for i in range(self.n_obs)]
            )

        else:
            t = np.sqrt(self.D_zones.r / sum(self.s ** 2))
            if t < 1:
                self.s = self.s * t

        if self.U_zones.shape == 'ball':
            is_unsafe = sum((self.s - self.U_zones.center) ** 2) < self.U_zones.r
        else:
            safe = 0
            for i in range(self.n_obs):
                if self.U_zones.low[i] <= self.s[i] <= self.U_zones.up[i]:
                    safe = safe + 1
            is_unsafe = (safe == self.n_obs)

        dis = sum((self.s - self.U_zones.center) ** 2) - self.U_zones.r
        reward = dis / 4



        return self.s, reward, is_unsafe

    def get_sign(self, u):
        sb = ['x' + str(i) for i in range(self.n_obs)]
        x = symbols(sb)  # 求导用
        B = self.B(x)  # 障碍函数
        x_0 = {k: v for k, v in zip(sb, self.s)}
        if B.subs(x_0) >= 0.1:
            return True
        su = sum([diff(B, x[i]).subs(x_0) * self.f[i](self.s, u) for i in range(self.n_obs)])
        return su > 0


def get_parameters_12_dim(i):

    examples = {
        0: Example(  # \cite{chen2020novel}
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
            U_zones=Zones('box', low=[0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
                          up=[0.6, 0.6, 0.6, 1.6, 1.6, 1.6, 1.6, -0.5, 1.5, 1.5, -0.1, 2.0]),
            f=[lambda x, u: x[3],
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: -7253.4927 * x[0] + 1936.3639 * x[10] - 1338.7624 * x[3] + 1333.3333 * x[7],
               lambda x, u: -1936.3639 * x[9] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
               lambda x, u: -769.2308 * x[2] - 770.2301 * x[5],
               lambda x, u: x[9],
               lambda x, u: x[10],
               lambda x, u: x[11],
               lambda x, u: 9.81 * x[1],
               lambda x, u: -9.81 * x[0],
               lambda x, u: -16.3541 * x[11] + u
               ],
            B=None,
            u=3,
            degree=3,
            path='dim12test_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=16,
            k=100
        ),
        1: Example(  # \cite{setta20}
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[1] * 12, up=[2] * 12),
            U_zones=Zones('box', low=[-1] * 12, up=[-0.5] * 12),
            f=[lambda x, u: x[0] * x[2],
               lambda x, u: x[0] * x[4],
               lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
               lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
               lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
               lambda x, u: 2 * x[1] * x[4] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=2,
            path='dim12_test_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=17,
            k=100,
        ),
    }
    a = random.random()
    b = random.random() * 0.1
    I_zones_list = [Zones('box', low=[random.uniform(-0.5+0.0001,-0.1)] * 12, up=[random.uniform(0.1,0.5-0.0001)] * 12),
                    Zones('box', low=[1] * 12, up=[2] * 12)]
    U_zones_list = [Zones('box',low=[0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
                          up=[0.6, 0.6, 0.6, 1.6, 1.6, 1.6, 1.6, -0.5, 1.5, 1.5, -0.1, 2.0]),
                    Zones('box', low=[random.uniform(-2, -1)] * 12, up=[random.uniform(-0.5, 1-0.0001)] * 12)]
    f_list = [[lambda x, u: x[3],
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: -7253.4927 * x[0] + 1936.3639 * x[10] - 1338.7624 * x[3] + 1333.3333 * x[7],
               lambda x, u: -1936.3639 * x[9] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
               lambda x, u: -769.2308 * x[2] - 770.2301 * x[5],
               lambda x, u: x[9],
               lambda x, u: x[10],
               lambda x, u: x[11],
               lambda x, u: 9.81 * x[1],
               lambda x, u: -9.81 * x[0],
               lambda x, u: -16.3541 * x[11] + u
               ],
              [lambda x, u: x[0] * x[2],
               lambda x, u: x[0] * x[4],
               lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
               lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
               lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
               lambda x, u: 2 * x[1] * x[4] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
              ]
    path_list = ['dim12test_0/model', 'dim12_test_1/model']
    for key in range(2, 50):
        feed = random.randint(0, 1)
        examples.update({
            key: Example(
                n_obs=12,
                D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
                I_zones=I_zones_list[feed],
                U_zones=U_zones_list[feed],
                f=f_list[feed],
                B=None,
                u=3,
                degree=2,
                path=path_list[feed],
                dense=5,
                units=30,
                activation='relu',
                id=17,
                k=100,
            )

        }
        )

    return Env(examples[i])

def compare_all(i):

    examples = {

        0: Example(  #\cite{prajna2005optimization}
            n_obs=12,
            D_zones=Zones('box', low=[-2]*12, up=[2]*12),
            I_zones=Zones('box', low=[1.14,-0.74,1.14,-0.74,1.14,-0.74,1.14,-0.74,1.14,-0.74,1.14,-0.74], up=[1.51,0.64,1.51,0.64,1.51,0.64,1.51,0.64,1.51,0.64,1.51,0.64]),
            U_zones=Zones('box', low=[-2]*12, up=[-1.9]*12),

            f=[lambda x, u: x[1],
               lambda x, u: -x[0] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],

            B=None,
            u=1,
            degree=1,
            path='Exp0/model',
            dense=5,
            units=30,
            activation='relu',
            id=0,
            old_id=0,
            k=50
        ),
        1: Example(  # ----\cite{aylward2008stability}
            n_obs=12,
            D_zones=Zones('ball', center=[0]*12, r=4),
            I_zones=Zones('box', low=[0.68,-1.42,0.68,-1.42,0.68,-1.42,0.68,-1.42,0.68,-1.42,0.68,-1.42], up=[1.72,1.42,1.72,1.42,1.72,1.42,1.72,1.42,1.72,1.42,1.72,1.42]),
            U_zones=Zones('box', low=[-2.18,-0.22,-2.18,-0.22,-2.18,-0.22,-2.18,-0.22,-2.18,-0.22,-2.18,-0.22], up=[-1.32,0.22,-1.32,0.22,-1.32,0.22,-1.32,0.22,-1.32,0.22,-1.32,0.22]),
            f=[lambda x, u: u - 0.5 * x[0] ** 3,
               lambda x, u: 3 * x[0] - x[1],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=1,
            degree=3,
            path='Exp1/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            old_id=1,
            k=50,
        ),
        2: Example(  # ---\cite{sassi2014iterative}
            n_obs=12,
            D_zones=Zones('box', low=[-3]*12, up=[3]*12),
            I_zones=Zones('box', low=[-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1], up=[-0.9, 1.1,-0.9, 1.1,-0.9, 1.1,-0.9, 1.1,-0.9, 1.1,-0.9, 1.1]),
            U_zones=Zones('ball', center=[-2.25, -1.75,-2.25, -1.75,-2.25, -1.75,-2.25, -1.75,-2.25, -1.75,-2.25, -1.75], r=0.25),

            f=[lambda x, u: -0.1 / 3 * x[0] ** 3 + 7 / 8 + u,
               lambda x, u: 0.8 * (x[0] - 0.8 * x[1] + 0.7),
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=0.3,
            degree=3,
            path='Exp2/model',
            dense=5,
            units=30,
            activation='relu',
            id=2,
            old_id=2,
            k=50
        ),
        3: Example(  # --------------\cite{bouissou2014computation}
            n_obs=12,
            D_zones=Zones('box', low=[-2]*12, up=[2]*12),
            I_zones=Zones('box', low=[-1.5]*12, up=[-1.4,-1.3,-1.4,-1.3,-1.4,-1.3,-1.4,-1.3,-1.4,-1.3,-1.4,-1.3]),
            U_zones=Zones('box', low=[-0.1,0.5,-0.1,0.5,-0.1,0.5,-0.1,0.5,-0.1,0.5,-0.1,0.5], up=[0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0,0.1,1.0]),
            f=[lambda x, u: -x[0] + x[0] * x[1],
               lambda x, u: u - x[0] + 0.25 * x[1],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=1,
            degree=2,
            path='Exp3/model',
            dense=5,
            units=30,
            activation='relu',
            id=3,
            old_id=3,
            k=50,
        ),
        4: Example( #\cite{prajna2004nonlinear}
            n_obs=12,
            D_zones=Zones('box', low=[-4]*12, up=[4]*12),
            I_zones=Zones('ball', center=[1, 0,1, 0,1, 0,1, 0,1, 0,1, 0], r=0.2816223536781477),
            U_zones=Zones('ball', center=[-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1], r=0.5045710224795428),
            f=[lambda x, u: -6 * x[0] * x[1] ** 2 - x[0] ** 2 * x[1] + 2 * x[1] ** 3,
               lambda x, u: x[1] * u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=1,
            degree=3,
            path='Exp4/model',
            dense=5,
            units=30,
            activation='relu',
            id=4,
            old_id=4,
            k=50,
        ),

        5: Example(  #\cite{jarvis2003lyapunov}
            n_obs=12,
            D_zones=Zones('ball', center=[0]*12, r=25),
            I_zones=Zones('ball', center=[0]*12, r=4.090297354693472),
            U_zones=Zones('ball', center=[10]*12, r=5),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2]),
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0],

            B=None,
            u=0.4,
            degree=3,
            path='dim4_0_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=5,
            old_id=5,
            k=50,
        ),

        6: Example(  # \cite{Chesi04}
            n_obs=12,
            D_zones=Zones('ball', center=[0]*12, r=16),

            I_zones=Zones('box', low=[-1]*12, up=[1]*12),

            U_zones=Zones('ball', center=[-2]*12, r=1),
            f=[lambda x, u: -x[0] - x[3] + u,
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u,
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0],
            B=None,
            u=1,
            degree=3,
            path='dim4_1_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=6,
            old_id=6,
            k=100,
        ),
        7: Example(  # \cite{jin2020neural}
            n_obs=12,


            D_zones=Zones('box', low=[-1.3] * 12, up=[1.3] * 12),

            I_zones=Zones('box', low=[-0.89] * 12, up=[0.89] * 12),

            U_zones=Zones('box', low=[0.9] * 12, up=[1.3] * 12),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: 1 + sin(x[1]) * (x[1] * x[1] - cos(x[1])),
               lambda x, u: u * cos(x[1]) + x[1] * x[1] * cos(x[1]) * sin(x[1]) - 2 * sin(x[1]) / (
                       (1 + sin(x[1])) ** 2),
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0,
            lambda x, u: 0],
            B=None,
            u=0.2,
            degree=3,
            path='Cartpole/model',
            dense=5,
            units=30,
            activation='relu',
            id=7,
            old_id=7,
            k=50
        ),

        8: Example(  #\cite{bouissou2014computation}
            n_obs=12,
            D_zones=Zones('box', low=[0]*12, up=[10]*12),
            I_zones=Zones('box', low=[2.13446723]*12, up=[3.96553277]*12),
            U_zones=Zones('box', low=[4, 4.1, 4.2, 4.3, 4.4, 4.5,4.6, 4.7, 4.8, 4.9, 5.0, 5.1], up=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6,4.7, 4.8, 4.9, 5.0, 5.1,5.2]),
            f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_0/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=8,
            old_id=8,
            k=100,
        ),
        9: Example(  # \cite{setta20}
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[1] * 12, up=[2] * 12),

            U_zones=Zones('box', low=[-1.69152329] * 12, up=[0.19152329] * 12),

            f=[lambda x, u: x[0] * x[2],
               lambda x, u: x[0] * x[4],
               lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
               lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
               lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
               lambda x, u: 2 * x[1] * x[4] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=9,
            old_id=9,
            k=100,
        ),
        10: Example(  # \cite{djaballah2017construction}
            n_obs=12,
            D_zones=Zones('box', low=[0, 0, 2, 0, 0, 0,0, 0, 2, 0, 0, 0], up=[10] * 12),
            I_zones=Zones('ball', center=[0, 3.05, 3.05, 3.05, 3.05, 3.05,3.05, 3.05, 3.05, 3.05, 3.05,3.05], r=0.01),
            U_zones=Zones('ball', center=[0, 7.05, 7.05, 7.05, 7.05, 7.05,7.05, 7.05, 7.05, 7.05, 7.05, 7.05,], r=0.01),
            f=[lambda x, u: -x[0] + 4 * x[1] - 6 * x[2] * x[3] + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[3] * x[5],
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_2/model',
            dense=5,
            units=30,
            activation='relu',
            id=10,
            old_id=10,
            k=100,
        ),
        11: Example(  # \cite{huang2017probabilistic}
            n_obs=12,
            D_zones=Zones('box', low=[-4.5] * 12, up=[4.5] * 12),
            I_zones=Zones('box', low=[-4, -0.5, -2.4, -2.5, 0, -4,-4, -0.5, -2.4, -2.5, 0, -4], up=[4, 4.5, 2.4, 3.5, 2, 0,4, 4.5, 2.4, 3.5, 2, 0]),
            U_zones=Zones('ball', center=[0, 3, 3, 3, 3, 3,3, 3, 3, 3, 3, 3], r=0.5),
            f=[lambda x, u: x[3] + u,
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: (-2 / 3) * x[0] + (1 / 3) * x[0] * x[2],
               lambda x, u: (1 / 3) * x[0] + (2 / 3) * x[1] + (1 / 3) * x[1] * x[2],
               lambda x, u: -4 + x[2] ** 2 + 8 * x[4],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_3/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            old_id=11,
            k=100,
        ),

        12: Example( #\cite{chen2020novelchen2020novel}
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[0.69530137] * 12, up=[1.30469863] * 12),
            U_zones=Zones('box', low=[1.8] * 12, up=[2] * 12),
            f=[lambda x, u: 3 * x[2] + u,
               lambda x, u: x[3] - x[1] * x[5],
               lambda x, u: x[0] * x[5] - 3 * x[2],
               lambda x, u: x[1] * x[5] - x[3],
               lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
               lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
               lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
               lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
               lambda x, u: 2 * x[5] * x[7] - x[8],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=2,
            path='dim9_0_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=12,
            old_id=12,
            k=100  # 3000
        ),

        13: Example(  # \cite{chen2020novel}
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
            U_zones=Zones('box', low=[0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
                          up=[0.6, 0.6, 0.6, 1.6, 1.6, 1.6, 1.6, -0.5, 1.5, 1.5, -0.1, 2.0]),
            f=[lambda x, u: x[3],
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: -7253.4927 * x[0] + 1936.3639 * x[10] - 1338.7624 * x[3] + 1333.3333 * x[7],
               lambda x, u: -1936.3639 * x[9] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
               lambda x, u: -769.2308 * x[2] - 770.2301 * x[5],
               lambda x, u: x[9],
               lambda x, u: x[10],
               lambda x, u: x[11],
               lambda x, u: 9.81 * x[1],
               lambda x, u: -9.81 * x[0],
               lambda x, u: -16.3541 * x[11] + u
               ],
            B=None,
            u=3,
            degree=3,
            path='dim12test_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=13,
            old_id=13,
            k=100
        )


    }


    return Env(examples[i])

def compare_DDPG_all(i):

    examples = {

        0: Example(  #\cite{prajna2005optimization}
            n_obs=2,
            D_zones=Zones('box', low=[-2]*2, up=[2]*2),
            I_zones=Zones('box', low=[1.14,-0.74], up=[1.51,0.64]),
            U_zones=Zones('box', low=[-2]*2, up=[-1.9]*2),

            f=[lambda x, u: x[1],
               lambda x, u: -x[0] + u,
               ],

            B=None,
            u=1,
            degree=1,
            path='Exp0/model',
            dense=5,
            units=30,
            activation='relu',
            id=0,
            old_id=0,
            k=50
        ),
        1: Example(  #\cite{aylward2008stability}
            n_obs=2,
            D_zones=Zones('ball', center=[0]*2, r=4),
            I_zones=Zones('box', low=[0.67957322,-1.41919684], up=[1.72042678,1.41919684]),
            U_zones=Zones('box', low=[-2.17836838,-0.22017129], up=[-1.32163162,0.22017129]),
            f=[lambda x, u: u - 0.5 * x[0] ** 3,
               lambda x, u: 3 * x[0] - x[1],

               ],
            B=None,
            u=1,
            degree=3,
            path='Exp1/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            old_id=1,
            k=50,
        ),
        2: Example(  # \cite{sassi2014iterative}
            n_obs=2,
            D_zones=Zones('box', low=[-3]*2, up=[3]*2),
            I_zones=Zones('box', low=[-1, 1], up=[-0.9, 1.1]),
            U_zones=Zones('ball', center=[-2.25, -1.75], r=0.25),

            f=[lambda x, u: -0.1 / 3 * x[0] ** 3 + 7 / 8 + u,
               lambda x, u: 0.8 * (x[0] - 0.8 * x[1] + 0.7),
               ],
            B=None,
            u=0.3,
            degree=3,
            path='Exp2/model',
            dense=5,
            units=30,
            activation='relu',
            id=2,
            old_id=2,
            k=50
        ),
        3: Example(  # \cite{bouissou2014computation}
            n_obs=2,
            D_zones=Zones('box', low=[-2]*2, up=[2]*2),
            I_zones=Zones('box', low=[-1.5]*2, up=[-1.4,-1.3]),
            U_zones=Zones('box', low=[-0.1,0.5], up=[0.1,1.0]),
            f=[lambda x, u: -x[0] + x[0] * x[1],
               lambda x, u: u - x[0] + 0.25 * x[1],

               ],
            B=None,
            u=1,
            degree=2,
            path='Exp3/model',
            dense=5,
            units=30,
            activation='relu',
            id=3,
            old_id=3,
            k=50,
        ),
        4: Example(  #\cite{prajna2004nonlinear}
            n_obs=2,
            D_zones=Zones('box', low=[-4]*2, up=[4]*2),
            I_zones=Zones('ball', center=[1, 0], r=0.2816223536781477),
            U_zones=Zones('ball', center=[-1, 1,], r=0.5045710224795428),
            f=[lambda x, u: -6 * x[0] * x[1] ** 2 - x[0] ** 2 * x[1] + 2 * x[1] ** 3,
               lambda x, u: x[1] * u,
               ],
            B=None,
            u=1,
            degree=3,
            path='Exp4/model',
            dense=5,
            units=30,
            activation='relu',
            id=4,
            old_id=4,
            k=50,
        ),

        5: Example(  #\cite{jarvis2003lyapunov}
            n_obs=4,
            D_zones=Zones('ball', center=[0]*4, r=25),
            I_zones=Zones('ball', center=[0]*4, r=4.090297354693472),
            U_zones=Zones('ball', center=[10]*4, r=5),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2]),
                ],

            B=None,
            u=0.4,
            degree=3,
            path='dim4_0_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=5,
            old_id=5,
            k=50,
        ),

        6: Example(  # \cite{Chesi04}
            n_obs=4,
            D_zones=Zones('ball', center=[0]*4, r=16),

            I_zones=Zones('box', low=[-1]*4, up=[1]*4),

            U_zones=Zones('ball', center=[-2]*4, r=1),
            f=[lambda x, u: -x[0] - x[3] + u,
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u,
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3,
           ],
            B=None,
            u=1,
            degree=3,
            path='dim4_1_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=6,
            old_id=6,
            k=100,
        ),
        7: Example(  # \cite{jin2020neural}
            n_obs=4,


            D_zones=Zones('box', low=[-1.3] * 4, up=[1.3] * 4),

            I_zones=Zones('box', low=[-0.89] * 4, up=[0.89] * 4),

            U_zones=Zones('box', low=[0.9] * 4, up=[1.3] * 4),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: 1 + sin(x[1]) * (x[1] * x[1] - cos(x[1])),
               lambda x, u: u * cos(x[1]) + x[1] * x[1] * cos(x[1]) * sin(x[1]) - 2 * sin(x[1]) / (
                       (1 + sin(x[1])) ** 2),
        ],
            B=None,
            u=0.2,
            degree=3,
            path='Cartpole/model',
            dense=5,
            units=30,
            activation='relu',
            id=7,
            old_id=7,
            k=50
        ),

        8: Example(  #\cite{bouissou2014computation}
            n_obs=6,
            D_zones=Zones('box', low=[0]*6, up=[10]*6),
            I_zones=Zones('box', low=[2.13446723]*6, up=[3.96553277]*6),
            U_zones=Zones('box', low=[4, 4.1, 4.2, 4.3, 4.4, 4.5], up=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6]),
            f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5],

               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_0/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=8,
            old_id=8,
            k=100,
        ),
        9: Example(  # \cite{setta20}
            n_obs=6,
            D_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
            I_zones=Zones('box', low=[1] * 6, up=[2] * 6),

            U_zones=Zones('box', low=[-1.69152329] * 6, up=[0.19152329] * 6),

            f=[lambda x, u: x[0] * x[2],
               lambda x, u: x[0] * x[4],
               lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
               lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
               lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
               lambda x, u: 2 * x[1] * x[4] + u,

               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=9,
            old_id=9,
            k=100,
        ),
        10: Example(  # \cite{djaballah2017construction}
            n_obs=6,
            D_zones=Zones('box', low=[0, 0, 2, 0, 0, 0,], up=[10] * 6),
            I_zones=Zones('ball', center=[3.05, 3.05, 3.05, 3.05, 3.05, 3.05], r=0.01),
            U_zones=Zones('ball', center=[7.05, 7.05, 7.05, 7.05, 7.05, 7.05], r=0.01),
            f=[lambda x, u: -x[0] + 4 * x[1] - 6 * x[2] * x[3] + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[3] * x[5],
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5],

               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_2/model',
            dense=5,
            units=30,
            activation='relu',
            id=10,
            old_id=10,
            k=100,
        ),
        11: Example(  # \cite{huang2017probabilistic}
            n_obs=6,
            D_zones=Zones('box', low=[-4.5] * 6, up=[4.5] * 6),
            I_zones=Zones('box', low=[-4, -0.5, -2.4, -2.5, 0, -4], up=[4, 4.5, 2.4, 3.5, 2, 0]),
            U_zones=Zones('ball', center=[3, 3, 3, 3, 3, 3], r=0.5),
            f=[lambda x, u: x[3] + u,
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: (-2 / 3) * x[0] + (1 / 3) * x[0] * x[2],
               lambda x, u: (1 / 3) * x[0] + (2 / 3) * x[1] + (1 / 3) * x[1] * x[2],
               lambda x, u: -4 + x[2] ** 2 + 8 * x[4],

               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_3/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            old_id=11,
            k=100,
        ),

        12: Example(  # \cite{chen2020novelchen2020novel}
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.69530137] * 9, up=[1.30469863] * 9),
            U_zones=Zones('box', low=[1.8] * 9, up=[2] * 9),
            f=[lambda x, u: 3 * x[2] + u,
               lambda x, u: x[3] - x[1] * x[5],
               lambda x, u: x[0] * x[5] - 3 * x[2],
               lambda x, u: x[1] * x[5] - x[3],
               lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
               lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
               lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
               lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
               lambda x, u: 2 * x[5] * x[7] - x[8],

               ],
            B=None,
            u=3,
            degree=2,
            path='dim9_0_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=12,
            old_id=12,
            k=100  # 3000
        ),

        13: Example(  # \cite{chen2020novel}
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
            U_zones=Zones('box', low=[0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
                          up=[0.6, 0.6, 0.6, 1.6, 1.6, 1.6, 1.6, -0.5, 1.5, 1.5, -0.1, 2.0]),
            f=[lambda x, u: x[3],
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: -7253.4927 * x[0] + 1936.3639 * x[10] - 1338.7624 * x[3] + 1333.3333 * x[7],
               lambda x, u: -1936.3639 * x[9] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
               lambda x, u: -769.2308 * x[2] - 770.2301 * x[5],
               lambda x, u: x[9],
               lambda x, u: x[10],
               lambda x, u: x[11],
               lambda x, u: 9.81 * x[1],
               lambda x, u: -9.81 * x[0],
               lambda x, u: -16.3541 * x[11] + u
               ],
            B=None,
            u=3,
            degree=3,
            path='dim12test_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=13,
            old_id=13,
            k=100
        )


    }


    return Env(examples[i])



