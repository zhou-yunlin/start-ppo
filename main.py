import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt


EP_MAX = 1000
EP_LEN = 200
BATCH = 32
GAMMA = 0.9
C_LR = 0.0002
A_LR = 0.0001
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # 剪掉代理目标，发现这样更好
][1]


class PPO:
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self._build_anet('Critic')
        with tf.compat.v1.variable_scope('closs'):
            self.tfdc_r = tf.compat.v1.placeholder(tf.float32, [None, 1], name='discounted_r')
            self.adv = self.tfdc_r - self.v
            closs = tf.reduce_mean(tf.square(self.adv))
            self.ctrain = tf.train.AdamOptimizer(C_LR).minimize(closs)

        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.compat.v1.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.compat.v1.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        with tf.compat.v1.variable_scope('aloss'):
            self.tfa = tf.compat.v1.placeholder(tf.float32, [None, A_DIM], 'action')
            self.tfadv = tf.compat.v1.placeholder(tf.float32, [None, 1], 'advantage')
            with tf.compat.v1.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
                if METHOD['name'] == 'kl_pen':
                    self.tflam = tf.compat.v1.placeholder(tf.float32, None, 'lambda')
                    kl = tf.compat.v1.distributions.kl_divergence(oldpi, pi)
                    self.kl_mean = tf.reduce_mean(kl)
                    self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
                else:  # clipping method, find this is better
                    self.aloss = -tf.reduce_mean(tf.minimum(
                        surr,
                        tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * self.tfadv))
            self.atrain = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        tf.summary.FileWriter('log/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _build_anet(self, name, trainable=True):
        if name == 'Critic':
            with tf.variable_scope(name):
                # self.s_Critic = tf.placeholder(tf.float32, [None, S_DIM], 'state')
                l1_Critic = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable, name='l1')
                self.v = tf.layers.dense(l1_Critic, 1, trainable=trainable, name='value_predict')

        else:
            with tf.variable_scope(name):
                # self.s_Actor = tf.placeholder(tf.float32, [None, S_DIM], 'state')
                l1_Actor = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable, name='l1')
                mu = 2 * tf.layers.dense(l1_Actor, A_DIM, tf.nn.tanh, trainable=trainable, name='mu')
                sigma = tf.layers.dense(l1_Actor, A_DIM, tf.nn.softplus, trainable=trainable, name='sigma')
                norm_list = tf.distributions.Normal(loc=mu, scale=sigma)
                params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
            return norm_list, params

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.adv, {self.tfdc_r: r, self.tfs: s})
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run([self.atrain, self.kl_mean], {self.tfa: a, self.tfadv: adv, self.tfs: s, self.tflam: METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:
            [self.sess.run(self.atrain, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        [self.sess.run(self.ctrain, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})


env = gym.make('Pendulum-v1').unwrapped
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
ppo = PPO()
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)
        s = s_
        ep_r += r

        if (t+1) % BATCH == 0 or t == EP_LEN - 1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA*v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(discounted_r)
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)

    if ep == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print('Ep:%d | Ep_r:%f' % (ep, ep_r))

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()

