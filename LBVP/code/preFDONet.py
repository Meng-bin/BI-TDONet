from keras import Model
import tensorflow as tf
import numpy as np


class model(Model):
    def __init__(self, Layers):
        super(model, self).__init__()

        self.Layers_branch1 = Layers[0]
        self.Layers_branch2 = Layers[1]
        self.Layers_inner = Layers[2]
        # Initialize NN
        (
            self.Weights_branch1,
            self.Biases_branch1,
            self.Weights_branch2,
            self.Biases_branch2,
            self.Weight_out,
            self.Biases_out,
        ) = self.initialize_NN(Layers)

    def initialize_NN(self, Layers):
        Layers1 = Layers[0]
        Layers2 = Layers[1]
        Layers3 = Layers[2]
        weights_branch1 = []
        biases_branch1 = []
        weights_branch2 = []
        biases_branch2 = []
        weights_out = []
        biases_out = []
        for l in range(0, len(Layers1) - 1):
            W = self.xavier_init(size=[Layers1[l], Layers1[l + 1]])
            b = tf.Variable(
                tf.zeros([1, Layers1[l + 1]], dtype=tf.float32), dtype=tf.float32
            )
            weights_branch1.append(W)
            biases_branch1.append(b)
        for l in range(0, len(Layers2) - 1):
            W = self.xavier_init(size=[Layers2[l], Layers2[l + 1]])
            b = tf.Variable(
                tf.zeros([1, Layers2[l + 1]], dtype=tf.float32), dtype=tf.float32
            )
            weights_branch2.append(W)
            biases_branch2.append(b)

        for l in range(0, len(Layers3) - 1):
            W = self.xavier_init(size=[Layers3[l], Layers3[l + 1]])
            b = tf.Variable(
                tf.zeros([1, Layers3[l + 1]], dtype=tf.float32), dtype=tf.float32
            )
            weights_out.append(W)
            biases_out.append(b)

        return (
            weights_branch1,
            biases_branch1,
            weights_branch2,
            biases_branch2,
            weights_out,
            biases_out,
        )

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
            dtype=tf.float32,
        )

    def l2_normalize(self, X):
        epsion = 1e-8
        return X / tf.reshape(tf.linalg.norm(X, axis=1) + epsion, [-1, 1])

    # @tf.function(jit_compile=True)
    def oper_net(self, X):
        # X=self.l2_normalize(X)
        weights_branch1 = self.Weights_branch1
        biases_branch1 = self.Biases_branch1
        weights_branch2 = self.Weights_branch2
        biases_branch2 = self.Biases_branch2
        weights_out = self.Weight_out
        biases_out = self.Biases_out
        num_Layers1 = len(weights_branch1)
        num_Layers2 = len(weights_branch2)
        num_Layers3 = len(weights_out)

        mid = []
        # branch1=gamma
        branch1 = X[:, : self.Layers_branch2[0]]
        branch2 = X[:, self.Layers_branch2[0] :]
        mid.append(X)
        for l in range(0, num_Layers1 - 1):
            W = weights_branch1[l]
            b = biases_branch1[l]
            mid.append(tf.nn.relu(tf.add(tf.matmul(mid[l], W), b)))
        W = weights_branch1[-1]
        b = biases_branch1[-1]
        Y_branch1 = tf.add(tf.matmul(mid[-1], W), b)

        mid = []
        mid.append(branch1)
        for l in range(0, num_Layers2 - 1):
            W = weights_branch2[l]
            b = biases_branch2[l]
            mid.append(tf.nn.relu(tf.add(tf.matmul(mid[l], W), b)))
        W = weights_branch2[-1]
        b = biases_branch2[-1]
        Y_branch2 = tf.add(tf.matmul(mid[-1], W), b)

        out = Y_branch1 * (Y_branch2)
        out = tf.concat([branch1, out], axis=1)
        mid = []
        mid.append(out)
        for l in range(0, num_Layers3 - 1):
            W = weights_out[l]
            b = biases_out[l]
            mid.append(tf.nn.relu(tf.add(tf.matmul(mid[l], W), b)))
        W = weights_out[-1]
        b = biases_out[-1]
        out = tf.add(tf.matmul(mid[-1], W), b)
        # +branch2
        return out

    def to_point(self, p, M):
        N = (p.shape[1] - 1) // 2
        t = np.linspace(0, 2 * np.pi, M)
        sin_part = []
        cos_part = []
        for i in range(1, N + 1):
            sin_part.append(tf.sin(i * t))
            cos_part.append(tf.cos(i * t))
        sin_part = tf.reshape(sin_part, [N, -1])
        cos_part = tf.reshape(cos_part, [N, -1])
        p_cos = tf.reshape(p[:, 1 : N + 1], [-1, N])
        p_sin = tf.reshape(p[:, N + 1 : 2 * N + 2], [-1, N])
        phi = (
            tf.matmul(p_sin, tf.cast(sin_part, tf.float32))
            + tf.matmul(p_cos, tf.cast(cos_part, tf.float32))
            + tf.reshape(p[:, 0], [-1, 1])
        )
        return phi

    def my_loss(self, y_true, y_pred):
        M = 128
        return tf.reduce_sum(
            tf.norm(self.to_point(y_pred, M) - self.to_point(y_true, M), axis=1)
            / tf.norm(self.to_point(y_true, M), axis=1)
        )

    def rse_coefficient_mean(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.norm(y_pred - y_true, axis=1) / tf.norm(y_true, axis=1)
        )

    def rse_coefficient_sum(self, y_true, y_pred):
        return tf.reduce_sum(tf.norm(y_pred - y_true, axis=1) / tf.norm(y_true, axis=1))

    def call(self, X):
        # X=tf.nn.l2_normalize(X,dim=1)
        # branch2 = tf.nn.l2_normalize(X[:, self.Layers_branch2[0] :],axis=1)
        output = self.oper_net(X)
        # + branch2

        return output

    def out(self, X):
        # X=tf.nn.l2_normalize(X,dim=1)
        # branch2 = tf.nn.l2_normalize(X[:, self.Layers_branch2[0] :],axis=1)
        out = self.oper_net(X)
        # + branch2

        return out.numpy()
