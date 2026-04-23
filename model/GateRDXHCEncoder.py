import tensorflow as tf
import numpy as np


class GateCrossEncoder(tf.keras.layers.Layer):

  def __init__(self, num_layers, d_model, num_heads, dff, n=2, rate=0.5):
    super(GateCrossEncoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers    
    self.n = n
    self.pos_encoding = positional_encoding(1000, self.d_model)      
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, n, rate, name=f"encoder_layer_{i}") for i in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, x2, training=True, mask=None): 
    seq_len = tf.shape(x)[1]
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)
    
    H=tf.stack([x]*self.n, axis=2)  # (B, L, n, d)

    seq_len = tf.shape(x2)[1]
    x2 += self.pos_encoding[:, :seq_len, :]
    x2 = self.dropout(x2, training=training)
    
    for layer in self.enc_layers:
      H = layer(H, x2, training, mask)

    # return tf.reduce_mean(H, axis=2)
    return tf.reduce_sum(H, axis=2)

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],np.arange(d_model)[np.newaxis, :],d_model)
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
  return tf.cast(pos_encoding, dtype=tf.float32)


class HyperConnection(tf.keras.layers.Layer):
    def __init__(self, n, d_model, name=None):
        super().__init__(name=name)
        self.n = n
        self.d = d_model

        # Am: width-connection (input aggregation)
        init_am=np.zeros((n, 1))
        init_am[0, 0] = 1.0
        self.Am = self.add_weight(
            name=f"{self.name}_Am",
            shape=(n, 1),
            initializer=tf.constant_initializer(init_am),
            trainable=True
        )

        # Ar: hidden-to-hidden mixing
        self.Ar = self.add_weight(
            name=f"{self.name}_Ar",
            shape=(n, n),
            initializer=tf.keras.initializers.Identity(),
            trainable=True
        )

        # B: output distribution
        self.B = self.add_weight(
            name=f"{self.name}_B",
            shape=(1, n),
            initializer="ones",
            trainable=True
        )

    def call(self, H, T):
        """
        H: (B, L, n, d)
        T: function (B, L, d) -> (B, L, d)
        """

        # h0 = Am^T H
        Am=tf.reshape(self.Am, (1, 1,self.n, 1))
        h0 = tf.reduce_sum(Am * H, axis=2)

        # Apply attention or FFN
        y = T(h0)  # (B, L, d)

        # B^T y
        B=tf.reshape(self.B, (1, 1,self.n, 1))
        y_exp=tf.expand_dims(y, axis=2)
        Hy = B*y_exp

        # Ar^T H
        Hr = tf.einsum("ij, bljd -> blid", self.Ar, H)

        return Hy + Hr
    

class DynamicHyperConnection(tf.keras.layers.Layer):
    def __init__(self, n, d_model, name=None):
        super().__init__(name=name)
        self.n = n
        self.d = d_model

        # Am: (n, 1)
        init_am = np.zeros((n, 1))
        init_am[0, 0] = 1.0
        self.Am_static = self.add_weight(
            name=f"{self.name}_Am_static",
            shape=(n, 1),
            initializer=tf.constant_initializer(init_am),
            trainable=True, 
        )

        # Ar: (n, n)
        self.Ar_static = self.add_weight(
            name=f"{self.name}_Ar_static",
            shape=(n, n),
            initializer=tf.keras.initializers.Identity(),
            trainable=True, 
        )

        # B: (1, n)
        self.B_static = self.add_weight(
            name=f"{self.name}_B_static",
            shape=(1, n),
            initializer="ones",
            trainable=True, 
        )

        self.alpha = self.add_weight(
          name=f"{self.name}_alpha",
          shape=(),
          initializer=tf.constant_initializer(0.02),
          trainable=True
        )

        self.dyn_Am = tf.keras.layers.Dense(
            n, kernel_initializer='zeros', name=f"{self.name}_dyn_Am"
        )
        self.dyn_Ar = tf.keras.layers.Dense(
            n * n, kernel_initializer='zeros', name=f"{self.name}_dyn_Ar"
        )
        self.dyn_B = tf.keras.layers.Dense(
            n, kernel_initializer='zeros', name=f"{self.name}_dyn_B"
        )
       
    # Token级别上下文平均   
    def call(self, H, T):
        """
        H: (B, L, n, d)
        T: 子层函数 (Attention 或 FFN)
        """
        B_dim = tf.shape(H)[0]
        L_dim = tf.shape(H)[1]

        # ========= ① Token级上下文=========
        # 只在路径维度 n 取平均，保留序列维度 L
        # H_avg: (B, L, d)
        H_avg = tf.reduce_mean(H, axis=2)

        # ========= ② 动态增量 =========
        # d_Am: (B, L, n, 1)
        d_Am = tf.expand_dims(tf.nn.tanh(self.dyn_Am(H_avg)), axis=-1) * self.alpha

        # d_Ar: (B, L, n, n)
        d_Ar = tf.reshape(
            tf.nn.tanh(self.dyn_Ar(H_avg)),
            (B_dim, L_dim, self.n, self.n)
        ) * self.alpha

        # d_B: (B, L, 1, n)
        d_B = tf.expand_dims(tf.nn.tanh(self.dyn_B(H_avg)), axis=2) * self.alpha

        # ========= ③ 静态 + 动态 =========
        # curr_Am: (B, L, n, 1)
        curr_Am = self.Am_static[None, None, :, :] + d_Am
        curr_Am = tf.nn.softmax(curr_Am, axis=2)

        # curr_Ar: (B, L, n, n)
        curr_Ar = self.Ar_static[None, None, :, :] + d_Ar

        # curr_B: (B, L, n, 1)
        curr_B = self.B_static[None, None, :, :] + d_B
        curr_B = tf.transpose(curr_B, perm=[0, 1, 3, 2])

        # ========= ④ Hyper aggregation =========
        # h0 = A_m^T H → (B, L, d)
        h0 = tf.reduce_sum(curr_Am * H, axis=2)

        # ========= ⑤ 子层计算 =========
        y = T(h0)  # (B, L, d)

        # ========= ⑥ 输出分配 =========
        # Hy: (B, L, n, d) - 对应公式 B^T * y
        Hy = curr_B * tf.expand_dims(y, axis=2)

        # Hr: (B, L, n, d) - 对应公式 Ar^T * H 
        Hr = tf.einsum("blji, bljd -> blid", curr_Ar, H)

        return Hy + Hr



class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, n=2, rate=0.1, name=None):
    super(EncoderLayer, self).__init__(name=name)
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.cond_proj = tf.keras.layers.Dense(d_model)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6) 
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

    self.n = n
    # self.hc_atten = HyperConnection(n, d_model, name=f"{self.name}_hc_atten")
    # self.hc_ffn = HyperConnection(n, d_model,  name=f"{self.name}_hc_ffn")
    self.hc_atten = DynamicHyperConnection(n, d_model, name=f"{self.name}_hc_atten")
    self.hc_ffn = DynamicHyperConnection(n, d_model,  name=f"{self.name}_hc_ffn")

  def call(self, H, x2, training=True, mask=None):
    def atten_fn(x):
      
      x=self.layernorm1(x)
      attn_output, _ = self.mha(x2, x, x, mask=mask)  
      attn_output = self.dropout1(attn_output, training=training)
      attn_output = self.cond_proj(attn_output)
      
      return attn_output 

    H=self.hc_atten(H, atten_fn)

    def ffn_fn(x):
      
      x=self.layernorm2(x) 
      ffn_output = self.ffn(x)  
      ffn_output = self.dropout2(ffn_output, training=training)
      
      return ffn_output
    
    H=self.hc_ffn(H, ffn_fn)

    return H

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)
  ])

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    self.dense = tf.keras.layers.Dense(d_model)
    
    # 输出维度改为 1，表示为每个 head 中的每个 token 生成一个 0-1 的权重
    self.token_gate_dense = tf.keras.layers.Dense(1, activation='sigmoid')
    

  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  # def call(self, q, k, v, context, mask=None):
  def call(self, q, k, v, mask):
    batch_size = tf.shape(q)[0]
    q = self.wq(q)  
    k = self.wk(k)  
    v = self.wv(v) 
    
    q = self.split_heads(q, batch_size) 
    k = self.split_heads(k, batch_size)  
    v = self.split_heads(v, batch_size)  
    
    scaled_attention, weights = scaled_dot_product_attention(q, k, v, mask)

    # 直接对 scaled_attention 进行变换。
    # Dense 层会自动作用在最后一个维度 (depth) 上。
    # gate_scores 形状: [batch, num_heads, seq_len, 1]
    gate_scores = self.token_gate_dense(scaled_attention) 

    # 利用广播机制 (Broadcasting)，每个 token 的 depth 维度共享同一个 gate_score
    gated_output = scaled_attention * gate_scores 

    gated_output = tf.transpose(gated_output, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(
        gated_output, (batch_size, -1, self.d_model)
    )

    output = self.dense(concat_attention)
    return output, weights
    
def scaled_dot_product_attention(q, k, v, mask=None):
  matmul_qk = tf.matmul(q, k, transpose_b=True)  
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  v = q + v 
  output = tf.matmul(attention_weights, v)  
  return output, attention_weights