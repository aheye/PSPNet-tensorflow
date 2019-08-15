import tensorflow as tf
from tensorflow.layers import conv2d

def fst(V_feats, L_feats, prev_fused, _lambda=0.1):
  """Feature Space Transformation function
  V_feats: Visual space features, 3d tensor
  L_feats: Lidar space features, 3d tensor
  """
  # Concatenate over feature map dimension
  C = tf.concat([V_feats, L_feats], axis=3)
  
  # Compute scalar and offset for FST linear function
  scalar = conv2d(C, 256, (1, 1), padding='same')
  offset = conv2d(C, 256, (1, 1), padding='same') 

  # Compute fst tensor
  fst = ((L_feats * scalar) + offset) + V_feats

  # Residucal based fuse function
  fused = prev_fused + (_lambda * fst)

  return fused


