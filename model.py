from network import Network
import tensorflow as tf

class PLARD(Network):
    def setup(self, is_training, num_classes, scale_channels=1):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
          scale_channels: scale the number of channels per layer by 1/`scale_channels`
        '''
        ## First Block Visual
        (self.feed('V_data')
             .conv(3, 3, 64, 2, 2, biased=False, relu=False, padding='SAME', name='V_conv1_1_3x3_s2')
             .batch_normalization(relu=False, name='V_conv1_1_3x3_s2_bn')
             .relu(name='V_conv1_1_3x3_s2_bn_relu')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='V_conv1_2_3x3')
             .batch_normalization(relu=True, name='V_conv1_2_3x3_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='V_conv1_3_3x3')
             .batch_normalization(relu=True, name='V_conv1_3_3x3_bn')
             .max_pool(3, 3, 2, 2, padding='SAME', name='V_pool1_3x3_s2'))

        ## First Block Lidar
        (self.feed('L_data')
             .conv(3, 3, 64/scale_channels, 2, 2, biased=False, relu=False, padding='SAME', name='L_conv1_1_3x3_s2')
             .batch_normalization(relu=False, name='L_conv1_1_3x3_s2_bn')
             .relu(name='L_conv1_1_3x3_s2_bn_relu')
             .conv(3, 3, 64/scale_channels, 1, 1, biased=False, relu=False, padding='SAME', name='L_conv1_2_3x3')
             .batch_normalization(relu=True, name='L_conv1_2_3x3_bn')
             .conv(3, 3, 128/scale_channels, 1, 1, biased=False, relu=False, padding='SAME', name='L_conv1_3_3x3')
             .batch_normalization(relu=True, name='L_conv1_3_3x3_bn')
             .max_pool(3, 3, 2, 2, padding='SAME', name='L_pool1_3x3_s2'))

        ## Compute first Feature Space Transform
        (self.feed('V_pool1_3x3_s2', 'L_pool1_3x3_s2')
             .fst(out_chan=128, name='fst1'))

        (self.feed('V_pool1_3x3_s2')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv2_1_1x1_proj')
             .batch_normalization(relu=False, name='V_conv2_1_1x1_proj_bn'))

        (self.feed('L_pool1_3x3_s2')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv2_1_1x1_proj')
             .batch_normalization(relu=False, name='L_conv2_1_1x1_proj_bn'))

        ## Second block, visual
        (self.feed('fst1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='V_conv2_1_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv2_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='V_padding1')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='V_conv2_1_3x3')
             .batch_normalization(relu=True, name='V_conv2_1_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv2_1_1x1_increase')
             .batch_normalization(relu=False, name='V_conv2_1_1x1_increase_bn'))

        (self.feed('V_conv2_1_1x1_proj_bn',
                   'V_conv2_1_1x1_increase_bn')
             .add(name='V_conv2_1')
             .relu(name='V_conv2_1/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='V_conv2_2_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv2_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='V_padding2')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='V_conv2_2_3x3')
             .batch_normalization(relu=True, name='V_conv2_2_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv2_2_1x1_increase')
             .batch_normalization(relu=False, name='V_conv2_2_1x1_increase_bn'))

        (self.feed('V_conv2_1/relu',
                   'V_conv2_2_1x1_increase_bn')
             .add(name='V_conv2_2')
             .relu(name='V_conv2_2/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='V_conv2_3_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv2_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='V_padding3')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='V_conv2_3_3x3')
             .batch_normalization(relu=True, name='conv2_3_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv2_3_1x1_increase')
             .batch_normalization(relu=False, name='V_conv2_3_1x1_increase_bn'))

        (self.feed('V_conv2_2/relu',
                   'V_conv2_3_1x1_increase_bn')
             .add(name='V_conv2_3')
             .relu(name='V_conv2_3/relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='V_conv3_1_1x1_proj')
             .batch_normalization(relu=False, name='V_conv3_1_1x1_proj_bn'))

        ## Second block, lidar
        (self.feed('L_pool1_3x3_s2')
             .conv(1, 1, 64/scale_channels, 1, 1, biased=False, relu=False, name='L_conv2_1_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv2_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='L_padding1')
             .conv(3, 3, 64/scale_channels, 1, 1, biased=False, relu=False, name='L_conv2_1_3x3')
             .batch_normalization(relu=True, name='L_conv2_1_3x3_bn')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv2_1_1x1_increase')
             .batch_normalization(relu=False, name='L_conv2_1_1x1_increase_bn'))

        (self.feed('L_conv2_1_1x1_proj_bn',
                   'L_conv2_1_1x1_increase_bn')
             .add(name='L_conv2_1')
             .relu(name='L_conv2_1/relu')
             .conv(1, 1, 64/scale_channels, 1, 1, biased=False, relu=False, name='L_conv2_2_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv2_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='L_padding2')
             .conv(3, 3, 64/scale_channels, 1, 1, biased=False, relu=False, name='L_conv2_2_3x3')
             .batch_normalization(relu=True, name='L_conv2_2_3x3_bn')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv2_2_1x1_increase')
             .batch_normalization(relu=False, name='L_conv2_2_1x1_increase_bn'))

        (self.feed('L_conv2_1/relu',
                   'L_conv2_2_1x1_increase_bn')
             .add(name='L_conv2_2')
             .relu(name='L_conv2_2/relu')
             .conv(1, 1, 64/scale_channels, 1, 1, biased=False, relu=False, name='L_conv2_3_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv2_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='L_padding3')
             .conv(3, 3, 64/scale_channels, 1, 1, biased=False, relu=False, name='L_conv2_3_3x3')
             .batch_normalization(relu=True, name='L_conv2_3_3x3_bn')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv2_3_1x1_increase')
             .batch_normalization(relu=False, name='L_conv2_3_1x1_increase_bn'))

        (self.feed('L_conv2_2/relu',
                   'L_conv2_3_1x1_increase_bn')
             .add(name='L_conv2_3')
             .relu(name='L_conv2_3/relu')
             .conv(1, 1, 512/scale_channels, 2, 2, biased=False, relu=False, name='L_conv3_1_1x1_proj')
             .batch_normalization(relu=False, name='L_conv3_1_1x1_proj_bn'))

        ## Compute Second Feature Space Transformation
        (self.feed('V_conv2_3/relu', 'L_conv2_3/relu')
            .fst(out_chan=256, prev_fused=self.layers['fst1'], name='fst2'))

        ## Third Block, visual
        (self.feed('fst2')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='V_conv3_1_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv3_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='V_padding4')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='V_conv3_1_3x3')
             .batch_normalization(relu=True, name='V_conv3_1_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='V_conv3_1_1x1_increase')
             .batch_normalization(relu=False, name='V_conv3_1_1x1_increase_bn'))

        (self.feed('V_conv3_1_1x1_proj_bn',
                   'V_conv3_1_1x1_increase_bn')
             .add(name='V_conv3_1')
             .relu(name='V_conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='V_conv3_2_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv3_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='V_padding5')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='V_conv3_2_3x3')
             .batch_normalization(relu=True, name='V_conv3_2_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='V_conv3_2_1x1_increase')
             .batch_normalization(relu=False, name='V_conv3_2_1x1_increase_bn'))

        (self.feed('V_conv3_1/relu',
                   'V_conv3_2_1x1_increase_bn')
             .add(name='V_conv3_2')
             .relu(name='V_conv3_2/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='V_conv3_3_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv3_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='V_padding6')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='V_conv3_3_3x3')
             .batch_normalization(relu=True, name='V_conv3_3_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='V_conv3_3_1x1_increase')
             .batch_normalization(relu=False, name='V_conv3_3_1x1_increase_bn'))

        (self.feed('V_conv3_2/relu',
                   'V_conv3_3_1x1_increase_bn')
             .add(name='V_conv3_3')
             .relu(name='V_conv3_3/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='V_conv3_4_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv3_4_1x1_reduce_bn')
             .zero_padding(paddings=1, name='V_padding7')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='V_conv3_4_3x3')
             .batch_normalization(relu=True, name='V_conv3_4_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='V_conv3_4_1x1_increase')
             .batch_normalization(relu=False, name='V_conv3_4_1x1_increase_bn'))

        (self.feed('V_conv3_3/relu',
                   'V_conv3_4_1x1_increase_bn')
             .add(name='V_conv3_4')
             .relu(name='V_conv3_4/relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_1_1x1_proj')
             .batch_normalization(relu=False, name='V_conv4_1_1x1_proj_bn'))

        ## Third Block, lidar
        (self.feed('L_conv2_3/relu')
             .conv(1, 1, 128/scale_channels, 2, 2, biased=False, relu=False, name='L_conv3_1_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv3_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='L_padding4')
             .conv(3, 3, 128/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_1_3x3')
             .batch_normalization(relu=True, name='L_conv3_1_3x3_bn')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_1_1x1_increase')
             .batch_normalization(relu=False, name='L_conv3_1_1x1_increase_bn'))

        (self.feed('L_conv3_1_1x1_proj_bn',
                   'L_conv3_1_1x1_increase_bn')
             .add(name='L_conv3_1')
             .relu(name='L_conv3_1/relu')
             .conv(1, 1, 128/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_2_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv3_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='L_padding5')
             .conv(3, 3, 128/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_2_3x3')
             .batch_normalization(relu=True, name='L_conv3_2_3x3_bn')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_2_1x1_increase')
             .batch_normalization(relu=False, name='L_conv3_2_1x1_increase_bn'))

        (self.feed('L_conv3_1/relu',
                   'L_conv3_2_1x1_increase_bn')
             .add(name='L_conv3_2')
             .relu(name='L_conv3_2/relu')
             .conv(1, 1, 128/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_3_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv3_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='L_padding6')
             .conv(3, 3, 128/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_3_3x3')
             .batch_normalization(relu=True, name='L_conv3_3_3x3_bn')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_3_1x1_increase')
             .batch_normalization(relu=False, name='L_conv3_3_1x1_increase_bn'))

        (self.feed('L_conv3_2/relu',
                   'L_conv3_3_1x1_increase_bn')
             .add(name='L_conv3_3')
             .relu(name='L_conv3_3/relu')
             .conv(1, 1, 128/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_4_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv3_4_1x1_reduce_bn')
             .zero_padding(paddings=1, name='L_padding7')
             .conv(3, 3, 128/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_4_3x3')
             .batch_normalization(relu=True, name='L_conv3_4_3x3_bn')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='L_conv3_4_1x1_increase')
             .batch_normalization(relu=False, name='L_conv3_4_1x1_increase_bn'))

        (self.feed('L_conv3_3/relu',
                   'L_conv3_4_1x1_increase_bn')
             .add(name='L_conv3_4')
             .relu(name='L_conv3_4/relu')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_1_1x1_proj')
             .batch_normalization(relu=False, name='L_conv4_1_1x1_proj_bn'))

        ## Compute Third Feature Space Transformation
        (self.feed('V_conv3_4/relu', 'L_conv3_4/relu')
            .fst(out_chan=512, prev_fused=self.layers['fst2'], name='fst3'))

        ## Fourth Block, visual
        (self.feed('fst3')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_1_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding8')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_1_3x3')
             .batch_normalization(relu=True, name='V_conv4_1_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_1_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_1_1x1_increase_bn'))

        (self.feed('V_conv4_1_1x1_proj_bn',
                   'V_conv4_1_1x1_increase_bn')
             .add(name='V_conv4_1')
             .relu(name='V_conv4_1/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_2_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding9')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_2_3x3')
             .batch_normalization(relu=True, name='V_conv4_2_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_2_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_2_1x1_increase_bn'))

        (self.feed('V_conv4_1/relu',
                   'V_conv4_2_1x1_increase_bn')
             .add(name='V_conv4_2')
             .relu(name='V_conv4_2/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_3_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_3_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding10')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_3_3x3')
             .batch_normalization(relu=True, name='V_conv4_3_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_3_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_3_1x1_increase_bn'))

        (self.feed('V_conv4_2/relu',
                   'V_conv4_3_1x1_increase_bn')
             .add(name='V_conv4_3')
             .relu(name='V_conv4_3/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_4_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_4_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding11')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_4_3x3')
             .batch_normalization(relu=True, name='V_conv4_4_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_4_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_4_1x1_increase_bn'))

        (self.feed('V_conv4_3/relu',
                   'V_conv4_4_1x1_increase_bn')
             .add(name='V_conv4_4')
             .relu(name='V_conv4_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_5_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_5_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding12')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_5_3x3')
             .batch_normalization(relu=True, name='V_conv4_5_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_5_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_5_1x1_increase_bn'))

        (self.feed('V_conv4_4/relu',
                   'V_conv4_5_1x1_increase_bn')
             .add(name='V_conv4_5')
             .relu(name='V_conv4_5/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_6_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_6_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding13')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_6_3x3')
             .batch_normalization(relu=True, name='V_conv4_6_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_6_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_6_1x1_increase_bn'))

        (self.feed('V_conv4_5/relu',
                   'V_conv4_6_1x1_increase_bn')
             .add(name='V_conv4_6')
             .relu(name='V_conv4_6/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_7_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_7_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding14')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_7_3x3')
             .batch_normalization(relu=True, name='V_conv4_7_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_7_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_7_1x1_increase_bn'))

        (self.feed('V_conv4_6/relu',
                   'V_conv4_7_1x1_increase_bn')
             .add(name='V_conv4_7')
             .relu(name='V_conv4_7/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_8_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_8_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding15')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_8_3x3')
             .batch_normalization(relu=True, name='V_conv4_8_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_8_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_8_1x1_increase_bn'))

        (self.feed('V_conv4_7/relu',
                   'V_conv4_8_1x1_increase_bn')
             .add(name='V_conv4_8')
             .relu(name='V_conv4_8/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_9_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_9_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding16')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_9_3x3')
             .batch_normalization(relu=True, name='V_conv4_9_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_9_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_9_1x1_increase_bn'))

        (self.feed('V_conv4_8/relu',
                   'V_conv4_9_1x1_increase_bn')
             .add(name='V_conv4_9')
             .relu(name='V_conv4_9/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_10_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_10_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding17')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_10_3x3')
             .batch_normalization(relu=True, name='V_conv4_10_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_10_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_10_1x1_increase_bn'))

        (self.feed('V_conv4_9/relu',
                   'V_conv4_10_1x1_increase_bn')
             .add(name='V_conv4_10')
             .relu(name='V_conv4_10/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_11_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_11_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding18')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_11_3x3')
             .batch_normalization(relu=True, name='V_conv4_11_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_11_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_11_1x1_increase_bn'))

        (self.feed('V_conv4_10/relu',
                   'V_conv4_11_1x1_increase_bn')
             .add(name='V_conv4_11')
             .relu(name='V_conv4_11/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_12_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_12_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding19')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_12_3x3')
             .batch_normalization(relu=True, name='V_conv4_12_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_12_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_12_1x1_increase_bn'))

        (self.feed('V_conv4_11/relu',
                   'V_conv4_12_1x1_increase_bn')
             .add(name='V_conv4_12')
             .relu(name='V_conv4_12/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_13_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_13_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding20')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_13_3x3')
             .batch_normalization(relu=True, name='V_conv4_13_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_13_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_13_1x1_increase_bn'))

        (self.feed('V_conv4_12/relu',
                   'V_conv4_13_1x1_increase_bn')
             .add(name='V_conv4_13')
             .relu(name='V_conv4_13/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_14_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_14_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding21')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_14_3x3')
             .batch_normalization(relu=True, name='V_conv4_14_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_14_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_14_1x1_increase_bn'))

        (self.feed('V_conv4_13/relu',
                   'V_conv4_14_1x1_increase_bn')
             .add(name='V_conv4_14')
             .relu(name='V_conv4_14/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_15_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_15_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding22')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_15_3x3')
             .batch_normalization(relu=True, name='V_conv4_15_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_15_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_15_1x1_increase_bn'))

        (self.feed('V_conv4_14/relu',
                   'V_conv4_15_1x1_increase_bn')
             .add(name='V_conv4_15')
             .relu(name='V_conv4_15/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_16_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_16_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding23')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_16_3x3')
             .batch_normalization(relu=True, name='V_conv4_16_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_16_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_16_1x1_increase_bn'))

        (self.feed('V_conv4_15/relu',
                   'V_conv4_16_1x1_increase_bn')
             .add(name='V_conv4_16')
             .relu(name='V_conv4_16/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_17_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_17_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding24')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_17_3x3')
             .batch_normalization(relu=True, name='V_conv4_17_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_17_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_17_1x1_increase_bn'))

        (self.feed('V_conv4_16/relu',
                   'V_conv4_17_1x1_increase_bn')
             .add(name='V_conv4_17')
             .relu(name='V_conv4_17/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_18_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_18_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding25')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_18_3x3')
             .batch_normalization(relu=True, name='V_conv4_18_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_18_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_18_1x1_increase_bn'))

        (self.feed('V_conv4_17/relu',
                   'V_conv4_18_1x1_increase_bn')
             .add(name='V_conv4_18')
             .relu(name='V_conv4_18/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_19_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_19_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding26')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_19_3x3')
             .batch_normalization(relu=True, name='V_conv4_19_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_19_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_19_1x1_increase_bn'))

        (self.feed('V_conv4_18/relu',
                   'V_conv4_19_1x1_increase_bn')
             .add(name='V_conv4_19')
             .relu(name='V_conv4_19/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_20_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_20_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding27')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_20_3x3')
             .batch_normalization(relu=True, name='V_conv4_20_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_20_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_20_1x1_increase_bn'))

        (self.feed('V_conv4_19/relu',
                   'V_conv4_20_1x1_increase_bn')
             .add(name='V_conv4_20')
             .relu(name='V_conv4_20/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_21_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_21_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding28')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_21_3x3')
             .batch_normalization(relu=True, name='V_conv4_21_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_21_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_21_1x1_increase_bn'))

        (self.feed('V_conv4_20/relu',
                   'V_conv4_21_1x1_increase_bn')
             .add(name='V_conv4_21')
             .relu(name='V_conv4_21/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_22_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_22_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding29')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_22_3x3')
             .batch_normalization(relu=True, name='V_conv4_22_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_22_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_22_1x1_increase_bn'))

        (self.feed('V_conv4_21/relu',
                   'V_conv4_22_1x1_increase_bn')
             .add(name='V_conv4_22')
             .relu(name='V_conv4_22/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='V_conv4_23_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv4_23_1x1_reduce_bn')
             .zero_padding(paddings=2, name='V_padding30')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='V_conv4_23_3x3')
             .batch_normalization(relu=True, name='V_conv4_23_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='V_conv4_23_1x1_increase')
             .batch_normalization(relu=False, name='V_conv4_23_1x1_increase_bn'))

        (self.feed('V_conv4_22/relu',
                   'V_conv4_23_1x1_increase_bn')
             .add(name='V_conv4_23')
             .relu(name='V_conv4_23/relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='V_conv5_1_1x1_proj')
             .batch_normalization(relu=False, name='V_conv5_1_1x1_proj_bn'))

        ## Start of fourth block, lidar
        (self.feed('L_conv3_4/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_1_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding8')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_1_3x3')
             .batch_normalization(relu=True, name='L_conv4_1_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_1_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_1_1x1_increase_bn'))

        (self.feed('L_conv4_1_1x1_proj_bn',
                   'L_conv4_1_1x1_increase_bn')
             .add(name='L_conv4_1')
             .relu(name='L_conv4_1/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_2_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding9')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_2_3x3')
             .batch_normalization(relu=True, name='L_conv4_2_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_2_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_2_1x1_increase_bn'))

        (self.feed('L_conv4_1/relu',
                   'L_conv4_2_1x1_increase_bn')
             .add(name='L_conv4_2')
             .relu(name='L_conv4_2/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_3_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_3_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding10')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_3_3x3')
             .batch_normalization(relu=True, name='L_conv4_3_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_3_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_3_1x1_increase_bn'))

        (self.feed('L_conv4_2/relu',
                   'L_conv4_3_1x1_increase_bn')
             .add(name='L_conv4_3')
             .relu(name='L_conv4_3/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_4_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_4_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding11')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_4_3x3')
             .batch_normalization(relu=True, name='L_conv4_4_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_4_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_4_1x1_increase_bn'))

        (self.feed('L_conv4_3/relu',
                   'L_conv4_4_1x1_increase_bn')
             .add(name='L_conv4_4')
             .relu(name='L_conv4_4/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_5_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_5_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding12')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_5_3x3')
             .batch_normalization(relu=True, name='L_conv4_5_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_5_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_5_1x1_increase_bn'))

        (self.feed('L_conv4_4/relu',
                   'L_conv4_5_1x1_increase_bn')
             .add(name='L_conv4_5')
             .relu(name='L_conv4_5/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_6_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_6_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding13')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_6_3x3')
             .batch_normalization(relu=True, name='L_conv4_6_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_6_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_6_1x1_increase_bn'))

        (self.feed('L_conv4_5/relu',
                   'L_conv4_6_1x1_increase_bn')
             .add(name='L_conv4_6')
             .relu(name='L_conv4_6/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_7_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_7_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding14')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_7_3x3')
             .batch_normalization(relu=True, name='L_conv4_7_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_7_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_7_1x1_increase_bn'))

        (self.feed('L_conv4_6/relu',
                   'L_conv4_7_1x1_increase_bn')
             .add(name='L_conv4_7')
             .relu(name='L_conv4_7/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_8_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_8_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding15')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_8_3x3')
             .batch_normalization(relu=True, name='L_conv4_8_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_8_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_8_1x1_increase_bn'))

        (self.feed('L_conv4_7/relu',
                   'L_conv4_8_1x1_increase_bn')
             .add(name='L_conv4_8')
             .relu(name='L_conv4_8/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_9_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_9_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding16')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_9_3x3')
             .batch_normalization(relu=True, name='L_conv4_9_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_9_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_9_1x1_increase_bn'))

        (self.feed('L_conv4_8/relu',
                   'L_conv4_9_1x1_increase_bn')
             .add(name='L_conv4_9')
             .relu(name='L_conv4_9/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_10_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_10_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding17')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_10_3x3')
             .batch_normalization(relu=True, name='L_conv4_10_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_10_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_10_1x1_increase_bn'))

        (self.feed('L_conv4_9/relu',
                   'L_conv4_10_1x1_increase_bn')
             .add(name='L_conv4_10')
             .relu(name='L_conv4_10/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_11_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_11_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding18')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_11_3x3')
             .batch_normalization(relu=True, name='L_conv4_11_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_11_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_11_1x1_increase_bn'))

        (self.feed('L_conv4_10/relu',
                   'L_conv4_11_1x1_increase_bn')
             .add(name='L_conv4_11')
             .relu(name='L_conv4_11/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_12_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_12_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding19')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_12_3x3')
             .batch_normalization(relu=True, name='L_conv4_12_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_12_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_12_1x1_increase_bn'))

        (self.feed('L_conv4_11/relu',
                   'L_conv4_12_1x1_increase_bn')
             .add(name='L_conv4_12')
             .relu(name='L_conv4_12/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_13_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_13_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding20')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_13_3x3')
             .batch_normalization(relu=True, name='L_conv4_13_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_13_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_13_1x1_increase_bn'))

        (self.feed('L_conv4_12/relu',
                   'L_conv4_13_1x1_increase_bn')
             .add(name='L_conv4_13')
             .relu(name='L_conv4_13/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_14_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_14_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding21')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_14_3x3')
             .batch_normalization(relu=True, name='L_conv4_14_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_14_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_14_1x1_increase_bn'))

        (self.feed('L_conv4_13/relu',
                   'L_conv4_14_1x1_increase_bn')
             .add(name='L_conv4_14')
             .relu(name='L_conv4_14/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_15_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_15_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding22')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_15_3x3')
             .batch_normalization(relu=True, name='L_conv4_15_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_15_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_15_1x1_increase_bn'))

        (self.feed('L_conv4_14/relu',
                   'L_conv4_15_1x1_increase_bn')
             .add(name='L_conv4_15')
             .relu(name='L_conv4_15/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_16_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_16_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding23')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_16_3x3')
             .batch_normalization(relu=True, name='L_conv4_16_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_16_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_16_1x1_increase_bn'))

        (self.feed('L_conv4_15/relu',
                   'L_conv4_16_1x1_increase_bn')
             .add(name='L_conv4_16')
             .relu(name='L_conv4_16/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_17_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_17_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding24')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_17_3x3')
             .batch_normalization(relu=True, name='L_conv4_17_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_17_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_17_1x1_increase_bn'))

        (self.feed('L_conv4_16/relu',
                   'L_conv4_17_1x1_increase_bn')
             .add(name='L_conv4_17')
             .relu(name='L_conv4_17/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_18_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_18_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding25')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_18_3x3')
             .batch_normalization(relu=True, name='L_conv4_18_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_18_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_18_1x1_increase_bn'))

        (self.feed('L_conv4_17/relu',
                   'L_conv4_18_1x1_increase_bn')
             .add(name='L_conv4_18')
             .relu(name='L_conv4_18/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_19_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_19_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding26')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_19_3x3')
             .batch_normalization(relu=True, name='L_conv4_19_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_19_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_19_1x1_increase_bn'))

        (self.feed('L_conv4_18/relu',
                   'L_conv4_19_1x1_increase_bn')
             .add(name='L_conv4_19')
             .relu(name='L_conv4_19/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_20_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_20_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding27')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_20_3x3')
             .batch_normalization(relu=True, name='L_conv4_20_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_20_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_20_1x1_increase_bn'))

        (self.feed('L_conv4_19/relu',
                   'L_conv4_20_1x1_increase_bn')
             .add(name='L_conv4_20')
             .relu(name='L_conv4_20/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_21_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_21_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding28')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_21_3x3')
             .batch_normalization(relu=True, name='L_conv4_21_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_21_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_21_1x1_increase_bn'))

        (self.feed('L_conv4_20/relu',
                   'L_conv4_21_1x1_increase_bn')
             .add(name='L_conv4_21')
             .relu(name='L_conv4_21/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_22_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_22_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding29')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_22_3x3')
             .batch_normalization(relu=True, name='L_conv4_22_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_22_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_22_1x1_increase_bn'))

        (self.feed('L_conv4_21/relu',
                   'L_conv4_22_1x1_increase_bn')
             .add(name='L_conv4_22')
             .relu(name='L_conv4_22/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_23_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv4_23_1x1_reduce_bn')
             .zero_padding(paddings=2, name='L_padding30')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='L_conv4_23_3x3')
             .batch_normalization(relu=True, name='L_conv4_23_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='L_conv4_23_1x1_increase')
             .batch_normalization(relu=False, name='L_conv4_23_1x1_increase_bn'))

        (self.feed('L_conv4_22/relu',
                   'L_conv4_23_1x1_increase_bn')
             .add(name='L_conv4_23')
             .relu(name='L_conv4_23/relu')
             .conv(1, 1, 2048/scale_channels, 1, 1, biased=False, relu=False, name='L_conv5_1_1x1_proj')
             .batch_normalization(relu=False, name='L_conv5_1_1x1_proj_bn'))

        ## Compute Fourth Feature Space Transformation
        (self.feed('V_conv4_23/relu', 'L_conv4_23/relu')
            .fst(out_chan=256, prev_fused=self.layers['fst3'], name='fst4'))

        ## Fifth Block, visual
        (self.feed('fst4')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='V_conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv5_1_1x1_reduce_bn')
             .zero_padding(paddings=4, name='V_padding31')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='V_conv5_1_3x3')
             .batch_normalization(relu=True, name='V_conv5_1_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='V_conv5_1_1x1_increase')
             .batch_normalization(relu=False, name='V_conv5_1_1x1_increase_bn'))

        (self.feed('V_conv5_1_1x1_proj_bn',
                   'V_conv5_1_1x1_increase_bn')
             .add(name='V_conv5_1')
             .relu(name='V_conv5_1/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='V_conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv5_2_1x1_reduce_bn')
             .zero_padding(paddings=4, name='V_padding32')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='V_conv5_2_3x3')
             .batch_normalization(relu=True, name='V_conv5_2_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='V_conv5_2_1x1_increase')
             .batch_normalization(relu=False, name='V_conv5_2_1x1_increase_bn'))

        (self.feed('V_conv5_1/relu',
                   'V_conv5_2_1x1_increase_bn')
             .add(name='V_conv5_2')
             .relu(name='V_conv5_2/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='V_conv5_3_1x1_reduce')
             .batch_normalization(relu=True, name='V_conv5_3_1x1_reduce_bn')
             .zero_padding(paddings=4, name='V_padding33')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='V_conv5_3_3x3')
             .batch_normalization(relu=True, name='V_conv5_3_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='V_conv5_3_1x1_increase')
             .batch_normalization(relu=False, name='V_conv5_3_1x1_increase_bn'))

        (self.feed('V_conv5_2/relu',
                   'V_conv5_3_1x1_increase_bn')
             .add(name='V_conv5_3')
             .relu(name='V_conv5_3/relu'))

        ## Fifth Block, lidar
        (self.feed('L_conv4_23/relu')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='L_conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv5_1_1x1_reduce_bn')
             .zero_padding(paddings=4, name='L_padding31')
             .atrous_conv(3, 3, 512/scale_channels, 4, biased=False, relu=False, name='L_conv5_1_3x3')
             .batch_normalization(relu=True, name='L_conv5_1_3x3_bn')
             .conv(1, 1, 2048/scale_channels, 1, 1, biased=False, relu=False, name='L_conv5_1_1x1_increase')
             .batch_normalization(relu=False, name='L_conv5_1_1x1_increase_bn'))

        (self.feed('L_conv5_1_1x1_proj_bn',
                   'L_conv5_1_1x1_increase_bn')
             .add(name='L_conv5_1')
             .relu(name='L_conv5_1/relu')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='L_conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv5_2_1x1_reduce_bn')
             .zero_padding(paddings=4, name='L_padding32')
             .atrous_conv(3, 3, 512/scale_channels, 4, biased=False, relu=False, name='L_conv5_2_3x3')
             .batch_normalization(relu=True, name='L_conv5_2_3x3_bn')
             .conv(1, 1, 2048/scale_channels, 1, 1, biased=False, relu=False, name='L_conv5_2_1x1_increase')
             .batch_normalization(relu=False, name='L_conv5_2_1x1_increase_bn'))

        (self.feed('L_conv5_1/relu',
                   'L_conv5_2_1x1_increase_bn')
             .add(name='L_conv5_2')
             .relu(name='L_conv5_2/relu')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='L_conv5_3_1x1_reduce')
             .batch_normalization(relu=True, name='L_conv5_3_1x1_reduce_bn')
             .zero_padding(paddings=4, name='L_padding33')
             .atrous_conv(3, 3, 512/scale_channels, 4, biased=False, relu=False, name='L_conv5_3_3x3')
             .batch_normalization(relu=True, name='L_conv5_3_3x3_bn')
             .conv(1, 1, 2048/scale_channels, 1, 1, biased=False, relu=False, name='L_conv5_3_1x1_increase')
             .batch_normalization(relu=False, name='L_conv5_3_1x1_increase_bn'))

        (self.feed('L_conv5_2/relu',
                   'L_conv5_3_1x1_increase_bn')
             .add(name='L_conv5_3')
             .relu(name='L_conv5_3/relu'))

        ## Compute Fifth Feature Space Transformation
        (self.feed('V_conv5_3/relu', 'L_conv5_3/relu')
            .fst(out_chan=2048, prev_fused=self.layers['fst4'], name='fst5'))


        conv5_3 = self.layers['fst5']
        shape = tf.shape(conv5_3)[1:3]

        (self.feed('fst5')
             .avg_pool(90, 90, 90, 90, name='conv5_3_pool1')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool1_conv')
             .batch_normalization(relu=True, name='conv5_3_pool1_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('fst5')
             .avg_pool(45, 45, 45, 45, name='conv5_3_pool2')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool2_conv')
             .batch_normalization(relu=True, name='conv5_3_pool2_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('fst5')
             .avg_pool(30, 30, 30, 30, name='conv5_3_pool3')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool3_conv')
             .batch_normalization(relu=True, name='conv5_3_pool3_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('fst5')
             .avg_pool(15, 15, 15, 15, name='conv5_3_pool6')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool6_conv')
             .batch_normalization(relu=True, name='conv5_3_pool6_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('fst5',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .concat(axis=-1, name='conv5_3_concat')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_4')
             .batch_normalization(relu=True, name='conv5_4_bn')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6'))


class PSPNet101(Network):
    def setup(self, is_training, num_classes, scale_channels=1):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
          scale_channels: scale the number of channels per layer by 1/`scale_channels`
        '''
        (self.feed('data')
             .conv(3, 3, 64/scale_channels, 2, 2, biased=False, relu=False, padding='SAME', name='conv1_1_3x3_s2')
             .batch_normalization(relu=False, name='conv1_1_3x3_s2_bn')
             .relu(name='conv1_1_3x3_s2_bn_relu')
             .conv(3, 3, 64/scale_channels, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_2_3x3')
             .batch_normalization(relu=True, name='conv1_2_3x3_bn')
             .conv(3, 3, 128/scale_channels, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_3_3x3')
             .batch_normalization(relu=True, name='conv1_3_3x3_bn')
             .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj')
             .batch_normalization(relu=False, name='conv2_1_1x1_proj_bn'))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 64/scale_channels, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 64/scale_channels, 1, 1, biased=False, relu=False, name='conv2_1_3x3')
             .batch_normalization(relu=True, name='conv2_1_3x3_bn')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase')
             .batch_normalization(relu=False, name='conv2_1_1x1_increase_bn'))

        (self.feed('conv2_1_1x1_proj_bn',
                   'conv2_1_1x1_increase_bn')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 64/scale_channels, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 64/scale_channels, 1, 1, biased=False, relu=False, name='conv2_2_3x3')
             .batch_normalization(relu=True, name='conv2_2_3x3_bn')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')
             .batch_normalization(relu=False, name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase_bn')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 64/scale_channels, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 64/scale_channels, 1, 1, biased=False, relu=False, name='conv2_3_3x3')
             .batch_normalization(relu=True, name='conv2_3_3x3_bn')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')
             .batch_normalization(relu=False, name='conv2_3_1x1_increase_bn'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase_bn')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 512/scale_channels, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj')
             .batch_normalization(relu=False, name='conv3_1_1x1_proj_bn'))

         ## Start of Block 3
        (self.feed('conv2_3/relu')
             .conv(1, 1, 128/scale_channels, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 128/scale_channels, 1, 1, biased=False, relu=False, name='conv3_1_3x3')
             .batch_normalization(relu=True, name='conv3_1_3x3_bn')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')
             .batch_normalization(relu=False, name='conv3_1_1x1_increase_bn'))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_1x1_increase_bn')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .conv(1, 1, 128/scale_channels, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 128/scale_channels, 1, 1, biased=False, relu=False, name='conv3_2_3x3')
             .batch_normalization(relu=True, name='conv3_2_3x3_bn')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')
             .batch_normalization(relu=False, name='conv3_2_1x1_increase_bn'))

        (self.feed('conv3_1/relu',
                   'conv3_2_1x1_increase_bn')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 128/scale_channels, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 128/scale_channels, 1, 1, biased=False, relu=False, name='conv3_3_3x3')
             .batch_normalization(relu=True, name='conv3_3_3x3_bn')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')
             .batch_normalization(relu=False, name='conv3_3_1x1_increase_bn'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase_bn')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 128/scale_channels, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 128/scale_channels, 1, 1, biased=False, relu=False, name='conv3_4_3x3')
             .batch_normalization(relu=True, name='conv3_4_3x3_bn')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')
             .batch_normalization(relu=False, name='conv3_4_1x1_increase_bn'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase_bn')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj')
             .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn'))

        ## Block 4
        (self.feed('conv3_4/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_1_3x3')
             .batch_normalization(relu=True, name='conv4_1_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
             .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_2_3x3')
             .batch_normalization(relu=True, name='conv4_2_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
             .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase_bn')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_3_3x3')
             .batch_normalization(relu=True, name='conv4_3_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')
             .batch_normalization(relu=False, name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase_bn')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_4_3x3')
             .batch_normalization(relu=True, name='conv4_4_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')
             .batch_normalization(relu=False, name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase_bn')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_5_3x3')
             .batch_normalization(relu=True, name='conv4_5_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')
             .batch_normalization(relu=False, name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase_bn')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_6_3x3')
             .batch_normalization(relu=True, name='conv4_6_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')
             .batch_normalization(relu=False, name='conv4_6_1x1_increase_bn'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase_bn')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_7_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_7_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding14')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_7_3x3')
             .batch_normalization(relu=True, name='conv4_7_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_7_1x1_increase')
             .batch_normalization(relu=False, name='conv4_7_1x1_increase_bn'))

        (self.feed('conv4_6/relu',
                   'conv4_7_1x1_increase_bn')
             .add(name='conv4_7')
             .relu(name='conv4_7/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_8_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_8_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding15')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_8_3x3')
             .batch_normalization(relu=True, name='conv4_8_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_8_1x1_increase')
             .batch_normalization(relu=False, name='conv4_8_1x1_increase_bn'))

        (self.feed('conv4_7/relu',
                   'conv4_8_1x1_increase_bn')
             .add(name='conv4_8')
             .relu(name='conv4_8/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_9_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_9_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding16')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_9_3x3')
             .batch_normalization(relu=True, name='conv4_9_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_9_1x1_increase')
             .batch_normalization(relu=False, name='conv4_9_1x1_increase_bn'))

        (self.feed('conv4_8/relu',
                   'conv4_9_1x1_increase_bn')
             .add(name='conv4_9')
             .relu(name='conv4_9/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_10_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_10_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding17')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_10_3x3')
             .batch_normalization(relu=True, name='conv4_10_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_10_1x1_increase')
             .batch_normalization(relu=False, name='conv4_10_1x1_increase_bn'))

        (self.feed('conv4_9/relu',
                   'conv4_10_1x1_increase_bn')
             .add(name='conv4_10')
             .relu(name='conv4_10/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_11_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_11_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding18')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_11_3x3')
             .batch_normalization(relu=True, name='conv4_11_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_11_1x1_increase')
             .batch_normalization(relu=False, name='conv4_11_1x1_increase_bn'))

        (self.feed('conv4_10/relu',
                   'conv4_11_1x1_increase_bn')
             .add(name='conv4_11')
             .relu(name='conv4_11/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_12_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_12_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding19')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_12_3x3')
             .batch_normalization(relu=True, name='conv4_12_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_12_1x1_increase')
             .batch_normalization(relu=False, name='conv4_12_1x1_increase_bn'))

        (self.feed('conv4_11/relu',
                   'conv4_12_1x1_increase_bn')
             .add(name='conv4_12')
             .relu(name='conv4_12/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_13_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_13_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding20')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_13_3x3')
             .batch_normalization(relu=True, name='conv4_13_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_13_1x1_increase')
             .batch_normalization(relu=False, name='conv4_13_1x1_increase_bn'))

        (self.feed('conv4_12/relu',
                   'conv4_13_1x1_increase_bn')
             .add(name='conv4_13')
             .relu(name='conv4_13/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_14_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_14_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding21')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_14_3x3')
             .batch_normalization(relu=True, name='conv4_14_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_14_1x1_increase')
             .batch_normalization(relu=False, name='conv4_14_1x1_increase_bn'))

        (self.feed('conv4_13/relu',
                   'conv4_14_1x1_increase_bn')
             .add(name='conv4_14')
             .relu(name='conv4_14/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_15_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_15_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding22')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_15_3x3')
             .batch_normalization(relu=True, name='conv4_15_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_15_1x1_increase')
             .batch_normalization(relu=False, name='conv4_15_1x1_increase_bn'))

        (self.feed('conv4_14/relu',
                   'conv4_15_1x1_increase_bn')
             .add(name='conv4_15')
             .relu(name='conv4_15/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_16_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_16_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding23')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_16_3x3')
             .batch_normalization(relu=True, name='conv4_16_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_16_1x1_increase')
             .batch_normalization(relu=False, name='conv4_16_1x1_increase_bn'))

        (self.feed('conv4_15/relu',
                   'conv4_16_1x1_increase_bn')
             .add(name='conv4_16')
             .relu(name='conv4_16/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_17_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_17_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding24')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_17_3x3')
             .batch_normalization(relu=True, name='conv4_17_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_17_1x1_increase')
             .batch_normalization(relu=False, name='conv4_17_1x1_increase_bn'))

        (self.feed('conv4_16/relu',
                   'conv4_17_1x1_increase_bn')
             .add(name='conv4_17')
             .relu(name='conv4_17/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_18_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_18_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding25')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_18_3x3')
             .batch_normalization(relu=True, name='conv4_18_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_18_1x1_increase')
             .batch_normalization(relu=False, name='conv4_18_1x1_increase_bn'))

        (self.feed('conv4_17/relu',
                   'conv4_18_1x1_increase_bn')
             .add(name='conv4_18')
             .relu(name='conv4_18/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_19_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_19_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding26')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_19_3x3')
             .batch_normalization(relu=True, name='conv4_19_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_19_1x1_increase')
             .batch_normalization(relu=False, name='conv4_19_1x1_increase_bn'))

        (self.feed('conv4_18/relu',
                   'conv4_19_1x1_increase_bn')
             .add(name='conv4_19')
             .relu(name='conv4_19/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_20_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_20_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding27')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_20_3x3')
             .batch_normalization(relu=True, name='conv4_20_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_20_1x1_increase')
             .batch_normalization(relu=False, name='conv4_20_1x1_increase_bn'))

        (self.feed('conv4_19/relu',
                   'conv4_20_1x1_increase_bn')
             .add(name='conv4_20')
             .relu(name='conv4_20/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_21_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_21_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding28')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_21_3x3')
             .batch_normalization(relu=True, name='conv4_21_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_21_1x1_increase')
             .batch_normalization(relu=False, name='conv4_21_1x1_increase_bn'))

        (self.feed('conv4_20/relu',
                   'conv4_21_1x1_increase_bn')
             .add(name='conv4_21')
             .relu(name='conv4_21/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_22_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_22_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding29')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_22_3x3')
             .batch_normalization(relu=True, name='conv4_22_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_22_1x1_increase')
             .batch_normalization(relu=False, name='conv4_22_1x1_increase_bn'))

        (self.feed('conv4_21/relu',
                   'conv4_22_1x1_increase_bn')
             .add(name='conv4_22')
             .relu(name='conv4_22/relu')
             .conv(1, 1, 256/scale_channels, 1, 1, biased=False, relu=False, name='conv4_23_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_23_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding30')
             .atrous_conv(3, 3, 256/scale_channels, 2, biased=False, relu=False, name='conv4_23_3x3')
             .batch_normalization(relu=True, name='conv4_23_3x3_bn')
             .conv(1, 1, 1024/scale_channels, 1, 1, biased=False, relu=False, name='conv4_23_1x1_increase')
             .batch_normalization(relu=False, name='conv4_23_1x1_increase_bn'))

        (self.feed('conv4_22/relu',
                   'conv4_23_1x1_increase_bn')
             .add(name='conv4_23')
             .relu(name='conv4_23/relu')
             .conv(1, 1, 2048/scale_channels, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj')
             .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn'))

        ## Start of 5th block
        (self.feed('conv4_23/relu')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding31')
             .atrous_conv(3, 3, 512/scale_channels, 4, biased=False, relu=False, name='conv5_1_3x3')
             .batch_normalization(relu=True, name='conv5_1_3x3_bn')
             .conv(1, 1, 2048/scale_channels, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
             .batch_normalization(relu=False, name='conv5_1_1x1_increase_bn'))

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding32')
             .atrous_conv(3, 3, 512/scale_channels, 4, biased=False, relu=False, name='conv5_2_3x3')
             .batch_normalization(relu=True, name='conv5_2_3x3_bn')
             .conv(1, 1, 2048/scale_channels, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
             .batch_normalization(relu=False, name='conv5_2_1x1_increase_bn'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase_bn')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding33')
             .atrous_conv(3, 3, 512/scale_channels, 4, biased=False, relu=False, name='conv5_3_3x3')
             .batch_normalization(relu=True, name='conv5_3_3x3_bn')
             .conv(1, 1, 2048/scale_channels, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
             .batch_normalization(relu=False, name='conv5_3_1x1_increase_bn'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase_bn')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        conv5_3 = self.layers['conv5_3/relu']
        shape = tf.shape(conv5_3)[1:3]

        (self.feed('conv5_3/relu')
             .avg_pool(90, 90, 90, 90, name='conv5_3_pool1')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv5_3_pool1_conv')
             .batch_normalization(relu=True, name='conv5_3_pool1_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(45, 45, 45, 45, name='conv5_3_pool2')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv5_3_pool2_conv')
             .batch_normalization(relu=True, name='conv5_3_pool2_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(30, 30, 30, 30, name='conv5_3_pool3')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv5_3_pool3_conv')
             .batch_normalization(relu=True, name='conv5_3_pool3_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(15, 15, 15, 15, name='conv5_3_pool6')
             .conv(1, 1, 512/scale_channels, 1, 1, biased=False, relu=False, name='conv5_3_pool6_conv')
             .batch_normalization(relu=True, name='conv5_3_pool6_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .concat(axis=-1, name='conv5_3_concat')
             .conv(3, 3, 512/scale_channels, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_4')
             .batch_normalization(relu=True, name='conv5_4_bn')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6'))

class PSPNet50(Network):
    def setup(self, is_training, num_classes):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
             .conv(3, 3, 64, 2, 2, biased=False, relu=False, padding='SAME', name='conv1_1_3x3_s2')
             .batch_normalization(relu=False, name='conv1_1_3x3_s2_bn')
             .relu(name='conv1_1_3x3_s2_bn_relu')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_2_3x3')
             .batch_normalization(relu=True, name='conv1_2_3x3_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_3_3x3')
             .batch_normalization(relu=True, name='conv1_3_3x3_bn')
             .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj')
             .batch_normalization(relu=False, name='conv2_1_1x1_proj_bn'))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_1_3x3')
             .batch_normalization(relu=True, name='conv2_1_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase')
             .batch_normalization(relu=False, name='conv2_1_1x1_increase_bn'))

        (self.feed('conv2_1_1x1_proj_bn',
                   'conv2_1_1x1_increase_bn')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_2_3x3')
             .batch_normalization(relu=True, name='conv2_2_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')
             .batch_normalization(relu=False, name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase_bn')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_3_3x3')
             .batch_normalization(relu=True, name='conv2_3_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')
             .batch_normalization(relu=False, name='conv2_3_1x1_increase_bn'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase_bn')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj')
             .batch_normalization(relu=False, name='conv3_1_1x1_proj_bn'))

        (self.feed('conv2_3/relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_1_3x3')
             .batch_normalization(relu=True, name='conv3_1_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')
             .batch_normalization(relu=False, name='conv3_1_1x1_increase_bn'))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_1x1_increase_bn')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_2_3x3')
             .batch_normalization(relu=True, name='conv3_2_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')
             .batch_normalization(relu=False, name='conv3_2_1x1_increase_bn'))

        (self.feed('conv3_1/relu',
                   'conv3_2_1x1_increase_bn')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_3_3x3')
             .batch_normalization(relu=True, name='conv3_3_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')
             .batch_normalization(relu=False, name='conv3_3_1x1_increase_bn'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase_bn')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_4_3x3')
             .batch_normalization(relu=True, name='conv3_4_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')
             .batch_normalization(relu=False, name='conv3_4_1x1_increase_bn'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase_bn')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj')
             .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn'))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_1_3x3')
             .batch_normalization(relu=True, name='conv4_1_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
             .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_2_3x3')
             .batch_normalization(relu=True, name='conv4_2_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
             .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase_bn')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_3_3x3')
             .batch_normalization(relu=True, name='conv4_3_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')
             .batch_normalization(relu=False, name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase_bn')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_4_3x3')
             .batch_normalization(relu=True, name='conv4_4_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')
             .batch_normalization(relu=False, name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase_bn')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_5_3x3')
             .batch_normalization(relu=True, name='conv4_5_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')
             .batch_normalization(relu=False, name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase_bn')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_6_3x3')
             .batch_normalization(relu=True, name='conv4_6_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')
             .batch_normalization(relu=False, name='conv4_6_1x1_increase_bn'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase_bn')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj')
             .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn'))

        (self.feed('conv4_6/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding31')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_1_3x3')
             .batch_normalization(relu=True, name='conv5_1_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
             .batch_normalization(relu=False, name='conv5_1_1x1_increase_bn'))

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding32')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_2_3x3')
             .batch_normalization(relu=True, name='conv5_2_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
             .batch_normalization(relu=False, name='conv5_2_1x1_increase_bn'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase_bn')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding33')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_3_3x3')
             .batch_normalization(relu=True, name='conv5_3_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
             .batch_normalization(relu=False, name='conv5_3_1x1_increase_bn'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase_bn')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        conv5_3 = self.layers['conv5_3/relu']
        shape = tf.shape(conv5_3)[1:3]

        (self.feed('conv5_3/relu')
             .avg_pool(60, 60, 60, 60, name='conv5_3_pool1')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool1_conv')
             .batch_normalization(relu=True, name='conv5_3_pool1_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(30, 30, 30, 30, name='conv5_3_pool2')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool2_conv')
             .batch_normalization(relu=True, name='conv5_3_pool2_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(20, 20, 20, 20, name='conv5_3_pool3')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool3_conv')
             .batch_normalization(relu=True, name='conv5_3_pool3_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(10, 10, 10, 10, name='conv5_3_pool6')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool6_conv')
             .batch_normalization(relu=True, name='conv5_3_pool6_conv_bn')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .concat(axis=-1, name='conv5_3_concat')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_4')
             .batch_normalization(relu=True, name='conv5_4_bn')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6'))
