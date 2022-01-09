import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv


def encoder_block(input, in_c, out_c):
    x1 = double_conv(in_c, out_c)(input)
    x2 = nn.MaxPool2d(kernel_size=2, stride=2)(x1)

    return x1, x2


def decoder_block(input, skip_features, in_c, out_c):
    x = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=2, stride=2)(input)
    y = crop_img(skip_features, x)
    x = double_conv(in_c, out_c)(torch.cat([x, y], 1))

    return x

def crop_img(tensor, tensor_target):
    target_size = tensor_target.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:target_size+delta]

def model_unet(input):
    down_conv_1, max_pooling_1 = encoder_block(input, 1, 64)
    down_conv_2, max_pooling_2 = encoder_block(max_pooling_1, 64, 128)
    down_conv_3, max_pooling_3 = encoder_block(max_pooling_2, 128, 256)
    down_conv_4, max_pooling_4 = encoder_block(max_pooling_3, 256, 512)
    down_conv_5 = double_conv(512, 1024)(max_pooling_4)

    up_conv_1 = decoder_block(down_conv_5, down_conv_4, 1024, 512)
    up_conv_2 = decoder_block(up_conv_1, down_conv_3, 512, 256)
    up_conv_3 = decoder_block(up_conv_2, down_conv_2, 256, 128)
    up_conv_4 = decoder_block(up_conv_3, down_conv_1, 128, 64)

    out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)(up_conv_4)

    print(out.size())


if __name__ == "__main__":
    input = torch.rand((1, 1, 572, 572))
    model = model_unet(input)
    # model.summary()
