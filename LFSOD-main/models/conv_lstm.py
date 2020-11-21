import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels, self.kernel_size, 1,
                              self.padding)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, h, c):

        combined = torch.cat((input, h), dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, A.size()[1] // self.num_features, dim=1)
        i = torch.sigmoid(ai)    #input gate
        f = torch.sigmoid(af)    #forget gate
        o = torch.sigmoid(ao)    #output
        g = torch.tanh(ag)       #update_Cell

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c, o

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    def __init__(self, input_size, output_channels, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.output_channels = output_channels
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)


        self.conv_fcn2 = nn.Conv2d(64*12, 64, 3, padding=1)         # 13 = 1+N (N=12)

        self.conv_h = nn.Conv2d(64, 64, 3, padding=1)
        self.pool_avg = nn.AvgPool2d(self.input_size, stride=2, ceil_mode=True)
        self.conv_c = nn.Conv2d(64, 12, 1, padding=0)

        self.conv_pre = nn.Conv2d(192, self.output_channels, 1, padding=0)



        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        internal_state = []

        fcn2_output = input # 12 *(64*64) * 64 channels
        input = torch.cat(torch.chunk(input, 12, dim=0), dim=1)


        for step in range(3):
            x = input
            if step == 0:
                basize, _, height, width = input.size()
                (h_step, c) = ConvLSTMCell.init_hidden(basize, self.hidden_channels[self.num_layers-1], (height, width))


            fcn2 = self.conv_fcn2(x)

            h_c = self.conv_h(h_step)

            fcn2_h_cat = fcn2 + h_c
            fcn2_h_cat = self.pool_avg(fcn2_h_cat)
            fcn2_h_cat = self.conv_c(fcn2_h_cat)

            # Attention Module
            fcn2_h_cat = torch.mul(F.softmax(fcn2_h_cat,dim=1), 12)
            Att = fcn2_h_cat

            basize, dime, h, w = fcn2_h_cat.size()
            fcn2_h_cat = fcn2_h_cat.view(1, basize, dime, h, w).transpose(0, 1).transpose(1, 2)
            fcn2_h_cat = torch.cat(torch.chunk(fcn2_h_cat, basize, dim=0), dim=1).view(basize*dime, 1, 1, 1)

            fcn2_h_cat = torch.mul(fcn2_output, fcn2_h_cat).view(1, basize*dime, 64, self.input_size, self.input_size)
            fcn2_h_cat = torch.cat(torch.chunk(fcn2_h_cat, basize, dim=1), dim=0)
            fcn2_h_cat = torch.sum(fcn2_h_cat, 1, keepdim=False)#.squeeze()

            x = fcn2_h_cat
            if step < self.step-1:
                for i in range(self.num_layers):
                    # all cells are initialized in the first step
                    if step == 0:
                        bsize, _, height, width = x.size()
                        (h, c) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[i], (height, width))
                        internal_state.append((h, c))
                    # do forward
                    name = 'cell{}'.format(i)
                    (h, c) = internal_state[i]

                    x, new_c, new_o = getattr(self, name)(x, h, c) # ConvLSTMCell forward
                    internal_state[i] = (x, new_c)
                h_step = x
                # only record effective steps
                #if step in self.effective_step:

                if step == 0:
                    outputs_o = new_o
                else:
                    outputs_o = torch.cat((outputs_o, new_o), dim=1)

        # outputs_o = torch.cat([outputs_o, new_o], dim=1)
        # outputs_o = torch.cat([outputs_o, outputs_o, outputs_o], dim=1)
        outputs = self.conv_pre(outputs_o)

       # output = F.upsample(outputs, scale_factor=4, mode='bilinear')

        return outputs
