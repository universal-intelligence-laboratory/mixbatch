import numpy as np
from torch.nn import Dropout
import torch

class MixBatch_torch(Dropout):
    # @weak_script_method
    def forward(self, input):
        return self.mixup(input, self.p, self.training)


    def mixup(self,x, p=0.2, training=True,use_cuda=True):
        # type: (Tensor, float, bool, bool) -> Tensor
        r"""
        During training, randomly zeroes some of the elements of the input
        tensor with probability :attr:`p` using samples from a Bernoulli
        distribution.

        See :class:`~torch.nn.Dropout` for details.

        Args:
            p: probability of an element to be zeroed. Default: 0.5
            training: apply dropout if is ``True``. Default: ``True``
            inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        """
        # if p < 0. or p > 1.:
        #     raise ValueError("mixup probability has to be between 0 and 1, "
        #                     "but got {}".format(p))

        '''Returns mixed inputs, pairs of targets, and lambda'''
        if training==True:
            alpha = p
            if alpha > 0:
                self.lam = np.random.beta(alpha, alpha)
            else:
                self.lam = 1

            batch_size = x.size()[0]
            if use_cuda:
                index = torch.randperm(batch_size).cuda()
            else:
                index = torch.randperm(batch_size)

            self.lam = np.maximum(self.lam, np.ones_like(self.lam) - self.lam)
            mixed_x = self.lam * x + (1 - self.lam) * x[index, :]
            return mixed_x
        else:
            return x


def test():
    net = MixBatch_torch(p=0.1)
    for i in range(100):
        x = torch.randn(2,1,2)
        y = net(x)
        print("lam",net.lam)


if __name__ == "__main__":
    test()   



