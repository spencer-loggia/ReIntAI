from intrinsic import module
import torch


def test_einsum_solution_simple():
    states = torch.zeros(size=(2, 1, 1, 1)) + .5
    mod = module.PlasticEdges(channels=1, spatial1=4, spatial2=4,
                                             kernel_size=3, num_nodes=2, normalize_conv=False)
    mod.spatial1 = 1
    mod.spatial2 = 1
    mod.kernel_size = 1
    mod.pad = 0
    mod.unfolder = torch.nn.Unfold(kernel_size=1, padding=0)
    mod.folder = torch.nn.Fold(kernel_size=1,
                                output_size=(1, 1),
                                padding=0)
    test_conv = torch.Tensor([[1, 2], [3, 4]])
    test_conv = test_conv.reshape((2, 2, 1, 1, 1, 1, 1))
    mod.weight = torch.nn.Parameter(test_conv)
    test_out_edge = torch.ones((2, 2, 1, 1))
    mod.chan_map = torch.nn.Parameter(test_out_edge)
    out = mod(states).flatten().detach().tolist()
    print(out)
    assert False not in [out[i] == r for i, r in enumerate([2, 3])]


def test_intrinsic():
    states = torch.sigmoid(torch.normal(mean=0, std=.5, size=(4, 2, 16, 16)))
    mod = module.PlasticEdges(channels=2, spatial1=16, spatial2=16, kernel_size=4, num_nodes=4)
    out = mod(states)
    next_activ = torch.sigmoid(out)
    mod.update(next_activ)
    print('done')


if __name__=='__main__':
    test_einsum_solution_simple()
    test_intrinsic()