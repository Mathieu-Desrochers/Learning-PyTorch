import numpy
import torch

def matrices():

    # Given a fully connected network
    #
    #     (1a)   (2a)   (3a)   (4a)
    #         \\  |  \ / |   //
    #            (1b)   (2b)
    
    # The values of a layer can be
    # expressed as a matrix
    
    layer_1 = numpy.array([
        [-0.5, 0.9, 0.1, -0.7]
        ])
    
    # So can the weights between the layers
    # (1a 1b) (1a 2b) (2a 1b) ...
    
    weights = numpy.array([
        [-0.3, 0.9],
        [ 0.2, 0.1],
        [-0.4, 0.6],
        [-0.7, 0.2]
        ])

    # Multiplying the matrices gives
    # the next layer values
    
    layer_2 = layer_1.dot(weights)
    print(layer_2)

    #    [[ 0.78 -0.44]]

def tensors():

    # Tensors are multiplied on the GPU

    device = torch.device("cuda:0")

    layer_1 = torch.tensor([
        [-0.5, 0.9, 0.1, -0.7]
        ], device=device, dtype=torch.float)

    # Tensors can calculate their gradient automatically

    weights = torch.tensor([
        [-0.3, 0.9],
        [ 0.2, 0.1],
        [-0.4, 0.6],
        [-0.7, 0.2]
        ], device=device, dtype=torch.float, requires_grad=True)

    # Calculate a numeric loss based on the difference
    # between the actual and the expected output

    layer_2 = layer_1.mm(weights)

    layer_2_expected = torch.tensor([
        [0.1, -0.7]
        ], device=device, dtype=torch.float)

    loss = (layer_2 - layer_2_expected).sum()

    # Have the gradients be calculated
    # based on that loss

    loss.backward()

    # Apply the gradients

    with torch.no_grad():
        learning_rate = 0.0001
        weights -= learning_rate * weights.grad
        weights.grad.zero_()
    
    # We get better weights

    print(weights);

    # tensor([[-0.3000,  0.9000],
    #         [ 0.1999,  0.0999],
    #         [-0.4000,  0.6000],
    #         [-0.6999,  0.2001]], device='cuda:0', requires_grad=True)

matrices()
tensors()
