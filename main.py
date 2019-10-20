import numpy as np
import torch

def basics():

    # Given a fully connected network
    #
    #     (1a)   (2a)   (3a)   (4a)
    #         \\  |  \ / |   //
    #            (1b)   (2b)
    
    # The first layer values
    
    layer_1 = np.array([
        [-0.5, 0.9, 0.1, -0.7]
        ])
    
    # The weight between all node combinations
    # (1a 1b) (1a 2b) (2a 1b) ...
    
    weight_layer_1_x_layer_2 = np.array([
        [-0.3, 0.9],
        [ 0.2, 0.1],
        [-0.4, 0.6],
        [-0.7, 0.2]
        ])
    
    # Multiplying the matrices gives the
    # second layer input values
    
    layer_2_input = layer_1.dot(weight_layer_1_x_layer_2)
    print(layer_2_input)
    
    #           [[0.78 -0.44]]
    #
    # Compute the second layer output values
    # using ReLU: 0 if negative, identity if positive
    
    layer_2_output = np.maximum(layer_2_input, 0) 
    print(layer_2_output)
    
    #            [[0.78 0]]
    #
    # The expected output
    
    expected_layer_2_output = np.array([
        [0.1, -0.7]
        ])
    
    # Calculate the difference
    
    loss = (layer_2_output - expected_layer_2_output).sum()
    print(loss)

    #              1.38
    #    
    # Update the weights based on that information
    # This is beyond my understanding, but it
    # will be handled by pytorch

def autograd():

    # Multiply on the GPU

    device = torch.device("cuda:0")

    # Same values as above

    layer_1 = torch.tensor([
        [-0.5, 0.9, 0.1, -0.7]
        ], device=device, dtype=torch.float)

    # We needed help with gradients
    # Autograd to the rescue

    weight_layer_1_x_layer_2 = torch.tensor([
        [-0.3, 0.9],
        [ 0.2, 0.1],
        [-0.4, 0.6],
        [-0.7, 0.2]
        ], device=device, dtype=torch.float, requires_grad=True)

    # Do ourselves some learning

    for t in range(6000):

        # Same computation as above

        layer_2_output = layer_1.mm(weight_layer_1_x_layer_2).clamp(min=0)
    
        # Same expected output

        expected_layer_2_output = torch.tensor([
            [0.1, -0.7]
            ], device=device, dtype=torch.float)
    
        # Calculate the difference

        loss = (layer_2_output - expected_layer_2_output).sum()
        if t % 1000 == 999:
            print(loss.item())
    
        # Update the weights based on that information

        loss.backward()
    
        with torch.no_grad():
            learning_rate = 0.0001
            weight_layer_1_x_layer_2 -= learning_rate * weight_layer_1_x_layer_2.grad
            weight_layer_1_x_layer_2.grad.zero_()
    
    # We learned better weights

    print(weight_layer_1_x_layer_2);

basics()

autograd()
