import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_layer = nn.Linear(6, 8)

        self.hidden_layer = nn.Linear (8, 8)
        self.hidden_layer_2 = nn.Linear (8, 8)

        self.output_layer = nn.Linear(8, 1)

    def forward(self, input):
        # Input and output are (batched) tensors.

        intermediate = self.input_layer(input)
        intermediate = torch.sigmoid(intermediate)

        intermediate = self.hidden_layer(intermediate)
        intermediate = torch.sigmoid(intermediate)

        intermediate = self.hidden_layer_2(intermediate)
        intermediate = torch.sigmoid(intermediate)

        intermediate = self.output_layer(intermediate)
        output = torch.sigmoid(intermediate)

        return output

def main():
    model = NeuralNetwork()

if __name__ == '__main__':
    main()
