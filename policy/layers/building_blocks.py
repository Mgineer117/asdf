from typing import Optional, Union

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[list[int], tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU(),
        initialization: str = "default",
        dropout_rate: Optional[float] = None,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + hidden_dims
        model = []

        # Derive gain from actual activation type (isinstance avoids false-negative == on modules)
        if isinstance(activation, nn.ReLU):
            gain = nn.init.calculate_gain("relu")       # sqrt(2) ≈ 1.414
        elif isinstance(activation, nn.LeakyReLU):
            gain = nn.init.calculate_gain("leaky_relu")
        elif isinstance(activation, nn.Tanh):
            gain = nn.init.calculate_gain("tanh")       # 5/3 ≈ 1.667
        elif isinstance(activation, nn.Sigmoid):
            gain = nn.init.calculate_gain("sigmoid")
        else:
            gain = 1.0

        # Initialize hidden layers
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            linear_layer = nn.Linear(in_dim, out_dim)
            if initialization == "default":
                nn.init.xavier_uniform_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.01)

            elif initialization == "actor":
                nn.init.orthogonal_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "critic":
                nn.init.orthogonal_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.0)

            model += (
                [linear_layer, activation] if activation is not None else [linear_layer]
            )

            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]

        # Initialize output layer
        if output_dim is not None:
            linear_layer = nn.Linear(hidden_dims[-1], output_dim)
            if initialization == "default":
                nn.init.xavier_uniform_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "actor":
                nn.init.orthogonal_(linear_layer.weight, gain=0.01)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "critic":
                nn.init.orthogonal_(linear_layer.weight, gain=1)
                linear_layer.bias.data.fill_(0.0)

            model += [linear_layer]
            self.output_dim = output_dim

        self.model = nn.Sequential(*model).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CNN(nn.Module):
    """
    Classic CNN architecture from the Nature DQN paper.
    Used for processing pixel-based observations.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        features_dim: int = 512,
        initialization: str = "actor",
        device=torch.device("cpu"),
    ):
        super().__init__()
        # input_shape is expected to be (Channels, Height, Width)
        n_input_channels = input_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass with a dummy tensor
        with torch.no_grad():
            dummy_tensor = torch.zeros(1, *input_shape)
            self.n_flatten = self.cnn(dummy_tensor).shape[1]

        # Made the linear head deeper here
        self.linear = nn.Sequential(
            nn.Linear(self.n_flatten, features_dim),
            nn.ReLU(),
        )
        self.output_dim = features_dim

        # Apply orthogonal initialization standard for PPO
        if initialization in ["actor", "critic"]:
            gain = nn.init.calculate_gain("relu")
            for module in self.cnn.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=gain)
                    module.bias.data.fill_(0.0)
            for module in self.linear.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=gain)
                    module.bias.data.fill_(0.0)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PPO requires inputs in [0, 1]. Ensure you divide image arrays by 255.0 before this step
        x = self.cnn(x)
        x = self.linear(x)
        return x
