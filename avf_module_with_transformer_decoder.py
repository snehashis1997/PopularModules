import torch
import torch.nn as nn
import torch.nn.functional as F

class AVFModule(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8, num_layers=6):
        super(AVFModule, self).__init__()

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Learnable SMPL Query Token
        self.smpl_query_token = nn.Parameter(torch.randn(1, 1, feature_dim))  # (batch_size, 1 token, feature_dim)

        # Final MLP to predict pose and shape parameters
        self.mlp_pose = nn.Linear(feature_dim, 72)  # 72D pose vector (SMPL model)
        self.mlp_shape = nn.Linear(feature_dim, 10) # 10D shape vector (SMPL model)

    def forward(self, feature_tokens):
        """
        feature_tokens: (batch_size, num_views, feature_dim) -> Extracted 2D feature tokens
        """
        batch_size = feature_tokens.shape[0]

        # Expand SMPL query token for batch
        smpl_query_token = self.smpl_query_token.expand(batch_size, -1, -1)

        # Transformer Decoder: Multi-view fusion
        smpl_query_output = self.transformer_decoder(smpl_query_token, feature_tokens)

        # Predict pose (θ) and shape (β)
        theta = self.mlp_pose(smpl_query_output.squeeze(1))  # (batch_size, 72)
        beta = self.mlp_shape(smpl_query_output.squeeze(1))  # (batch_size, 10)

        return theta, beta
