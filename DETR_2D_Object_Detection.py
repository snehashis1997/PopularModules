import torchvision.models.detection.transformer as transformer

class DETR(nn.Module):
    def __init__(self, num_classes=91, hidden_dim=256, num_heads=8, num_layers=6):
        super(DETR, self).__init__()

        # Backbone (ResNet)
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Transformer
        self.transformer = transformer.Transformer(hidden_dim, num_heads, num_layers)

        # Object Queries
        self.object_queries = nn.Parameter(torch.randn(100, hidden_dim))  # 100 queries

        # Final Linear layers
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Linear(hidden_dim, 4)  # Bounding box (x, y, w, h)

    def forward(self, x):
        feature_maps = self.backbone(x)
        feature_maps = feature_maps.flatten(2).permute(2, 0, 1)  # Reshape for transformer

        transformer_out = self.transformer(feature_maps, self.object_queries)
        class_logits = self.class_embed(transformer_out)
        bboxes = self.bbox_embed(transformer_out)

        return class_logits, bboxes
