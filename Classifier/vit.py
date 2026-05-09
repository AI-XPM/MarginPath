import torch
import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=10, pretrained=True):
        super(ViTClassifier, self).__init__()
        
        # Load ViT-Large backbone from timm
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # Modify classification head
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def build_model(model_name='vit_base_patch16_224', num_classes=10, pretrained=False, device='cuda'):
    model = ViTClassifier(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
    return model.to(device)


# Example usage
if __name__ == "__main__":
    model = build_model(num_classes=10, pretrained=True)
    dummy_input = torch.randn(2, 3, 224, 224).to('cuda' if torch.cuda.is_available() else 'cpu')
    output = model(dummy_input)
    print("Output shape:", output.shape)
