import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Image loader that preserves original dimensions
class ImageLoader:
    def __init__(self):
        # No fixed size - will preserve original dimensions
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load(self, path):
        img = Image.open(path).convert('RGB')
        orig_size = img.size  # (width, height)
        img_tensor = self.transform(img).unsqueeze(0).to(device)
        return img_tensor, orig_size

    def denormalize(self, tensor, target_size=None):
        tensor = tensor.detach().cpu().squeeze(0).permute(1, 2, 0)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        tensor = tensor * std + mean
        if target_size:
            tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)
            tensor = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
            tensor = tensor.squeeze(0).permute(1, 2, 0)
        return tensor.clamp(0, 1)


# High quality style transfer with YUV color preservation
class HighQualityStyleTransfer:
    def __init__(self):
        self.vgg = models.vgg19(pretrained=True).features.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.style_layers = [0, 5, 10, 14, 19, 21, 23]
        self.content_layer = 28

        self.style_weight = 1e8
        self.content_weight = 1e0
        self.tv_weight = 1e-6
        self.noise_factor = 0.25
        self.max_size = 512

    def resize_to_max(self, img_tensor):
        _, _, h, w = img_tensor.shape
        if max(h, w) > self.max_size:
            if h > w:
                new_h = self.max_size
                new_w = int(w * (self.max_size / h))
            else:
                new_w = self.max_size
                new_h = int(h * (self.max_size / w))
            return F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return img_tensor

    def get_features(self, image):
        features = {}
        x = image
        for i, layer in enumerate(self.vgg.children()):
            x = layer(x)
            if i in self.style_layers:
                features[f'style_{i}'] = x
            if i == self.content_layer:
                features['content'] = x
                break
        return features

    def gram_matrix(self, tensor):
        b, c, h, w = tensor.size()
        features = tensor.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

    def total_variation_loss(self, img):
        return torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
               torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

    # ===== ИСПРАВЛЕНИЕ: работаем с денормализованными тензорами NCHW в диапазоне [0,1] =====

    def preserve_color_luminance(self, content_img, styled_img, color_strength=0.0):
        """
        Preserve original content colors using YUV color space.

        color_strength: float 0 to 1
        """
        content_img = content_img.detach().cpu()
        styled_img = styled_img.detach().cpu()

        content_yuv = self.rgb_to_yuv(content_img)
        styled_yuv = self.rgb_to_yuv(styled_img)

        # Интерполируем между оригинальными и стилизованными цветами
        # color_strength=0: берём U,V из контента
        # color_strength=1: берём U,V из стиля
        blended_uv = (1 - color_strength) * content_yuv[:, 1:3, :, :] + \
                     color_strength * styled_yuv[:, 1:3, :, :]

        styled_yuv[:, 1:3, :, :] = blended_uv

        result = self.yuv_to_rgb(styled_yuv)
        return result.to(device)

    def rgb_to_yuv(self, rgb):
        transform = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]
        ]).float().to(rgb.device)
        b, c, h, w = rgb.size()
        rgb_reshaped = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
        yuv = torch.matmul(rgb_reshaped, transform.t())
        return yuv.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    def yuv_to_rgb(self, yuv):
        transform = torch.tensor([
            [1.0, 0.0, 1.13983],
            [1.0, -0.39465, -0.58060],
            [1.0, 2.03211, 0.0]
        ]).float().to(yuv.device)
        b, c, h, w = yuv.size()
        yuv_reshaped = yuv.permute(0, 2, 3, 1).reshape(-1, 3)
        rgb = torch.matmul(yuv_reshaped, transform.t())
        return rgb.clamp(0, 1).reshape(b, h, w, 3).permute(0, 3, 1, 2)

    def transfer(self, content_path, style_path, steps=200):
        loader = ImageLoader()
        content_img, content_size = loader.load(content_path)
        style_img, _ = loader.load(style_path)

        orig_h, orig_w = content_size[1], content_size[0]

        # Resize для обработки (сохраняем пропорции)
        content_resized = self.resize_to_max(content_img)
        style_resized = self.resize_to_max(style_img)

        # Инициализация выхода
        noise = torch.randn_like(content_resized) * self.noise_factor
        output = (content_resized * (1 - self.noise_factor) + noise).requires_grad_(True)

        with torch.no_grad():
            style_features = self.get_features(style_resized)
            content_features = self.get_features(content_resized)
            style_targets = {k: self.gram_matrix(v) for k, v in style_features.items()}
            content_target = content_features['content']

        optimizer = optim.Adam([output], lr=0.05)

        start_time = time.time()
        for i in range(steps):
            output_features = self.get_features(output)

            style_loss = 0
            for k, v in output_features.items():
                if k.startswith('style_'):
                    style_loss += F.mse_loss(self.gram_matrix(v), style_targets[k])
            style_loss *= self.style_weight

            content_loss = F.mse_loss(output_features['content'], content_target) * self.content_weight
            tv_loss = self.total_variation_loss(output) * self.tv_weight

            total_loss = style_loss + content_loss + tv_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                output.clamp_(0, 1)

            if i % 50 == 0:
                print(f"Step {i}, Loss: {total_loss.item():.2f}")

        print(f"Style transfer completed in {time.time() - start_time:.2f} seconds")

        # ===== ИСПРАВЛЕНИЕ: сначала денормализуем, потом YUV, потом resize =====
        # Денормализуем output и content в диапазон [0,1], формат NCHW
        output_denorm = loader.denormalize(output)  # HWC [0,1]
        content_denorm = loader.denormalize(content_resized)  # HWC [0,1]

        # Переводим в NCHW для YUV-функций
        output_nchw = output_denorm.permute(2, 0, 1).unsqueeze(0).to(device)
        content_nchw = content_denorm.permute(2, 0, 1).unsqueeze(0).to(device)

        # Применяем YUV-сохранение цветов к реальным RGB-значениям
        result = self.preserve_color_luminance(content_nchw, output_nchw, 0.05)

        # Resize обратно к оригинальному размеру (сохраняем пропорции!)
        result = F.interpolate(result, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        result = result.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1)

        # Лёгкое усиление контраста (опционально)
        mean_val = result.mean()
        result = (result - mean_val) * 1.75 + mean_val  # коэффициент 1.15 — настройте под себя
        result = result.clamp(0, 1)

        return result


def tensor_to_image(tensor):
    tensor = tensor.numpy() * 255
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)


# Main execution
if __name__ == "__main__":
    content_path = 'images/HSE.jpg'
    style_path = 'images/9val.jpg'

    content_img = Image.open(content_path)
    style_img = Image.open(style_path)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(content_img)
    plt.title('Content Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(style_img)
    plt.title('Style Image')
    plt.axis('off')

    print("\n--- High quality style transfer with YUV color preservation ---")
    transfer = HighQualityStyleTransfer()
    result = transfer.transfer(content_path, style_path, steps=1000)

    result_img = tensor_to_image(result)
    plt.subplot(1, 3, 3)
    plt.imshow(result_img)
    plt.title('Styled Result')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    result_img.save('styled_result.jpg')
    print("Result saved as 'styled_result.jpg'")
