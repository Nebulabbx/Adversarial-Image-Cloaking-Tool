
"""
  python cloak_image.py --input ./photos/alice.jpg --output ./cloaked/alice.png
  python cloak_image.py --indir ./photos --outdir ./cloaked --eps 8 --steps 400
"""
import os
import argparse
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms, models

# Utility / loss functions
def load_image_as_tensor(path, device, 
max_size=None):
    img = Image.open(path).convert("RGB")
    if max_size is not None:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    to_tensor = transforms.ToTensor()  
    return to_tensor(img).unsqueeze(0).to(device), img

def save_tensor_as_image(x_tensor, out_path):
    x = x_tensor.detach().cpu().clamp(0, 1).squeeze(0)
    to_pil = transforms.ToPILImage()
    pil = to_pil(x)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pil.save(out_path)

def tv_loss(x):
    # total variation loss encourages smoothness in perturbation
    batch_size = x.shape[0]
    h_variation = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    w_variation = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return (h_variation + w_variation)


class FeatureExtractor(torch.nn.Module):
    """
    Use a pretrained ResNet50 up to the avgpool layer as a feature extractor.
    The output is a flattened vector per image.
    """
    def __init__(self, device):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        # remove fc layer
        modules = list(self.model.children())[:-1]  # up to avgpool
        self.backbone = torch.nn.Sequential(*modules).to(device).eval()
        # freeze params
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x normalized 
        feat = self.backbone(x) 
        feat = feat.view(feat.size(0), -1)  
        return feat

# Transformations for "robustness"

class RobustTransforms:
    """
    Collection of randomized differentiable transforms applied to the image
    during optimization to encourage perturbation to survive resizing / crop / blur.
    Not all transforms are strictly differentiable (PIL operations are not),
    but using several augmented variants in expectation improves robustness.
    """
    def __init__(self, device):
        self.device = device
        # Normalization parameters for ImageNet (ResNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def apply(self, x):
        # Random resized crop (differentiable via F.interpolate)
        B, C, H, W = x.shape
        # random scale between 0.9 and 1.0
        scale = 0.9 + 0.1 * torch.rand(B, device=self.device)
        out = []
        for i in range(B):
            s = scale[i].item()
            new_h = max(2, int(H * s))
            new_w = max(2, int(W * s))
            xi = x[i:i+1]
            # center crop to new size and resize back (approx differentiable)
            start_h = (H - new_h) 
            start_w = (W - new_w)//2
            cropped = xi[:, :, start_h:start_h+new_h, start_w:start_w+new_w]
            resized = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
            # small Gaussian blur (implemented as conv) with prob 0.5
            if torch.rand(1).item() < 0.3:
                # simple 3x3 kernel blur
                kernel = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]], device=self.device)
                kernel = kernel / kernel.sum()
                kernel = kernel.view(1,1,3,3).repeat(3,1,1,1)
                pad = (1,1,1,1)
                blurred = F.conv2d(F.pad(resized, pad, mode='reflect'), kernel, groups=3)
                resized = blurred
            # color jitter approx: multiply by small factor
            if torch.rand(1).item() < 0.3:
                factor = 0.98 + 0.04 * torch.rand(1).item()
                resized = resized * factor
            out.append(resized)
        out = torch.cat(out, dim=0)
        return self.normalize(out)

# Optimization routine

def craft_perturbation(image_tensor, extractor, device,
                       eps=8/255.0, steps=300, step_size=1.0/255.0,
                       tv_weight=1e-3, l2_weight=1e-4, save_intermediate=False):
    """
    image_tensor: (1,3,H,W) in [0,1]
    extractor: feature extractor model
    """
    x_orig = image_tensor.to(device)
    transforms_obj = RobustTransforms(device)

    # store the original reference features (normalized)
    with torch.no_grad():
        feat_orig = extractor(transforms_obj.normalize(x_orig)).detach()

    # delta is the learned perturbation (initialized zero)
    delta = torch.zeros_like(x_orig, device=device, requires_grad=True)

    # use Adam optimizer on delta
    optimizer = torch.optim.Adam([delta], lr=step_size)

    for i in range(steps):
        optimizer.zero_grad()
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)

        # apply randomized transforms and compute average feature
        # do a small ensemble of transforms per step for stability
        n_ensembles = 3
        feats = []
        for _ in range(n_ensembles):
            x_norm = transforms_obj.apply(x_adv)
            feats.append(extractor(x_norm))
        feat_adv = torch.stack(feats, dim=0).mean(dim=0)

        cos = F.cosine_similarity(feat_adv, feat_orig, dim=1)
        feat_loss = torch.mean(cos)
        # So minimize feat_loss 

        # regularizers: TV and L2 on delta
        tv = tv_loss(delta)
        l2 = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).mean()

        loss = feat_loss + tv_weight * tv + l2_weight * l2

        loss.backward()
        optimizer.step()

        # Project delta to L_inf ball
        with torch.no_grad():
            delta.clamp_(-eps, eps)
            # ensure x_orig + delta in [0,1]
            delta.data = torch.clamp(x_orig + delta.data, 0.0, 1.0) - x_orig

        if (i % 50 == 0) or (i == steps - 1):
            # progress logging
            with torch.no_grad():
                curr_cos = F.cosine_similarity(extractor(transforms_obj.normalize(torch.clamp(x_orig + delta,0,1))), feat_orig, dim=1).item()
                # current L_inf
                linf = delta.detach().abs().max().item()
            tqdm.write(f"[iter {i}/{steps}] loss={loss.item():.6f} cos={curr_cos:.4f} linf={linf:.6f}")

    adv = torch.clamp(x_orig + delta.detach(), 0.0, 1.0)
    return adv

# CLI / batch processing

def process_file(infile, outfile, extractor, device, args):
    x, pil_img = load_image_as_tensor(infile, device, max_size=args.max_size)
    adv = craft_perturbation(x, extractor, device,
                             eps=args.eps/255.0,
                             steps=args.steps,
                             step_size=args.step_size/255.0,
                             tv_weight=args.tv_weight,
                             l2_weight=args.l2_weight)
    save_tensor_as_image(adv, outfile)

def main():
    parser = argparse.ArgumentParser(description="Per-image cloaking tool")
    parser.add_argument("--input", type=str, help="Single input image path")
    parser.add_argument("--output", type=str, help="Single output image path")
    parser.add_argument("--indir", type=str, help="Input directory (process all .jpg/.png)")
    parser.add_argument("--outdir", type=str, help="Output directory for indir mode")
    parser.add_argument("--eps", type=float, default=8.0, help="L_inf epsilon in *255 scale* (default 8)")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps (default 300)")
    parser.add_argument("--step-size", type=float, default=2.0, help="Adam lr in *255 scale* (default 2). Will be divided by 255")
    parser.add_argument("--tv-weight", type=float, default=1e-3, help="TV regularizer weight")
    parser.add_argument("--l2-weight", type=float, default=1e-4, help="L2 regularizer on delta")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--max-size", type=int, default=1024, help="Max image dimension to process")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    extractor = FeatureExtractor(device)

    if args.input and args.output:
        process_file(args.input, args.output, extractor, device, args)
        print("Saved cloaked image to:", args.output)
    elif args.indir and args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        files = [f for f in os.listdir(args.indir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fname in tqdm(files):
            infile = os.path.join(args.indir, fname)
            outfile = os.path.join(args.outdir, os.path.splitext(fname)[0] + ".png")
            process_file(infile, outfile, extractor, device, args)
        print("Batch processing done.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
