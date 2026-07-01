import argparse

import torch

from model import get_model


class _CfgModel:
    def __init__(self, name, num_classes, fuse):
        self.name = name
        self.model_kwargs = dict(
            pretrained=False,
            checkpoint_path='',
            ema=False,
            strict=True,
            num_classes=num_classes,
            fuse=fuse,
        )


def main():
    parser = argparse.ArgumentParser(description='Smoke-test course MRFFI MobileMamba variants.')
    parser.add_argument('--model', default='MobileMamba_S6_MRFFI')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--fuse', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model = get_model(_CfgModel(args.model, args.num_classes, args.fuse)).to(device).eval()
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        y = model(x)
    print(f'model={args.model}')
    print(f'input={tuple(x.shape)}')
    print(f'output={tuple(y.shape)}')
    assert y.shape == (args.batch_size, args.num_classes)


if __name__ == '__main__':
    main()
