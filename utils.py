import torch
import torchvision.transforms as transforms
from PIL import Image

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(0)
    print("Example 1 correct: Dog on a beach by the ocean")
    print("Example 1 output:" +
          " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
          )

    test_img2 = transform(Image.open("test_examples/child.jpg").convert("RGB")).unsqueeze(0)
    print("Example 2 correct: Child holding red frisbee outdoors")
    print("Example 2 output:" +
          " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
          )

    test_img3 = transform(Image.open("test_examples/bus.png").convert("RGB")).unsqueeze(0)
    print("Example 3 correct: Bus driving by parked cars")
    print("Example 3 output:" +
          " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
          )

    model.train()

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("= saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("= loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step