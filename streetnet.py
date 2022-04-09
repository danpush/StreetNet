import json
import matplotlib.pyplot as plt
import mss
import numpy as np
import torch
from PIL import Image, ImageTk, ImageDraw
from tkinter import Tk, Canvas, NW
from torchvision import transforms


class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = (output.cpu()).data.numpy()

    def remove(self):
        self.hook.remove()


def take_screenshot():
    with mss.mss() as sct:
        monitor = {"left": 100, "top": 130, "width": 1305, "height": 810}
        sct_img = sct.grab(monitor)
        image = Image.frombytes("RGB", sct_img.size, sct_img.rgb)

    image = image.convert("RGB")
    width, height = (29 * 20, 18 * 20)
    image = image.resize((width, height), Image.ANTIALIAS)
    new_width, new_height = (29 * 18, 18 * 18)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    image = image.crop((left, top, right, bottom))  # centre crop
    image_tensor = transforms.ToTensor()(image)
    image_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
        image_tensor
    )
    return image_tensor, image


def make_prediction():
    image_tensor, image = take_screenshot()

    data = torch.unsqueeze(
        image_tensor, dim=0
    )  # Make image tensor into a batch of size one, shape is [1,3,H,W]

    outputs = model(data)
    softmax = torch.nn.Softmax(dim=1)
    preds, _ = outputs.topk(5)
    softmax_preds, indices = softmax(outputs).topk(5)

    text_label = ""
    for i in range(5):
        text_label += f"{class_names[indices[0, i].item()]} {round(softmax_preds[0, i].item() * 100, 2)}% - {round(preds[0, i].item(), 2)}\n"

    # Heatmap
    weight_softmax_params = list(model.fc.parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    _, nc, h, w = activated_features.features.shape
    cam = weight_softmax[indices[0, 0].item()].dot(
        activated_features.features.reshape((nc, h * w))
    )
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cm = plt.get_cmap("jet")
    cam_img = cm(cam_img)
    heatmap_image = Image.fromarray((cam_img[:, :, :3] * 255).astype(np.uint8))
    heatmap_image = heatmap_image.resize(image.size, Image.ANTIALIAS)

    # Overlay heatmap on top of original image
    mask_im = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask_im)
    draw.rectangle((0, 0, image.size[0], image.size[1]), fill=100)
    image.paste(heatmap_image, (0, 0), mask_im)

    return text_label, image.copy()


def update_window():
    global preview_image
    c_text, c_img = make_prediction()
    canvas.delete("all")
    preview_image = ImageTk.PhotoImage(c_img.resize((58 * 5, 36 * 5)))

    canvas.create_text(9, 9, anchor=NW, text=c_text)
    canvas.create_image(9, 90, anchor=NW, image=preview_image)
    root.after(500, update_window)  # reschedule event


if __name__ == "__main__":
    counties_path = "countries.json"
    model_path = "model.pt"

    class_names = json.load(open("countries.json"))
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    final_layer = model.layer4
    activated_features = SaveFeatures(final_layer)
    root = Tk()

    canvas = Canvas(root, width=307, height=280)
    preview_image = None

    root.title("StreetNet AI")
    root.resizable(False, False)
    root.geometry("307x280+1573+200")
    canvas.pack()
    canvas.create_text(50, 10, text="Loading...")
    root.wm_attributes("-topmost", 1)
    root.after(500, update_window)
    root.mainloop()