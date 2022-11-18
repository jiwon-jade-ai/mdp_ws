
import cv2
import torch
import numpy as np 
from torchvision import transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
MODEL_PATH = "/home/mcg1/jw/mdp_ws/Unet-diagdataset.pt"
model = torch.load(MODEL_PATH)
model.eval()

# define predict_image_mask_miou function
def predict_image_mask_miou(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    # mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        # mask = mask.unsqueeze(0)
        
        output = model(image)
        # score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)

    return masked #, score

image_path = "/home/mcg1/jw/mdp/folder2/1.png"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
image_resize = cv2.resize(image, (640, 384))
pred_mask = predict_image_mask_miou(model, image_resize)

cimage = pred_mask[248:328, 235:385] / 85
count = (cimage == 1.0).sum()
percentage = float(count / np.array(cimage).size * 100)

print(percentage)

