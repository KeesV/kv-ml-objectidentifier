import cntk as C
import numpy as np
from PIL import Image
from IPython.core.display import display
import pickle

z = C.Function.load("vgg19/model.onnx", device=C.device.cpu(), format=C.ModelFormat.ONNX)
img = Image.open("assets/home.jpg")
display(img) #show the image

img = img.resize((224,224))
rgb_img = np.asarray(img, dtype=np.float32) - 128
bgr_img = rgb_img[..., [2,1,0]]
img_data = np.ascontiguousarray(np.rollaxis(bgr_img,2))

predictions = np.squeeze(z.eval({z.arguments[0]:[img_data]}))
top_class = np.argmax(predictions)
print(top_class)

labels_dict = pickle.load(open("imagenet1000_clsid_to_human.pkl", "rb"))
print(labels_dict[top_class])
