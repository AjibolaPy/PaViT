<h1>About:</h1><br>
Pathway Vision Transformer (PaViT), developed by Ajibola Emmanuel Oluwaseun, draws inspiration from Google's PaLM (Pathways Language Model) and seeks to replicate the success of few-shot learning techniques in natural language tasks to the field of image recognition. After training PaViT on a 4GB RAM CPU using a dataset of 2000 Kaggle images of 4 classes, the model demonstrated remarkable results. With 4 self-attention heads, PaViT achieved a 74% accuracy and further improved to 87% accuracy with the addition of 10 self-attention heads and linear layers. These results not only highlight the model's fast training speed on a CPU, but also its potential to outperform existing Vision Transformer models. The author believes that with continued development and contributions from the community, PaViT has the potential to be a leading model in the field of image recognition. Used Batch normalization layer and got better performance compared to the custom layer normalisation, also used 12 self attention heads and 18 linearly stacked Dense layer to get best accuracy yet on the same dataset and had about 90% accuracy. Later trained it on a larger dataset of 15000 Kaggle plant images with 15 classes with Google Colab NVIDIA T4 Tensor Core GPUs and got 96% accuracy.."



<h1>How to use:</h1>
<br>
On inference</br>

```ruby
import PaViT 
import cv2
from tensorflow.keras.models import *
image=cv2.imread(image) #Load image
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert image to RGB
imahe=cv2.resize(224, 224) #Deafult image size
model=load_model('trained_weight.h5') #Load weight
prediction=model.predict(image) #run inference
prediction=np.argmax(prediction, axis=-1) #Show highest probability class
```
<br>On Training</br>
```ruby

model=PaViT.PaviT(out=15, activation='sigmoid') #output dense_layer is 15, output activation 15
model.load_weights('trained_weight.h5')
model.compile(...)
model.fit(...)
```


