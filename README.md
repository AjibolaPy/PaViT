The Pathway Vision Transformer (PaViT), developed by Ajibola Emmanuel Oluwaseun, draws inspiration from Google's PaLM (Pathways Language Model) and aims to demonstrate the potential of using few-shot learning techniques in image recognition tasks. Through rigorous experimentation on a 4GB RAM CPU, PaViT was trained on a dataset of 2000 Kaggle images of 4 classes, resulting in a 74% accuracy with 4 heads and an 87% accuracy with 10 self-attention heads. These results highlight the model's impressive performance and fast training speed on a CPU, despite being trained on a relatively small dataset. The author believes that PaViT has the potential to outperform existing Vision Transformer models and is eager to see it continue to evolve through the contributions of developers and other contributors."



How to use:
'\n'
#On inference
```ruby
import PaViT 
import cv2
from tensorflow.keras.models import *
image=cv2.imread(image) #Load image
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert image to RGB
imahe=cv2.resize(224, 224) #Deafult image size
model=load_model('trainined_weight.h5') #Load weight
prediction=model.predict(image) #run inference
prediction=np.argmax(predication, axis=-1) #Show highest probability class
```

```ruby
#On training
model=PaViT.PaviT(out=15, activation='sigmoid') #output dense_layer is 15, output activation 15
model.load_weights('trained_weight.h5)
model.compile(...)
model.fit(...)
```


