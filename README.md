<h1>README for Pathway Vision Transformer</h1><br>


<p>PaViT is a Pathway Vision Transformer (PaViT)-based image recognition model developed by Ajibola Emmanuel Oluwaseun. The model is inspired by Google's PaLM (Pathways Language Model) and aims to demonstrate the potential of using few-shot learning techniques in image recognition tasks.</p>

<h1>Model Performance</h1>
PaViT was trained on a 4GB RAM CPU using a dataset of 15000 Kaggle images of 15 classes, achieving a remarkable 88% accuracy with 4 self-attention heads. The model's accuracy further improved to 96% when trained with 12 self-attention heads and 12 linearly stacked linear layers. These results demonstrate the model's impressive performance and fast training speed on a CPU, despite being trained on a relatively small dataset.

<h1>Usage</h1>
The model can be used for image recognition tasks by using the trained weights provided in the repository. The code can be modified to use custom datasets, and the model's performance can be further improved by adding more self-attention heads and linear layers.

<h1>Contribution</h1>
The author believes that PaViT has the potential to outperform existing Vision Transformer models and is eager to see it continue to evolve through the contributions of developers and other contributors.
<br></br>
Contributions to the project are welcome and can be made through pull requests. Developers can also report issues or suggest new features for the project.

<h1>License</h1>
This project is licensed under the MIT License.


<h1>How to use:</h1>
<br>
On inference</br>

```ruby
import PaViT 
import cv2
from tensorflow.keras.models import *
image=cv2.imread(image) #Load image
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert image to RGB
image=cv2.resize(224, 224) #Default image size
model=load_model('trained_weight.h5') #Load weight
prediction=model.predict(image) #run inference
prediction=np.argmax(prediction, axis=-1) #Show highest probability class
```
<br>On Training</br>
```ruby
model=PaViT.PaViT() 
#Output means the unit default is 15 and activation 'sigmoid'
#Output_class default is None so it uses Dense layer as the output_layer
#Ouptut_class is the output layer 
p_model=model.model(output_class=None, activation='softmax', output=15) 
p_model.summary()
p_model.compile(...)
p_model.fit(...)

```

<h1>Prediction on Test data</h1>
<img src='https://github.com/AjibolaPy/PaViT/blob/main/test.png' alt='test image'>



