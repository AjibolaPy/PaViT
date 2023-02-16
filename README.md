# PaViT
Pathway Vision Transformer Model

#Import the model

from PaVit import PaViT
from keras.models import load_model

#for inference:
#Load the image and resize to (224, 224) and normalize to (0, 1)

model=load_model('trained weight.h5')
prediction=model.predict(img)
predicted_output=np.argmax(prediction, axis=-1)


