from numpy import loadtxt
from keras.models import model_from_json
dataset = loadtxt('diabetes_dataset.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

json_file = open('diabetes_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("diabetes_model.h5")
print("Model Loading completed")

predictions = model.predict(x)
for i in range(5,15):
  expected_label = 'diabetes positive' if y[i] == 1 else 'diabetes negative'
  print('%s => %d (expected %d, %s)' % (x[i].tolist(), predictions[i], y[i], expected_label))
