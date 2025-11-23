from dotmap import DotMap
import torch
from verifai.client import Client
import uuid 
import os
from predict import predict_car_count
import torch.nn.functional as F
from model import ClassifierCNN_128p
from generate_images import genImage

MODEL_PATH = 'model/car_counter_model_1.pth'

class Classifier(Client):
	def __init__(self, classifier_data):
		port = classifier_data.port
		bufsize = classifier_data.bufsize
		super().__init__(port, bufsize)
		model = ClassifierCNN_128p()
		state_dict = torch.load(MODEL_PATH, map_location='cpu')
		model.load_state_dict(state_dict)
		model.eval()
		self.nn = model

	def simulate(self, sample):
		img = genImage(sample)
		if img is None:
			return 1
		yTrue = int(sample.numCars)
		logits = predict_car_count(img, self.nn)
		
		res = {}
		probs = F.softmax(logits, dim=1)
		
		yPred = probs.argmax(dim=1).item()

		confidence_in_truth = probs[0][yTrue].item()

		res['yTrue'] = yTrue
		res['yPred'] = yPred
		res['confidence'] = confidence_in_truth
		# print(probs[0])
		print(f"Predicted: {yPred}, True: {yTrue}, confidence: {confidence_in_truth}")

		if confidence_in_truth < 0.8 or yPred != yTrue:
			folder_path = f"misclassified/car_{yTrue}"
			os.makedirs(folder_path, exist_ok=True)
			unique_name = f"fail_{uuid.uuid4().hex[:8]}.png"
			full_path = os.path.join(folder_path, unique_name)
			try:
				img.save(full_path)
				# print(f"Saved misclassified image to: {full_path}")
			except Exception as e:
				print(f"Error saving image: {e}")
		else:
			folder_path = f"correctly_classified/car_{yTrue}"
			os.makedirs(folder_path, exist_ok=True)
			unique_name = f"correct_{uuid.uuid4().hex[:8]}.png"
			full_path = os.path.join(folder_path, unique_name)
			try:
				img.save(full_path)
				# print(f"Saved correctly classified image to: {full_path}")
			except Exception as e:
				print(f"Error saving image: {e}")

		return res



PORT = 8888
BUFSIZE = 4096

classifier_data = DotMap()
classifier_data.port = PORT
classifier_data.bufsize = BUFSIZE

client_task = Classifier(classifier_data)
while True:
	if not client_task.run_client():
		print("End of all classifier calls")
		break
