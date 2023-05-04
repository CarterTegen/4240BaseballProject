import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from pickle import load
from math import sqrt
import random

class FNN_EventModeler:
    event_names = ['Ball in Dirt',
      'Ball',
      'Bunt Groundout',
      'Bunt Pop Out',
      'Called Strike',
      'Double',
      'Double Play',
      'Foul Ball',
      'Flyout',
      'Forceout',
      'Grounded Into DP',
      'Groundout',
      'Hit by pitch',
      'Home Run',
      'Foul Bunt',
      'Lineout',
      'Missed Bunt',
      'Pop Out',
      'Swinging Strike',
      'Sac Bunt',
      'Sac Fly',
      'Sac Fly DP',
      'Single',
      'Foul Tip',
      'Triple',
      'Triple Play',
      'Swinging Strike (Blocked)']

    PITCH_TYPES = ['CH', 'CU', 'FC', 'FF', 'FS', 'FT', 'KC', 'KN', 'SI', 'SL']

    def __init__(self, model_path, scaler_path):
        self.model = FFNetPytorch(28, 27, hl1=512)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        scaler = load(open(scaler_path, 'rb'))

        self.input_means = scaler.mean_.tolist()
        self.input_variance = scaler.var_.tolist()
        

    def det_event(self, pitch, context):
        context = [val for key, val in context.items()]
        pitches = [1 if pitch["pitch_type"] == val else 0 for val in FNN_EventModeler.PITCH_TYPES]
        pitch = [val for key, val in pitch.items() if key != "pitch_type"]

        pitch = [[(val - self.input_means[i])/(sqrt(self.input_variance[i])) for i, val in enumerate(pitch + pitches + context)]]

        with torch.no_grad():
          probabilities = self.model(torch.tensor(pitch, dtype=torch.float32))
        probabilities = probabilities.detach().tolist()[0]

        probabilities = [val if val > 1e-3 else 0 for val in probabilities]
        # sum = 0
        # for val in probabilities: sum += val
        # probabilities = [val/sum for val in probabilities]
        # event = np.random.choice(FNN_EventModeler.event_names, p = probabilities)

        event = random.choices(FNN_EventModeler.event_names, weights = probabilities)[0]
        return event


class FFNetPytorch(nn.Module):
  sm = torch.nn.Softmax(dim = 1)
  def __init__(self, input_size, output_size, hl1=100):
    super(FFNetPytorch, self).__init__()
    '''
    Define the layers of the neural network. One hidden layer and output layer.
    The activation function used in between the two layers is sigmoid.
    '''
    self.layer1 = nn.Linear(input_size, hl1, bias = True)
    self.layer2 = nn.Linear(hl1, output_size, bias = True)
    # self.layer3 = nn.Linear(hl2, output_size, bias = True)

    self.to(torch.float32)
    

  def forward(self, x):
    '''
    :param x: input to the model (N, NUM_FEATURES)

    :return:
      output: logits of the last layer of the model 
    '''
    x = torch.relu(self.layer1(x))
    # x = torch.relu(self.layer2(x))
    x = self.sm(self.layer2(x))

    return x