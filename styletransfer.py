from PIL import Image
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

# скачиваем модель
vgg = models.vgg19(pretrained=True).features

# замораживаем все параметры, так как мы меняем только target image
for param in vgg.parameters():
    param.requires_grad_(False)

# перемещаем модель на gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

def load_image(img_path, max_size=400, shape=None):
    '''Загрузка и преобразования изображения.'''
    
    image = Image.open(img_path).convert('RGB')
    
    # ставим ограничение на размер картинки
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # убираем пустые размерности и добавляем размерность батчей
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

def im_convert(tensor):
''' Преобразование tensor в nparray.'''

  image = tensor.to("cpu").clone().detach()
  image = image.numpy().squeeze()
  image = image.transpose(1,2,0)
  image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
  image = image.clip(0, 1)

  return image

def get_features(image, model, layers=None):
'''Получаение представления картинки.'''
  if layers is None:
      layers = {'0': 'conv1_1',
                '5': 'conv2_1', 
                '10': 'conv3_1', 
                '19': 'conv4_1',
                '21': 'conv4_2',  ## представление content image
                '28': 'conv5_1'}
        
  features = {}
  x = image
  for name, layer in model._modules.items():
      x = layer(x)
      if name in layers:
          features[layers[name]] = x
          
  return features

def gram_matrix(tensor):
  '''Подсчет матриц Грама.'''

  _, d, h, w = tensor.size()
  
  tensor = tensor.view(d, h * w)

  gram = torch.mm(tensor, tensor.t())
  
  return gram 

def style_transfer(content_url, style_url):
  '''Перенос стиля.'''

  # загрузка content и style images
  content = load_image(content_url).to(device)
  style = load_image(style_url, shape=content.shape[-2:]).to(device)

  # получаем представление content и style images
  content_features = get_features(content, vgg)
  style_features = get_features(style, vgg)

  # считаем матрицы Грама для style image
  style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

  # создаем третье target изображение, подготавливаем его для изменения и берем за основу content image
  target = content.clone().requires_grad_(True).to(device)

  # веса для каждого слоя
  style_weights = {'conv1_1': 1.,
                  'conv2_1': 0.75,
                  'conv3_1': 0.2,
                  'conv4_1': 0.2,
                  'conv5_1': 0.2}

  content_weight = 1  
  style_weight = 1e9  


  # задаем гиперпараметры
  optimizer = optim.Adam([target], lr=0.003)
  steps = 400  

  for ii in range(1, steps+1):
      
      # получаем представление target image
      target_features = get_features(target, vgg)
      
      # content loss
      content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
      
      # style loss
      style_loss = 0
      for layer in style_weights:
          target_feature = target_features[layer]
          target_gram = gram_matrix(target_feature)
          _, d, h, w = target_feature.shape
          style_gram = style_grams[layer]
          layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
          style_loss += layer_style_loss / (d * h * w)
          
      # подсчет итоговог лосса
      total_loss = content_weight * content_loss + style_weight * style_loss
      
      # изменяем target image
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
  stylized = im_convert(target)
  img = Image.fromarray((stylized * 255).astype(np.uint8))
  img.save('stylized.jpg')
