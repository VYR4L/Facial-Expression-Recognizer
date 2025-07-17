from pathlib import Path
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
import torch.nn as nn


# Definindo o diretório raiz e os caminhos para os conjuntos de dados de treinamento e validação
ROOT_DIR = Path(__file__).parent.parent
DATA_SET = ROOT_DIR / 'dataset' / 'images'
TRAIN_IMAGES = ROOT_DIR / 'dataset' / 'images' / 'train'
VALIDATION_IMAGES = ROOT_DIR / 'dataset' / 'images' / 'validation'
OUTPUT_DIR = ROOT_DIR / 'results'

# Definindo o caminho para o conjunto de dados
output_type = {}
count_type = [0 for i in range(7)]

# Definindo os tipos de emoções e suas respectivas contagens
for index, val in enumerate(glob.glob(str(TRAIN_IMAGES / '*'))):
    output_type[Path(val).parts[-1]] = index


class Data(Dataset):
    '''
    Classe para carregar os dados de treinamento e validação.

    A classe herda de Dataset do PyTorch e implementa os métodos __len__ e __getitem__.
    '''
    def __init__(self, is_train=True):
        self.is_train = is_train
        data_type = 'train' if is_train else 'validation'
        
        self.path_file = glob.glob(str(DATA_SET / data_type / '*' / '*'))

        if is_train:
            temp_data = []
            for i in range(9):
                self.path_file += glob.glob(str(DATA_SET / data_type / 'disgust' / '*'))
            
            for type in output_type:
                counter = 0
                for path in self.path_file:
                    path_parts = Path(path).parts
                    if len(path_parts) > 6 and path_parts[-2] == type:
                        counter += 1
                        temp_data.append(path)
                        count_type[output_type[type]] += 1
            self.path_file = temp_data

        self.transform = ToTensor()
        self.data = nn.Sequential(
                                RandomResizedCrop((48, 48),
                                    scale=(0.8, 1),
                                    ratio=(0.5, 1)),
                                RandomHorizontalFlip(),
                                RandomVerticalFlip(),
        )

    def __len__(self):
        '''
        Retorna o número total de imagens no conjunto de dados.
        O número total de imagens é obtido contando o número de arquivos de imagem no diretório especificado.
        '''
        return len(self.path_file)

    def __getitem__(self, idx):
        '''
        Retorna uma imagem e seu rótulo correspondente com base no índice fornecido.
        O método carrega a imagem correspondente ao índice, aplica as transformações necessárias e retorna a imagem e seu rótulo.

        params:
            idx (int): índice da imagem a ser carregada.

        returns:
            tuple: imagem transformada e rótulo correspondente.
        '''
        image_path = self.path_file[idx]
        image = Image.open(image_path)
        label = output_type[Path(image_path).parts[-2]]
        image = self.transform(image)
        if self.is_train:
            image = self.data(image)
        
        return (image, label)