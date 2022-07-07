import sys

sys.path.append('./otherwork/misgan/src/')

from celeba_generator import ConvDataGenerator as ConvDataGeneratorCeleb
from celeba_generator import ConvMaskGenerator as ConvMaskGeneratorCeleb
from celeba_critic import ConvCritic as ConvCriticCeleb

from cifar_generator import ConvDataGenerator as ConvDataGeneratorCifar
from cifar_generator import ConvMaskGenerator as ConvMaskGeneratorCifar
from cifar_critic import ConvCritic as ConvCriticCifar

from imputer import UNetImputer


from mnist_generator import ConvDataGenerator as ConvDataGeneratorMnist
from mnist_generator import ConvMaskGenerator as ConvMaskGeneratorMnist
from mnist_critic import ConvCritic as ConvCriticMnist
from mnist_imputer import ComplementImputer as ComplementImputerMnist

from fc_generator import FCDataGenerator as FCDataGenerator
from fc_generator import FCMaskGenerator as FCMaskGenerator
from fc_critic import FCCritic as FCCritic
from fc_imputer import FCImputer as FCImputer

