pip install -r requirements.txt

run train.py


24年 triplet sota redo: [https://colab.research.google.com/drive/1Bfy4fwz5_Gh7JP-R3z0iXowfowXCwlIL](https://colab.research.google.com/drive/1Bfy4fwz5_Gh7JP-R3z0iXowfowXCwlIL?usp=sharing)

action+sent contrastive: https://colab.research.google.com/drive/1GkokaigioUM-vUxDyPEiw4SV-UZMApMS?usp=sharing

21年的BART因为用的版本老，所以在本地跑的，源repo在https://github.com/yhcc/BARTABSA/tree/main，把peng/data的train融合了之后，改peng/train.py的training set跑4个数据集分别的testing set
