# cyclegan_pytorch
### Генеративно-состязательная сеть, реализованная на PyTorch.  

Данная сеть перекладывает стилистику изображения на исходные фотографии.
Пример наложения стиля Симпсонов на фото.

![Image alt](https://github.com/Chupacabrik/cyclegan_pytorch/blob/master/results.png)

### Установка

! git clone https://github.com/Chupacabrik/cyclegan_pytorch.git

! sudo pip3 install -r requirements.txt

! python3 cyclegan.py \
--dataset_photo *Папка с фотографиями для обработки* \
--dataset_style *Папка с картинками стиля* \
--num_epochs *Количество эпох для обучения*

Для запуска на Kaggle

! python3 cyclegan_kaggle.py \
--dataset_photo *Папка с фотографиями для обработки* \
--dataset_style *Папка с картинками стиля* \
--num_epochs *Количество эпох для обучения*
