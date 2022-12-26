import shutil
import os

data_dir = 'C:/Users/user/Desktop/project_2/data/train'
# Каталог с набором данных

train_dir = 'train_1'  # Каталог с данными для обучения
val_dir = 'val_1'  # Каталог с данными для проверки
test_dir = 'test_1'  # Каталог с данными для тестирования

test_data_portion = 0.15  # Часть набора данных для тестирования
val_data_portion = 0.15  # Часть набора данных для проверки

nb_images = 551  # Количество элементов данных в одном классе


def create_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "add"))
    os.makedirs(os.path.join(dir_name, "sub"))
    os.makedirs(os.path.join(dir_name, "mul"))
    os.makedirs(os.path.join(dir_name, "div"))


create_dir(train_dir)
create_dir(val_dir)
create_dir(test_dir)


def copy_img(start_index, end_index, source_dir, dest_dir):
    for i in range(start_index, end_index):
        shutil.copy2(os.path.join(source_dir, "add(" + str(i) + ").jpg"), os.path.join(dest_dir, "add"))
        shutil.copy2(os.path.join(source_dir, "sub(" + str(i) + ").jpg"), os.path.join(dest_dir, "sub"))
        shutil.copy2(os.path.join(source_dir, "mul(" + str(i) + ").jpg"), os.path.join(dest_dir, "mul"))
        shutil.copy2(os.path.join(source_dir, "div(" + str(i) + ").jpg"), os.path.join(dest_dir, "div"))


start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))
print("Стартовый индекс валидационной выборки: ", start_val_data_idx)
print("Стартовый индекс тестовой выборки: ", start_test_data_idx)


copy_img(0, start_val_data_idx, data_dir, train_dir)
copy_img(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
copy_img(start_test_data_idx, nb_images, data_dir, test_dir)
