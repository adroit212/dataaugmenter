import os
import shutil
import zipfile
import requests
from io import StringIO

import numpy as np
from PIL import Image
from datetime import datetime

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, Http404
from django.conf import settings
from django.shortcuts import render
from django.utils.encoding import smart_str
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Create your views here.
def home(request):
    return render(request, 'mainapp/home.html')


def data_history(request):
    parentfolder = "mainapp/static/mainapp/data"
    allzips = []
    for filename in sorted(os.listdir(parentfolder), key=str, reverse=True):
        if filename.endswith(".zip"):
            zippath = os.path.join(parentfolder, filename)
            raw_filename = filename[:-4]  # remove .zip extension from name
            name_to_date = datetime.strptime(raw_filename, '%Y%m%d%H%M%S')  # convert the string of filename to date
            subzip = {'filename': name_to_date, 'zippath': zippath}
            allzips.append(subzip)
    content = {'allzips': allzips}

    if 'delete' in request.POST:
        os.remove(request.POST['delete'])  # remove zipped file
        shutil.rmtree(request.POST['delete'][:-4])  # remove folder

    if 'download' in request.POST:
        folderpath = request.POST['download']
        print(folderpath)
        file_path = os.path.join(settings.MEDIA_ROOT, folderpath)
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/zip")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response

    return render(request, 'mainapp/data_history.html', content)


def delete_folder(folderpath):
    pass


def extract_zipped(zip):
    newdest = "mainapp/static/mainapp/data/" + datetime.today().strftime('%Y%m%d%H%M%S')
    os.mkdir(newdest)
    with zipfile.ZipFile(zip, 'r') as zip_ref:
        zip_ref.extractall(newdest)
    return newdest


# Position augmentation for image data augmentation
def augment_image(filepath, savedir):
    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.5, 1.5))
    img = load_img(filepath)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=savedir,
                              save_prefix='image', save_format='jpg'):
        i += 1
        if i > 3:
            break


def perform_augmentation(newfolderpath):
    for filename in os.listdir(newfolderpath):  # loop for data augmentation process
        newpath = os.path.join(newfolderpath, filename)
        if os.path.isdir(newpath):
            perform_augmentation(newpath)
        else:
            augment_image(newpath, newfolderpath)


# dimensionality reduction using Singular Value Decomposition (SVD)
# compress channels
def channel_compress(color_channel, singular_value_limit):
    u, s, v = np.linalg.svd(color_channel)
    compressed = np.zeros((color_channel.shape[0], color_channel.shape[1]))
    n = singular_value_limit

    left_matrix = np.matmul(u[:, 0:n], np.diag(s)[0:n, 0:n])
    inner_compressed = np.matmul(left_matrix, v[0:n, :])
    compressed = inner_compressed.astype('uint8')
    return compressed

def reduce_dim(filepath, savedir):
    image = Image.open(filepath)
    im_array = np.array(image)

    red = im_array[:, :, 0]
    green = im_array[:, :, 1]
    blue = im_array[:, :, 2]
    singular_val_lim = 200

    # compress image
    compressed_red = channel_compress(red, singular_val_lim)
    compressed_green = channel_compress(green, singular_val_lim)
    compressed_blue = channel_compress(blue, singular_val_lim)

    im_red = Image.fromarray(compressed_red)
    im_blue = Image.fromarray(compressed_blue)
    im_green = Image.fromarray(compressed_green)

    new_image = Image.merge("RGB", (im_red, im_green, im_blue))
    # new_image.show()
    new_image.save(filepath)


def perform_dim_reduction(newfolderpath):
    for filename in os.listdir(newfolderpath):  # loop for dimension reduction process
        newpath = os.path.join(newfolderpath, filename)
        if os.path.isdir(newpath):
            perform_dim_reduction(newpath)
        else:
            reduce_dim(newpath, newfolderpath)


def zip_new_dataset_folder(zippath):
    with zipfile.ZipFile(zippath + ".zip", mode='w') as zipf:
        len_dir_path = len(zippath)
        for root, _, files in os.walk(zippath):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path[len_dir_path:])


def data_operation(request):
    contents = {}
    if ('ds' in request.FILES):
        file = request.FILES['ds']
        newpath = extract_zipped(file)
        perform_augmentation(newpath)
        perform_dim_reduction(newpath)
        zip_new_dataset_folder(newpath)
        refined_path = newpath[:-1] + ".zip"
        contents = {'zipfile': newpath, 'flag': True}

    elif ('fp' in request.POST):
        print("here")
        folderpath = request.POST['fp'] + ".zip"
        print(folderpath)
        file_path = os.path.join(settings.MEDIA_ROOT, folderpath)
        print(file_path)
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/zip")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    return render(request, 'mainapp/home.html', contents)
    # return HttpResponse("<script>alert('Operation Successful. File download will begin in few minutes'); "
    # "window.location.href='/download_zip?folderpath=" + newpath + "/';</script>")
