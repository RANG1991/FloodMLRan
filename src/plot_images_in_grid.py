from matplotlib import pyplot as plt
import glob
import re
import cv2
import numpy as np


def crop_hydrograph_image(image):
    hydrograph_image_color = cv2.imread(image)
    hydrograph_image_greyscale = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(hydrograph_image_greyscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(hydrograph_image_greyscale)
    cv2.drawContours(mask, contours, 0, 255, -1)
    hydrograph_image_cropped = np.zeros_like(hydrograph_image_greyscale)
    hydrograph_image_cropped[mask == 255] = hydrograph_image_greyscale[mask == 255]
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    hydrograph_image_cropped = hydrograph_image_color[topy:bottomy + 1, topx:bottomx + 1]
    return hydrograph_image_cropped


def main(folder_name, model_name):
    num_rows = 3
    num_cols = 1
    all_images_file = [file for file in glob.glob(f"{folder_name}/Hydrograph_of_*.png")]
    filtered_all_images = list(
        filter(lambda image: int(re.search("in_epoch_(\\d+)_", image).group(1)) in [1, 15, 25],
               all_images_file))
    all_images_file = sorted(filtered_all_images, key=lambda image: int(re.search("in_epoch_(\\d+)_", image).group(1)))
    for i in range(len(all_images_file) // (num_rows * num_cols)):
        images = all_images_file[i * (num_rows * num_cols):(i + 1) * (
                num_rows * num_cols)]
        f, axarr = plt.subplots(num_rows, num_cols, figsize=(45, 39), gridspec_kw=dict(hspace=0, wspace=0))
        row = 0
        column = 0
        for image in images:
            epoch_num = int(re.search("in_epoch_(\\d+)_", image).group(1))
            hydrograph_image_cropped = crop_hydrograph_image(image)
            if num_cols == 1:
                axarr[row].imshow(hydrograph_image_cropped)
                axarr[row].axis("off")
                axarr[row].set_title(f"Hydrograph of epoch: {epoch_num}", size=30)
            elif num_rows == 1:
                axarr[column].imshow(hydrograph_image_cropped)
                axarr[column].axis("off")
                axarr[column].set_title(f"Hydrography of epoch: {epoch_num}", size=30)
            else:
                axarr[column, row].imshow(hydrograph_image_cropped)
                axarr[column, row].axis("off")
                axarr[column, row].set_title(f"Hydrograph in epoch: {epoch_num}", size=30)
            if column % num_rows == (num_rows - 1) or num_cols == 1:
                column = 0
                row += 1
            else:
                column += 1
        # plt.tight_layout()
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.suptitle(f"Hydrographs of model: {model_name.replace('_', '-')}", fontsize=32)
        f.savefig(f"./analysis_images/Hydrographs_images_{model_name}.png")


if __name__ == "__main__":
    for model_name in ["CNN_LSTM", "CNN_Transformer"]:
        main(
            fr"C:\Users\galun\PyCharmProjects\FloodMLRan\slurm_output_files\slurm_files_ensemble_comparison\hydrographs_{model_name}",
            model_name)
