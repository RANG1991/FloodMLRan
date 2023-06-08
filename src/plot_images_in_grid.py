from matplotlib import pyplot as plt
import glob
import re


def main(folder_name):
    num_rows = 2
    num_cols = 1
    all_images_file = [file for file in glob.glob(f"{folder_name}/*.png")]
    for i in range(len(all_images_file) // (num_rows * num_cols)):
        images = all_images_file[i * (num_rows * num_cols):(i + 1) * (
                num_rows * num_cols)]
        f, axarr = plt.subplots(num_rows, num_cols, figsize=(30, 26))
        row = 0
        column = 0
        for image in images:
            epoch_num = int(re.search("in_epoch_(\\d+)_", image).group(1))
            if num_cols == 1:
                axarr[row].imshow(plt.imread(image))
                axarr[row].axis("off")
                axarr[row].set_title(f"Hydrograph of epoch: {epoch_num}", size=20)
            elif num_rows == 1:
                axarr[column].imshow(plt.imread(image))
                axarr[column].axis("off")
                axarr[column].set_title(f"Hydrography of epoch: {epoch_num}", size=20)
            else:
                axarr[column, row].imshow(plt.imread(image))
                axarr[column, row].axis("off")
                axarr[column, row].set_title(f"Hydrograph in epoch: {epoch_num}", size=20)
            if column % num_rows == (num_rows - 1) or num_cols == 1:
                column = 0
                row += 1
            else:
                column += 1
        f.tight_layout()
        # f.tight_layout(rect=[0.0, 0.0, 0.1, 0.1])
        f.savefig(f"{folder_name}/images_{i}.png")


if __name__ == "__main__":
    main(r"C:\Users\galun\Desktop\basin_images")
