from matplotlib import pyplot as plt
import glob


def main(folder_name):
    num_of_images_in_row = 2
    num_of_images_in_col = 2
    all_images_file = [file for file in glob.glob(f"{folder_name}/*.png")]
    for i in range(len(all_images_file[::(num_of_images_in_row * num_of_images_in_col)])):
        images = all_images_file[i:i + (num_of_images_in_row * num_of_images_in_col)]
        f, axarr = plt.subplots(num_of_images_in_row, num_of_images_in_col, figsize=(10, 8))
        row = 0
        column = 0
        for image in images:
            axarr[column, row].imshow(plt.imread(image))
            axarr[column, row].axis("off")
            if column % num_of_images_in_row == (num_of_images_in_row - 1):
                column = 0
                row += 1
            else:
                column += 1
        f.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
        f.savefig(f"{folder_name}/images_{i}.png")


if __name__ == "__main__":
    main(r"C:\Users\galun\Desktop\images_FM")
