from matplotlib import pyplot
import re

PATTERN = r"The NSE loss for basin with ID: (\d+) is: ([-+]?\d*[.,]?\d*)"


def create_dict_basin_id_to_NSE(logs_filename):
    d = {}
    with open(logs_filename, "r") as f:
        for row in f:
            basin_nse_loss_str = re.search(PATTERN, row)
            if basin_nse_loss_str is not None:
                basin_id = basin_nse_loss_str.group(1)
                nse_loss = basin_nse_loss_str.group(2)
                if basin_id not in d.keys():
                    d[basin_id] = []
                d[basin_id].append(float(nse_loss))
    return d


def main():
    with open("./nse_per_basin", "w") as f:
        d = create_dict_basin_id_to_NSE("./output.log")
        for basin_id, nse_losses_list in d.items():
            print("Basin ID: {}, NSE losses: {}, mean NSE losses: {}".format(basin_id, nse_losses_list,
                                                                             sum(nse_losses_list) / len(
                                                                                 nse_losses_list)))
            f.write("Basin ID: {}, NSE losses: {}, mean NSE losses: {}\n".format(basin_id, nse_losses_list,
                                                                                 sum(nse_losses_list) / len(
                                                                                     nse_losses_list)))


if __name__ == "__main__":
    main()
