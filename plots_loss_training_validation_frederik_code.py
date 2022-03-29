from matplotlib import pyplot
import re

DECIMAL_POINT_PATTERN = "\d*[.,]?\d*"


def get_validation_and_training_scores_from_logs(logs_filename):
    val_losses = []
    train_losses = []
    with open(logs_filename, "r") as f:
        for row in f:
            result_train_search = re.search("Epoch ({}) average loss: ({})".format(DECIMAL_POINT_PATTERN,
                                                                                   DECIMAL_POINT_PATTERN), row)
            result_val_search = re.search("Epoch ({}) average validation loss: "
                                          "({}) -- Median validation metrics: NSE: "
                                          "({})".format(DECIMAL_POINT_PATTERN, DECIMAL_POINT_PATTERN,
                                                        DECIMAL_POINT_PATTERN), row)
            if result_val_search is not None:
                val_losses.append((int(result_val_search.group(1)), float(result_val_search.group(2))))
            elif result_train_search is not None:
                train_losses.append((int(result_train_search.group(1)), float(result_train_search.group(2))))
    return train_losses, val_losses


def main():
    train_losses, val_losses = get_validation_and_training_scores_from_logs("./output.log")
    pyplot.plot([x[0] for x in train_losses], [x[1] for x in train_losses])
    pyplot.plot([x[0] for x in val_losses], [x[1] for x in val_losses])
    pyplot.legend(["train loss", "validation loss"])
    pyplot.show()


if __name__ == "__main__":
    main()
