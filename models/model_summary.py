from torchinfo import summary

from models.hybrid_mednext import HybridMedNeXtPlus


def main():

    model = HybridMedNeXtPlus(
        in_channels=1,
        num_classes=1,
    )

    summary(
        model,
        input_size=(1, 1, 64, 64, 64),
        depth=10,
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
            "trainable",
        ),
        row_settings=("depth", "var_names"),
        verbose=2,
    )


if __name__ == "__main__":
    main()