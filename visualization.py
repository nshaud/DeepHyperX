import matplotlib.pyplot as plt
import numpy as np
import spectral


def display_predictions(pred, writer, gt=None, caption=""):
    writer.add_image("Segmentation/" + caption, pred, dataformats="HWC")


def display_dataset(img, gt, bands, labels, palette, writer=None):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype="uint8")

    # Display the RGB composite image
    caption = "HSI/RGB (bands {}, {}, {})".format(*bands)
    # Send to Tensorboard
    writer.add_image(caption, rgb, dataformats="HWC")


def explore_spectrums(img, complete_gt, class_names, writer, ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        writer : TensorBoard writer
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure(figsize=(8, 6))
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(
            range(len(mean_spectrum)), lower_spectrum, higher_spectrum, color="#3F5D7D"
        )
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        writer.add_figure(f"Spectra/{class_names[c]}", fig)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, writer, title=""):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        writer: TensorBoard writer
    """
    fig = plt.figure(figsize=(12, 10))
    for name, spectrum in spectrums.items():
        n_bands = len(spectrum)
        plt.plot(np.arange(n_bands), spectrum, label=name)
        plt.legend()
        plt.xlim(0, n_bands)
        plt.title(title)
    writer.add_figure(title, fig)
