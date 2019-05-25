import time
import numpy as np
from PIL import Image as pil_image
from keras.preprocessing.image import save_img
from keras import layers
from keras.applications import vgg16
from keras import backend as K
import matplotlib.pyplot as plt


def normalize(x):
    """utility function to normalize a tensor.

    # Arguments
        x: An input tensor.

    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_image(x):
    """utility function to convert a float array into a valid uint8 image.

    # Arguments
        x: A numpy-array representing the generated image.

    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def process_image(x, former):
    """utility function to convert a valid uint8 image back into a float array.
       Reverses `deprocess_image`.

    # Arguments
        x: A numpy-array, which could be used in e.g. imshow.
        former: The former numpy-array.
                Need to determine the former mean and variance.

    # Returns
        A processed numpy-array representing the generated image.
    """
    if K.image_data_format() == 'channels_first':
        x = x.transpose((2, 0, 1))
    return (x / 255 - 0.5) * 4 * former.std() + former.mean()


def visualize_layer(model,
                    layer_name,
                    step=0.5,
                    epochs=20,
                    upscaling_steps=10,
                    upscaling_factor=1.2,
                    output_dim=(128, 128),
                    filter_range=(0, None),
                    grid_columns=4,
                    show_filters=True):
    """Visualizes the most relevant filters of one conv-layer in a certain model.

    # Arguments
        model: The model containing layer_name.
        layer_name: The name of the layer to be visualized.
                    Has to be a part of model.
        step: step size for gradient ascent.
        epochs: Number of iterations for gradient ascent.
        upscaling_steps: Number of upscaling steps.
                         Starting image is in this case (80, 80).
        upscaling_factor: Factor to which to slowly upgrade
                          the image towards output_dim.
        output_dim: [img_width, img_height] The output image dimensions.
        filter_range: Tupel[lower, upper]
                      Determines the to be computed filter numbers.
                      If the second value is `None`,
                      the last filter will be inferred as the upper boundary.
    """

    def _generate_filter_image(input_img,
                               layer_output,
                               filter_index,
                               channels=3):
        """Generates image for one particular filter.

        # Arguments
            input_img: The input-image Tensor.
            layer_output: The output-image Tensor.
            filter_index: The to be processed filter number.
                          Assumed to be valid.

        #Returns
            Either None if no image could be generated.
            or a tuple of the image (array) itself and the last loss.
        """
        s_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from a gray image with some random noise
        intermediate_dim = tuple(
            int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)

        def _get_input_random_image():
            if K.image_data_format() == 'channels_first':
                input_img_data = np.random.random(
                    (1, channels, intermediate_dim[0], intermediate_dim[1]))
            else:
                input_img_data = np.random.random(
                    (1, intermediate_dim[0], intermediate_dim[1], channels))
            input_img_data = (input_img_data - 0.5) * 20 + 128
            return input_img_data

        input_img_data = _get_input_random_image()

        # Slowly upscaling towards the original size prevents
        # a dominating high-frequency of the to visualized structure
        # as it would occur if we directly compute the 412d-image.
        # Behaves as a better starting point for each following dimension
        # and therefore avoids poor local minima
        losses, grads = [],[]
        for up in reversed(range(upscaling_steps)):
            # we run gradient ascent for e.g. 20 steps
            for _ in range(epochs):
                loss_value, grads_value = iterate([input_img_data])
                losses.append(loss_value)
                grads.append(np.sum(np.abs(grads_value)))


                # some filters get stuck to 0, we can skip them
                if np.sum(losses) <= 1e-06:
                    input_img_data = _get_input_random_image()
                    # return None
                input_img_data += grads_value * step

            # Calulate upscaled dimension
            intermediate_dim = tuple(
                int(x / (upscaling_factor ** up)) for x in output_dim)
            # Upscale
            mode = "L" if channels == 1 else None
            img = deprocess_image(input_img_data[0])
            if channels == 1:
                img = img.reshape((img.shape[0], img.shape[1]))
                img = np.array(pil_image.fromarray(img, mode).resize(intermediate_dim,
                                                                     pil_image.BICUBIC))
                img = img.reshape((img.shape[0], img.shape[1], 1))
            else:
                img = np.array(pil_image.fromarray(img).resize(intermediate_dim,
                                                               pil_image.BICUBIC))
            input_img_data = [process_image(img, input_img_data[0])]

        # decode the resulting input image
        img = deprocess_image(input_img_data[0])
        e_time = time.time()
        print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(filter_index,
                                                                  loss_value,
                                                                  e_time - s_time))
        return img, loss_value

    def _draw_filters(filters, columns=4, show_filters=True, channels=3):
        """Draw the best filters in a nxn grid.

        # Arguments
            filters: A List of generated images and their corresponding losses
                     for each processed filter.
            n: dimension of the grid.
               If none, the largest possible square will be used
        """

        rows = int(np.ceil(len(filters) / columns))

        output_dim = (filters[0][0].shape[0], filters[0][0].shape[1])

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top n*n filters.
        filters.sort(key=lambda x: x[1], reverse=True)

        # build a black picture with enough space for
        # e.g. our 8 x 8 filters of size 412 x 412, with a 5px margin in between
        MARGIN = 5
        width = rows * output_dim[0] + (rows - 1) * MARGIN
        height = columns * output_dim[1] + (columns - 1) * MARGIN
        stitched_filters = np.zeros((width, height, channels), dtype='uint8')
        # fill the picture with our saved filters
        for i in range(rows):
            for j in range(columns):
                idx = min(i * columns + j, len(filters) - 1)
                if i * columns + j > len(filters) - 1:
                    img = np.zeros_like(filters[0][0])
                else:
                    img, _ = filters[idx]
                width_margin = (output_dim[0] + MARGIN) * i
                height_margin = (output_dim[1] + MARGIN) * j
                stitched_filters[
                width_margin: width_margin + output_dim[0],
                height_margin: height_margin + output_dim[1], :] = img
        if show_filters:
            fig_height = rows * 4
            fig_width = columns * 4

            fig = plt.figure(figsize=(fig_width, fig_height))
            plt.imshow(stitched_filters)
            plt.title('{0:}_{1:}x{2:}.png'.format(layer_name, rows, columns))
            plt.show()
        # save the result to disk
        save_img('{0:}_{1:}x{2:}.png'.format(layer_name, rows, columns), stitched_filters)

    # this is the placeholder for the input images
    assert len(model.inputs) == 1
    input_img = model.inputs[0]
    channels = K.int_shape(model.inputs[0])[-1]

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])

    output_layer = layer_dict[layer_name]
    assert isinstance(output_layer, layers.Conv2D)

    # Compute to be processed filter range
    filter_lower = filter_range[0]
    filter_upper = (filter_range[1]
                    if filter_range[1] is not None
                    else len(output_layer.get_weights()[1]))
    assert (filter_lower >= 0
            and filter_upper <= len(output_layer.get_weights()[1])
            and filter_upper > filter_lower)
    print('Compute filters {:} to {:}'.format(filter_lower, filter_upper))

    # iterate through each filter and generate its corresponding image
    processed_filters = []
    for f in range(filter_lower, filter_upper):
        img_loss = _generate_filter_image(input_img, output_layer.output, f, channels)

        if img_loss is not None:
            processed_filters.append(img_loss)

    print('{} filter processed.'.format(len(processed_filters)))
    # Finally draw and store the best filters to disk

    print("Filter Losses\n",[loss for f, loss in processed_filters])
    _draw_filters(processed_filters, grid_columns, show_filters)


if __name__ == '__main__':
    # the name of the layer we want to visualize
    # (see model definition at keras/applications/vgg16.py)
    LAYER_NAME = 'block5_conv1'

    # build the VGG16 network with ImageNet weights
    vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')
    vgg.summary()

    # example function call
    visualize_layer(vgg, LAYER_NAME, filter_range=(0, 8))