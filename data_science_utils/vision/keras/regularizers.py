import numpy as np


def get_cutout_eraser(proba=1.0, s_l=0.04, s_h=0.06, r_1=0.35, r_2=1 / 0.35,
                      max_erasures_per_image=1, pixel_level=True):
    """

    :param p:
    :param s_l: Minimum Area Proportion of Original that may be cut
    :param s_h: Maximum Area Proportion of Original that may be cut
    :param r_1: Min Aspect Ratio
    :param r_2: Max Aspect Ratio
    :param max_erasures_per_image:
    :param pixel_level:
    :return: Eraser to be used as Preprocessing Function
    """
    assert max_erasures_per_image >= 1

    def eraser(input_img):
        p_1 = np.random.rand()
        if p_1 > proba:
            return input_img
        img_h, img_w, img_c = input_img.shape
        shape = input_img.shape

        v_l = np.min(input_img)
        v_h = np.max(input_img)

        #         mx = np.random.randint(1, max_erasures_per_image + 1)
        mx = max_erasures_per_image
        for i in range(mx):
            while True:
                s = np.random.uniform(s_l, s_h) * img_h * img_w
                r = np.random.uniform(r_1, r_2)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, img_w)
                top = np.random.randint(0, img_h)

                if left + w <= img_w and top + h <= img_h:
                    break

            if pixel_level:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            else:
                c = np.random.uniform(v_l, v_h)

            input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser
