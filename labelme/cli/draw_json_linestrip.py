#!/usr/bin/env python

import argparse
import base64
import json
import os
import sys
import io

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from labelme import utils


PY2 = sys.version_info[0] == 2

def draw_label(label, img=None, label_names=None, colormap=None, legend=True, **kwargs):
    """Draw pixel-wise label with colorization and label names.

    label: ndarray, (H, W)
        Pixel-wise labels to colorize.
    img: ndarray, (H, W, 3), optional
        Image on which the colorized label will be drawn.
    label_names: iterable
        List of label names.
    """
    import matplotlib.pyplot as plt

    backend_org = plt.rcParams['backend']
    plt.switch_backend('agg')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if label_names is None:
        label_names = [str(l) for l in range(label.max() + 1)]

    colormap = utils.draw._validate_colormap(colormap, len(label_names))

    label_viz = utils.draw.label2rgb(
        label, None, n_labels=len(label_names), colormap=colormap, **kwargs
    )
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names):
        if label_value not in label:
            continue
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append('{value}: {name}'
                          .format(value=label_value, name=label_name))
    if legend:
        plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)

    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    plt.switch_backend(backend_org)

    out_size = (label_viz.shape[1], label_viz.shape[0])
    out = PIL.Image.open(f).resize(out_size, PIL.Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('--visualize', default=False)
    args = parser.parse_args()

    json_file = args.json_file

    print('generating data from %s'%(json_file))
    data = json.load(open(json_file))

    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {'_background_': 0}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    print(label_names)
    # semantic image
    cmap = np.array([[0, 0, 0],
                     [20, 20, 20],
                     [70, 70, 70],
                     [120, 120, 120],
                     [170, 170, 170],
                     [220, 220, 220]])
    cmap = np.vstack([cmap, np.zeros([10,3])]) # !!WARNING!! ignore line 6-
    cmap = cmap / 255.0
    lbl_viz = draw_label(lbl, img, label_names, colormap=cmap, legend=False)

    if args.visualize:
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(lbl_viz)
        plt.show()
    basename = os.path.splitext(json_file)[0]
    plt.imsave(basename + '_semantic.png', lbl_viz)

    # binary image
    cmap[cmap>0] = 1.0
    lbl_bin = draw_label(lbl, img, label_names, colormap=cmap, legend=False)

    basename = os.path.splitext(json_file)[0]
    plt.imsave(basename + '_binary.png', lbl_bin)





if __name__ == '__main__':
    main()
