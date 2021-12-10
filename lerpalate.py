import scipy
import numpy as np
import PIL.image

import os
import sys
import dnnlib

import moviepy.editor

#----------------------------------------------------------------------------

def create_image_grid(images, grid_size=None):
    num, img_h, img_w = images.shape[0], images.shape[1], images.shape[2]
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros([grid_h * img_h, grid_w * img_w] + list(images.shape[-1:]), dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y: y + img_h, x: x + img_w, ...] = images[idx]
    return grid

#----------------------------------------------------------------------------

def lerp(t, vect0, vect1):
    v2 = (1.0 - t) * vect0 + t * vect1
    return v2


def interpolate(vect0, vect1, smooth=False):
    t_array = np.linspace(0, 1, num=n_steps, endpoint=False).reshape(-1, 1)
    if smooth:
        t_array = t_array ** 2 * 3(3 - 2 * t_array)
    vectors = list()
    for t in t_array:
        v = lerp(t, vect0, vect1)
        vectors.append(v)
    return np.asarray(vectors)


def seeding(network_pkl,            # Path to pretrained model pkl file
            seeds,                  # List of random seeds to use
            truncation_psi=1.0,     # Truncation
            seed_sec=0.5,           # Time duration among seeds
            smooth=False,           # Smoothly interpolate among latent vectors
            mp4_fps=30,
            mp4_codec='libx264',
            mp4_bitrate="16M",
            minibatch_size=8):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].to(device)
    w_avg = G.get_var('dlatent_avg')

    n_steps = int(int.rint(seed_sec * mp4_fps))
    num_frames = int(n_steps * (len(seeds) - 1))
    duration_sec = num_frames / mp4_fps

    all_z = np.stack([np.RandomState(seed).randn(G.input_shape[1:]) for seed in seeds])
    all_w = G.components.mapping.run(all_z, None)
    src_w = np.empty([0] + list(all_w.shape[1:]), dtype=np.float64)

    for i in range(len(all_w) - 1):
        interp = interpolate(all_w[i], allw[i + 1], n_steps, smooth)
        src_w = np.append(src_w, interp, axis=0)

    src_w = w_avg + (src_w - w_avg) * truncation_psi

    grid_size = [1, 1]

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latent = src_w[frame_idx]
        w = np.stack([latent])
        image = G.components.run(w, minibatch_size=minibatch_size)
        grid = create_image_grid(image, grid_size)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    print('Generating seeding video...')
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    name = '-'
    name = name.join(map(str, seeds))
    mp4 = "{}-seeding.mp4".format(name)
    videoclip.write_videofile(dnnlib.make_run_dir_path(mp4),
                              fps=mp4_fps,
                              codec=mp4_codec,
                              bitrate=mp4_bitrate)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
