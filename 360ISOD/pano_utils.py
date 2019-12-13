import numpy as np
from scipy.ndimage import map_coordinates
import os

def xyzcube(face_w):
    '''
    Return the xyz cordinates of the unit cube in [F R B L U D] format.
    '''
    out = np.zeros((face_w, face_w * 6, 3), np.float32)
    rng = np.linspace(-0.5, 0.5, num=face_w, dtype=np.float32)
    grid = np.stack(np.meshgrid(rng, -rng), -1)

    # Front face (z = 0.5)
    out[:, 0*face_w:1*face_w, [0, 1]] = grid
    out[:, 0*face_w:1*face_w, 2] = 0.5

    # Right face (x = 0.5)
    out[:, 1*face_w:2*face_w, [2, 1]] = grid
    out[:, 1*face_w:2*face_w, 0] = 0.5

    # Back face (z = -0.5)
    out[:, 2*face_w:3*face_w, [0, 1]] = grid
    out[:, 2*face_w:3*face_w, 2] = -0.5

    # Left face (x = -0.5)
    out[:, 3*face_w:4*face_w, [2, 1]] = grid
    out[:, 3*face_w:4*face_w, 0] = -0.5

    # Up face (y = 0.5)
    out[:, 4*face_w:5*face_w, [0, 2]] = grid
    out[:, 4*face_w:5*face_w, 1] = 0.5

    # Down face (y = -0.5)
    out[:, 5*face_w:6*face_w, [0, 2]] = grid
    out[:, 5*face_w:6*face_w, 1] = -0.5

    return out


def equirect_uvgrid(h, w):
    u = np.linspace(-np.pi, np.pi, num=w, dtype=np.float32)
    v = np.linspace(np.pi, -np.pi, num=h, dtype=np.float32) / 2

    return np.stack(np.meshgrid(u, v), axis=-1)


def equirect_facetype(h, w):
    '''
    0F 1R 2B 3L 4U 5D
    '''
    tp = np.roll(np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1)

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), np.bool)
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(np.int32)


def xyzpers(h_fov, v_fov, u, v, out_hw, in_rot):
    out = np.ones((*out_hw, 3), np.float32)

    x_max = np.tan(h_fov / 2)
    y_max = np.tan(v_fov / 2)
    x_rng = np.linspace(-x_max, x_max, num=out_hw[1], dtype=np.float32)
    y_rng = np.linspace(-y_max, y_max, num=out_hw[0], dtype=np.float32)
    out[..., :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = rotation_matrix(v, [1, 0, 0])
    Ry = rotation_matrix(u, [0, 1, 0])
    Ri = rotation_matrix(in_rot, np.array([0, 0, 1.0]).dot(Rx).dot(Ry))

    return out.dot(Rx).dot(Ry).dot(Ri)


def xyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x**2 + z**2)
    v = np.arctan2(y, c)

    return np.concatenate([u, v], axis=-1)


def uv2unitxyz(uv):
    u, v = np.split(uv, 2, axis=-1)
    y = np.sin(v)
    c = np.cos(v)
    x = c * np.sin(u)
    z = c * np.cos(u)

    return np.concatenate([x, y, z], axis=-1)


def uv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5

    return np.concatenate([coor_x, coor_y], axis=-1)


def coor2uv(coorxy, h, w):
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * np.pi
    v = -((coor_y + 0.5) / h - 0.5) * np.pi

    return np.concatenate([u, v], axis=-1)


def sample_equirec(e_img, coor_xy, order):
    w = e_img.shape[1]
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    pad_u = np.roll(e_img[[0]], w // 2, 1)
    pad_d = np.roll(e_img[[-1]], w // 2, 1)
    e_img = np.concatenate([e_img, pad_d, pad_u], 0)
    return map_coordinates(e_img, [coor_y, coor_x],
                           order=order, mode='wrap')[..., 0]


def sample_cubefaces(cube_faces, tp, coor_y, coor_x, order):
    cube_faces = cube_faces.copy()
    cube_faces[1] = np.flip(cube_faces[1], 1)
    cube_faces[2] = np.flip(cube_faces[2], 1)
    cube_faces[4] = np.flip(cube_faces[4], 0)

    # Pad up down
    pad_ud = np.zeros((6, 2, cube_faces.shape[2]))
    pad_ud[0, 0] = cube_faces[5, 0, :]
    pad_ud[0, 1] = cube_faces[4, -1, :]
    pad_ud[1, 0] = cube_faces[5, :, -1]
    pad_ud[1, 1] = cube_faces[4, ::-1, -1]
    pad_ud[2, 0] = cube_faces[5, -1, ::-1]
    pad_ud[2, 1] = cube_faces[4, 0, ::-1]
    pad_ud[3, 0] = cube_faces[5, ::-1, 0]
    pad_ud[3, 1] = cube_faces[4, :, 0]
    pad_ud[4, 0] = cube_faces[0, 0, :]
    pad_ud[4, 1] = cube_faces[2, 0, ::-1]
    pad_ud[5, 0] = cube_faces[2, -1, ::-1]
    pad_ud[5, 1] = cube_faces[0, -1, :]
    cube_faces = np.concatenate([cube_faces, pad_ud], 1)

    # Pad left right
    pad_lr = np.zeros((6, cube_faces.shape[1], 2))
    pad_lr[0, :, 0] = cube_faces[1, :, 0]
    pad_lr[0, :, 1] = cube_faces[3, :, -1]
    pad_lr[1, :, 0] = cube_faces[2, :, 0]
    pad_lr[1, :, 1] = cube_faces[0, :, -1]
    pad_lr[2, :, 0] = cube_faces[3, :, 0]
    pad_lr[2, :, 1] = cube_faces[1, :, -1]
    pad_lr[3, :, 0] = cube_faces[0, :, 0]
    pad_lr[3, :, 1] = cube_faces[2, :, -1]
    pad_lr[4, 1:-1, 0] = cube_faces[1, 0, ::-1]
    pad_lr[4, 1:-1, 1] = cube_faces[3, 0, :]
    pad_lr[5, 1:-1, 0] = cube_faces[1, -2, :]
    pad_lr[5, 1:-1, 1] = cube_faces[3, -2, ::-1]
    cube_faces = np.concatenate([cube_faces, pad_lr], 2)

    return map_coordinates(cube_faces, [tp, coor_y, coor_x], order=order, mode='wrap')


def cube_h2list(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    return np.split(cube_h, 6, axis=1)


def cube_list2h(cube_list):
    assert len(cube_list) == 6
    assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6
    return np.concatenate(cube_list, axis=1)


def cube_h2dict(cube_h):
    cube_list = cube_h2list(cube_h)
    return dict([(k, cube_list[i])
                 for i, k in enumerate(['F', 'R', 'B', 'L', 'U', 'D'])])


def cube_dict2h(cube_dict, face_k=['F', 'R', 'B', 'L', 'U', 'D']):
    assert len(face_k) == 6
    return cube_list2h([cube_dict[k] for k in face_k])


def cube_h2dice(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    w = cube_h.shape[0]
    cube_dice = np.zeros((w * 3, w * 4, cube_h.shape[2]), dtype=cube_h.dtype)
    cube_list = cube_h2list(cube_h)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_list[i]
        if i in [1, 2]:
            face = np.flip(face, axis=1)
        if i == 4:
            face = np.flip(face, axis=0)
        cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w] = face
    return cube_dice


def cube_dice2h(cube_dice):
    w = cube_dice.shape[0] // 3
    assert cube_dice.shape[0] == w * 3 and cube_dice.shape[1] == w * 4
    cube_h = np.zeros((w, w * 6, cube_dice.shape[2]), dtype=cube_dice.dtype)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w]
        if i in [1, 2]:
            face = np.flip(face, axis=1)
        if i == 4:
            face = np.flip(face, axis=0)
        cube_h[:, i*w:(i+1)*w] = face
    return cube_h


def rotation_matrix(rad, ax):
    ax = np.array(ax)
    assert len(ax.shape) == 1 and ax.shape[0] == 3
    ax = ax / np.sqrt((ax**2).sum())
    R = np.diag([np.cos(rad)] * 3)
    R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))

    ax = ax * np.sin(rad)
    R = R + np.array([[0, -ax[2], ax[1]],
                      [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])

    return R

def sphere2UnitVector(sequence):
    UVec = np.zeros([sequence.shape[0], 3])
    UVec[:, 0] = np.cos(sequence[:,1]) * np.cos(sequence[:,0])
    UVec[:, 1] = np.cos(sequence[:,1]) * np.sin(sequence[:,0])
    UVec[:, 2] = np.sin(sequence[:,1])

    return UVec

# Helper functions
def cond_mkdir(path):
    '''Helper function to create a directory if it doesn't exist already.'''
    if not os.path.exists(path):
        os.makedirs(path)


def gnomonic2lat_lon(x_y_coords, fov_vert_hor, center_lat_lon):
    '''
    Converts gnomonic (x, y) coordinates to (latitude, longitude) coordinates.

    x_y_coords: numpy array of floats of shape (num_coords, 2)
    fov_vert_hor: tuple of vertical, horizontal field of view in degree
    center_lat_lon: The (lat, lon) coordinates of the center of the viewport that the x_y_coords are referencing.
    '''
    sphere_radius_lon = 1. / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
    sphere_radius_lat = 1. / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

    x, y = x_y_coords[:, 0], x_y_coords[:, 1]

    x_y_hom = np.column_stack([x.ravel(), y.ravel(), np.ones(len(x.ravel()))])

    K_inv = np.zeros((3, 3))
    K_inv[0, 0] = 1.0 / sphere_radius_lon
    K_inv[1, 1] = 1.0 / sphere_radius_lat
    K_inv[0, 2] = -1. / (2.0 * sphere_radius_lon)
    K_inv[1, 2] = -1. / (2.0 * sphere_radius_lat)
    K_inv[2, 2] = 1.0

    R_lat = np.zeros((3, 3))
    R_lat[0, 0] = 1.0
    R_lat[1, 1] = np.cos(np.radians(-center_lat_lon[0]))
    R_lat[2, 2] = R_lat[1, 1]
    R_lat[1, 2] = -1.0 * np.sin(np.radians(-center_lat_lon[0]))
    R_lat[2, 1] = -1.0 * R_lat[1, 2]

    R_lon = np.zeros((3, 3))
    R_lon[2, 2] = 1.0
    R_lon[0, 0] = np.cos(np.radians(-center_lat_lon[1]))
    R_lon[1, 1] = R_lon[0, 0]
    R_lon[0, 1] = - np.sin(np.radians(-center_lat_lon[1]))
    R_lon[1, 0] = - R_lon[0, 1]

    R_full = np.matmul(R_lon, R_lat)

    dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(1, 3, 3) * x_y_hom.reshape(-1, 1, 3), axis=2)

    sphere_points = dot_prod / np.linalg.norm(dot_prod, axis=1, keepdims=True)

    lat = np.degrees(np.arccos(sphere_points[:, 2]))
    lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

    lat_lon = np.column_stack([lat, lon])
    lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

    return lat_lon


def angle2img(lat_lon_array, img_height_width):
    '''
    Convertes an array of latitude, longitude coordinates to image coordinates with range (0, height) x (0, width)
    '''
    return lat_lon_array / np.array([180., 360.]).reshape(1, 2) * np.array(img_height_width).reshape(1, 2)


def stitch2video(video_name, frame_dir, fps=30., print_output=False):
    '''
    Uses ffmpeg to stitch together a bunch of frames to a video. frame_dir has to be an absolute path.
    '''
    framename_format_string = os.path.join(frame_dir, "%06d.png")
    ffmpeg_cmd = "ffmpeg -r %d -f image2 -s 1920x1080 -i %s -vcodec libx264 -crf 25  %s -y" % \
                 (fps, framename_format_string, video_name)
    args = shlex.split(ffmpeg_cmd)

    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()

    if print_output:
        pprint(output)
        pprint(err)


def get_head_salmap(list_of_runs, height_width=(720, 1440)):
    '''Computes head saliency maps.'''
    thresh = 37.196
    all_head_velos = []
    all_head_lat_lons = []

    for run in list_of_runs:
        all_head_velos.append(run['ang_head_velo'])
        all_head_lat_lons.append(run['headLatLon'])

    head_velos = np.vstack(all_head_velos)
    head_lat_lons = np.vstack(all_head_lat_lons)

    fixation_idcs = head_velos[:, 1] < thresh
    fix_lat_lons = head_lat_lons[fixation_idcs]
    norm_fix_coords = fix_lat_lons / np.array([180, 360])

    # Get this cubemap's gaze salmap
    salmap = salmap_from_norm_coords(norm_fix_coords, sigma=19.0, height_width=height_width)

    return salmap


def overlay_image_salmap(img_path, salmap):
    '''Overlays an image with a saliency map.'''
    image = matplotlib.image.imread(img_path).astype(float)[:, :, :3]
    salmap_resized = cv2.resize(salmap, (image.shape[1], image.shape[0])).astype(float)

    fig, ax = plt.subplots(frameon=False)
    fig.set_size_inches(16, 8)

    ax.imshow(image)
    ax.imshow(salmap_resized, cmap=plt.cm.jet, alpha=0.4)

    ax.axis('tight')
    ax.axis('off')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    return fig, ax


def filter_starting_point(list_of_runs, threshold_deg=20.0):
    '''Filters all measurements of a run that are at the
    beginning and in a vicinity of threshold_deg around the starting coordinates.'''
    cleaned_runs = []
    for run in list_of_runs:
        init_starting_point_lon = run['gaze_lat_lon'][0, 1]
        outside_bool = np.absolute(init_starting_point_lon - run['gaze_lat_lon'][:, 1]) > threshold_deg
        if np.any(outside_bool):
            first_left = np.amin(np.where(outside_bool)[0])
            cleaned_runs.append({key: value[first_left:] for key, value in run.iteritems()})
        else:
            cleaned_runs.append(run)
    return cleaned_runs


def get_data_matrix(list_of_runs, column_set='vr'):
    '''Joins specified columns of a list of runs to a single data matrix
    gaze_lat_lon_offset_lon'''
    if column_set == 'vr':
        columns = ['gaze_lat_lon_offset', 'gaze_fixations_single', 'ang_gaze_offset_velo', 'ang_head_velo',
                   'ang_gaze_velo', 'headTilt', 'gaze_fixations_ind']
    else:
        columns = ['gaze_lat_lon_offset', 'gaze_fixations_single', 'ang_gaze_offset_velo', 'ang_head_velo',
                   'ang_gaze_velo', 'gaze_fixations_ind']

    data = []
    column_names = []
    for column in columns:
        array_list = []

        for run in list_of_runs:
            array_list.append(np.squeeze(np.array(run[column])))

        if len(array_list[0].shape) == 1:
            joined_array = np.concatenate(array_list)

            data.append(joined_array)
            column_names.append(column)
        else:
            joined_array = np.vstack(array_list)

            data.append(joined_array[:, 0])
            data.append(joined_array[:, 1])
            column_names.append(column + '_lat')
            column_names.append(column + '_lon')

    return np.column_stack(data), column_names


def reject_gauss_outliers(data, column_names, m=2, take_absolute=True):
    '''Function to reject the outliers visible in above plot. Assumes a fundamentally normal distribution,
    which is close to what we see in the data.'''
    if take_absolute:
        data_ = np.absolute(data)
    else:
        data_ = np.copy(data)

    data_ = np.ma.masked_invalid(data_)

    col_means = np.nanmean(data_, axis=0, keepdims=True)
    centered_data = data_ - col_means
    column_stds = np.nanstd(data_, axis=0)

    bad_values_bool = np.absolute(centered_data) > m * column_stds
    bad_values = np.where(bad_values_bool)

    cleaned_data = np.copy(data)
    cleaned_data[bad_values] = np.take(col_means, bad_values[1])

    print("Rejection percentages:")
    for i in range(bad_values_bool.shape[1]):
        print("Column %s: %0.4f" % (column_names[i],
                                    float(np.sum(bad_values_bool[:, i], axis=0)) / bad_values_bool.shape[0]))

    return cleaned_data, col_means, column_stds


def plot_with_viewport(img, viewport_coords, out_path):
    viewport_coords_resh = viewport_coords.reshape(800, 800, 2)
    upper_line = viewport_coords_resh[0, :, :]
    lower_line = viewport_coords_resh[-1, :, :]
    right_line = viewport_coords_resh[:, -1, :]
    left_line = viewport_coords_resh[:, 0, :]

    lines = [upper_line, lower_line, right_line, left_line]

    split_lines = []
    for line in lines:
        diff = np.diff(line, axis=0)
        wrap_idcs = np.where(np.abs(diff) > 10)[0]
        if not len(wrap_idcs):
            split_lines.append(line)
        else:
            split_lines.append(line[:wrap_idcs[0] + 1])
            split_lines.append(line[wrap_idcs[0] + 1:])

    fig, ax = plt.subplots(frameon=False)
    fig.set_size_inches(48, 24)
    ax.imshow(img)

    for line in split_lines:
        ax.plot(line[:, 1], line[:, 0], color='black', linewidth=10)

    for line in split_lines:
        ax.plot(line[:, 1], line[:, 0], color='white', linewidth=8)

    ax.axis('tight')
    ax.axis('off')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    fig.clf()
    plt.close()


def get_gnomonic_hom(center_lat_lon, origin_image, height_width, fov_vert_hor=(60.0, 60.0)):
    '''Extracts a gnomonic viewport with height_width from origin_image
    at center_lat_lon with field of view fov_vert_hor.
    '''
    org_height_width, _ = origin_image.shape[:2], origin_image.shape[-1]
    height, width = height_width

    if len(origin_image.shape) == 3:
        result_image = np.zeros((height, width, 3))
    else:
        result_image = np.zeros((height, width))

    sphere_radius_lon = width / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
    sphere_radius_lat = height / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

    y, x = np.mgrid[0:height, 0:width]
    x_y_hom = np.column_stack([x.ravel(), y.ravel(), np.ones(len(x.ravel()))])

    K_inv = np.zeros((3, 3))
    K_inv[0, 0] = 1.0 / sphere_radius_lon
    K_inv[1, 1] = 1.0 / sphere_radius_lat
    K_inv[0, 2] = -width / (2.0 * sphere_radius_lon)
    K_inv[1, 2] = -height / (2.0 * sphere_radius_lat)
    K_inv[2, 2] = 1.0

    R_lat = np.zeros((3, 3))
    R_lat[0, 0] = 1.0
    R_lat[1, 1] = np.cos(np.radians(-center_lat_lon[0]))
    R_lat[2, 2] = R_lat[1, 1]
    R_lat[1, 2] = -1.0 * np.sin(np.radians(-center_lat_lon[0]))
    R_lat[2, 1] = -1.0 * R_lat[1, 2]

    R_lon = np.zeros((3, 3))
    R_lon[2, 2] = 1.0
    R_lon[0, 0] = np.cos(np.radians(-center_lat_lon[1]))
    R_lon[1, 1] = R_lon[0, 0]
    R_lon[0, 1] = - np.sin(np.radians(-center_lat_lon[1]))
    R_lon[1, 0] = - R_lon[0, 1]

    R_full = np.matmul(R_lon, R_lat)

    dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(1, 3, 3) * x_y_hom.reshape(-1, 1, 3), axis=2)

    sphere_points = dot_prod / np.linalg.norm(dot_prod, axis=1, keepdims=True)

    lat = np.degrees(np.arccos(sphere_points[:, 2]))
    lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

    lat_lon = np.column_stack([lat, lon])
    lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

    org_img_y_x = lat_lon / np.array([180.0, 360.0]) * np.array(org_height_width)
    org_img_y_x = np.clip(org_img_y_x, 0.0, np.array(org_height_width).reshape(1, 2) - 1.0).astype(int)
    org_img_y_x = org_img_y_x.astype(int)

    if len(origin_image.shape) == 3:
        result_image[x_y_hom[:, 1].astype(int), x_y_hom[:, 0].astype(int), :] = origin_image[org_img_y_x[:, 0],
                                                                                org_img_y_x[:, 1], :]
    else:
        result_image[x_y_hom[:, 1].astype(int), x_y_hom[:, 0].astype(int)] = origin_image[org_img_y_x[:, 0],
                                                                                          org_img_y_x[:, 1]]
    return result_image.astype(float), org_img_y_x


def get_pano_no(pano_no, undersample=3):
    '''Helper function to load the panorama oc scene pano_no and downsample it by a factor of undersample.'''
    path = os.path.join(settings.IMG_PATH, 'cubemap_%04d.png' % pano_no)
    pano = mplimg.imread(path)
    pano = pano[::undersample, ::undersample, :3]
    return pano


def extract_vid_frames(vid_path, target_dir, fps=15, print_output=False):
    '''Extract frames from a video, wrapping ffmpeg.'''
    vid_name = os.path.basename(vid_path)
    vid_name_no_ext = os.path.splitext(vid_name)[0]

    ffmpeg_template = 'ffmpeg -i {} -vf fps={} {}'

    ffmpeg_cmd = ffmpeg_template.format(vid_path, fps, os.path.join(target_dir, '%06d.png'))
    args = shlex.split(ffmpeg_cmd)

    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()

    if print_output:
        pprint(output)
        pprint(err)


def interpolate_nan_rows(array, bad_rows_bool):
    '''Interpolates nan rows in an array.'''
    if True not in bad_rows_bool.astype(bool): return array

    good_rows = np.where(np.logical_not(bad_rows_bool))[0]

    # Since extrapolation is a bad idea, we identify the low-confidence indices that are outside the interpolatable range.
    low_non_nan, high_non_nan = np.amin(good_rows), np.amax(good_rows)
    interp_idcs = np.copy(bad_rows_bool)
    interp_idcs[high_non_nan:] = False
    interp_idcs[:low_non_nan] = False

    extra_idcs = np.logical_and(bad_rows_bool,
                                np.logical_or(np.arange(len(bad_rows_bool)) <= low_non_nan,
                                              np.arange(len(bad_rows_bool)) >= high_non_nan))

    interp_func = scipy.interpolate.interp1d(good_rows,
                                             np.take(array, good_rows, axis=0),
                                             kind='linear',
                                             axis=0)
    array[interp_idcs] = interp_func(np.where(interp_idcs)[0])
    # The indices outside the interpolatable range are set to the mean of the series.
    array[extra_idcs] = np.mean(array[good_rows], axis=0)

    return array