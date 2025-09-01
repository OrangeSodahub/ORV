import torch
import math
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer



def apply_depth_colormap(gray, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    Input:
        gray: gray image, tensor/numpy, (H, W)
    Output:
        depth: (3, H, W), tensor
    """
    if type(gray) is not np.ndarray:
        gray = gray.detach().cpu().numpy().astype(np.float32)
    gray = gray.squeeze()
    assert len(gray.shape) == 2
    x = np.nan_to_num(gray)  # change nan to 0
    if minmax is None:
        mi = np.min(x)  # get minimum positive value
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    # TODO
    x = 1 - x  # reverse the colormap
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(cv2.applyColorMap(x, cmap))
    x = T.ToTensor()(x)  # (3, H, W)
    return x


def apply_semantic_colormap(semantic):
    """
    Input:
        semantic: semantic image, tensor/numpy, (N, H, W)
    Output:
        depth: (3, H, W), tensor
    """

    color_id = np.zeros((20, 3), dtype=np.uint8)
    color_id[0, :] = [255, 120, 50]
    color_id[1, :] = [255, 192, 203]
    color_id[2, :] = [255, 255, 0]
    color_id[3, :] = [0, 150, 245]
    color_id[4, :] = [0, 255, 255]
    color_id[5, :] = [255, 127, 0]
    color_id[6, :] = [255, 0, 0]
    color_id[7, :] = [255, 240, 150]
    color_id[8, :] = [135, 60, 0]
    color_id[9, :] = [160, 32, 240]
    color_id[10, :] = [255, 0, 255]
    color_id[11, :] = [139, 137, 137]
    color_id[12, :] = [75, 0, 75]
    color_id[13, :] = [150, 240, 80]
    color_id[14, :] = [230, 230, 250]
    color_id[15, :] = [0, 175, 0]
    color_id[16, :] = [0, 255, 127]
    color_id[17, :] = [222, 155, 161]
    color_id[18, :] = [140, 62, 69]
    color_id[19, :] = [227, 164, 30]

    if semantic.shape[0] != 1:
        semantic = torch.max(semantic, dim=0)[1].squeeze()
    else:
        semantic = semantic.squeeze()

    x = torch.zeros((3, semantic.shape[0], semantic.shape[1]), dtype=torch.float)
    for i in range(12):
        x[0][semantic == i] = color_id[i][0]
        x[1][semantic == i] = color_id[i][1]
        x[2][semantic == i] = color_id[i][2]

    return x / 255.0


def create_full_center_coords(range, dim):

    shape = torch.from_numpy(((range[1] - range[0]) / dim)).long()
    x_range = range[:, 0]
    y_range = range[:, 1]
    z_range = range[:, 2]

    x = torch.linspace(x_range[0], x_range[1], shape[0]).view(-1, 1, 1).expand(*shape)
    y = torch.linspace(y_range[0], y_range[1], shape[1]).view(1, -1, 1).expand(*shape)
    z = torch.linspace(z_range[0], z_range[1], shape[2]).view(1, 1, -1).expand(*shape)

    center_coords = torch.stack((x, y, z), dim=-1)

    return center_coords


def render(extrinsics, intrinsics, image_shape,
           pts_xyz, pts_rgb, feat, rotations, scales, opacity, bg_color):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    height, width = image_shape

    # Set up rasterization configuration
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    FovX = focal2fov(fx, width)
    FovY = focal2fov(fy, height)
    tan_fov_x = math.tan(FovX * 0.5)
    tan_fov_y = math.tan(FovY * 0.5)

    extrinsics = torch.inverse(extrinsics) # w2c

    # projection_matrix = get_projection_matrix(near=0.1, far=200.0, fov_x=FovX, fov_y=FovY).transpose(0, 1).cuda()
    projection_matrix = get_projection_matrix_c(fx, fy, cx, cy, width, height, 0.1, 200.0).transpose(0, 1).cuda()
    world_view_transform = extrinsics.transpose(0, 1).cuda()
    full_projection = world_view_transform.float() @ projection_matrix

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_projection,
        sh_degree=3,
        campos=world_view_transform.inverse()[3, :3],
        prefiltered=False,
        debug=False,
        include_feature=True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_feat, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=pts_rgb,
        language_feature_precomp=feat,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return {
        'render_color': rendered_image,
        'radii': radii,
        'render_depth': rendered_depth,
        'render_alpha': rendered_alpha,
        'render_feat': rendered_feat,
    }


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def get_projection_matrix(near, far, fov_x, fov_y):
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = math.tan(0.5 * fov_x)
    tan_fov_y = math.tan(0.5 * fov_y)

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    result = torch.zeros((4, 4), dtype=torch.float32)

    result[0, 0] = 2 * near / (right - left)
    result[1, 1] = 2 * near / (top - bottom)
    result[0, 2] = (right + left) / (right - left)
    result[1, 2] = (top + bottom) / (top - bottom)
    result[3, 2] = 1
    result[2, 2] = far / (far - near)
    result[2, 3] = -(far * near) / (far - near)

    return result


def get_projection_matrix_c(fx, fy, cx, cy, W, H, znear, zfar):
    top = cy * znear / fy
    bottom = -(H - cy) * znear / fy

    right = cx * znear / fx
    left = -(W - cx) * znear / fx

    P = torch.zeros(4, 4)

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P
