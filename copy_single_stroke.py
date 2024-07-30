import pydiffvg
import torch
import skimage
import numpy as np

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256, 256
num_control_points = torch.tensor([2])
points = torch.tensor([[120.0,  30.0], # base
                       [150.0,  60.0], # control point
                       [ 90.0, 198.0], # control point
                       [ 60.0, 218.0]]) # base
path = pydiffvg.Path(num_control_points = num_control_points,
                     points = points,
                     is_closed = False,
                     stroke_width = torch.tensor(5.0))
shapes = [path]
path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.0, 0.0, 0.0, 0.0]),
                                 stroke_color = torch.tensor([0.6, 0.3, 0.6, 0.8]))
shape_groups = [path_group]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)



render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/single_stroke/target.png', gamma=2.2)
target = img.clone()

# Visibility function
def visibility_function(t):
    mod_t = t % 0.2
    if mod_t < 0.1:
        return 1 - 10 * mod_t
    else:
        return 0

# Predefined zeros of the visibility function
zeros = [0.1, 0.3, 0.5, 0.7, 0.9]

# Split Bezier curve at the zeros of the visibility function
def split_bezier(control_points, t_values):
    """
    split_bezier splits a Bézier curve at multiple parameter values.
    Inputs:
      - control_points: An n x 2 matrix where each row represents a control point.
      - t_values: A vector of parameter values at which to split the curve.
    Outputs:
      - segments: A list containing the control points of the resulting curve segments.
    """
    
    # Sort the parameter values to ensure correct sequential splitting
    t_values = np.sort(t_values)
    
    # Initialize a list to hold the control points of the split segments
    segments = []
    remaining_points = control_points.copy()
    
    for t in t_values:
        # Initialize matrices for left and right segments
        left = np.zeros_like(remaining_points)
        right = np.zeros_like(remaining_points)
        
        # Initialize points for de Casteljau's algorithm
        points = remaining_points.copy()
        n = points.shape[0]
        
        for r in range(n - 1):
            left[r, :] = points[0, :]
            for i in range(n - r):
                points[i, :] = (1 - t) * points[i, :] + t * points[i + 1, :]
        
        left[n - 1, :] = points[0, :]
        right[0:n - 1, :] = points[1:n, :]
        right[n - 1, :] = points[n - 1, :]
        
        # Store the left segment and update the remaining points
        segments.append(left) 
        remaining_points = right
    
    # Add the final segment
    segments.append(remaining_points)
    
    return segments
# Apply the visibility function to the path points
#while t =! 0:
# for t in visibility_function(t):
    #if visibility_function(t) != -1:
        #continue
    #else:
        #break
      
# Update the path with visible points and render (need to figure out new points for the following )
points_one = torch.tensor([[120.0,  30.0], # base
                       [150.0,  60.0], # control point
                       [ 90.0, 198.0], # control point
                       [ 60.0, 218.0]]) # base
path_one = pydiffvg.Path(num_control_points = new_num_control_points,
                     points = points,
                     is_closed = False,
                     stroke_width = torch.tensor(5.0))

points_two = torch.tensor([[ 90.0, 198.0], # base
                       [150.0,  60.0], # control point
                       [ 60.0, 218.0]]) # base
path_two = pydiffvg.Path(num_control_points = new_num_control_points,
                     points = test_points,
                     is_closed = False,
                     stroke_width = torch.tensor(5.0))

shapes = [path_one]
shapes.extend([path_two])
path_group_one = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.0, 0.0, 0.0, 0.0]),
                                 stroke_color = torch.tensor([0.6, 0.3, 0.6, 0.8]))
path_group_two = pydiffvg.ShapeGroup(shape_ids = torch.tensor([1]),
                                 fill_color = torch.tensor([0.0, 0.0, 0.0, 0.0]),
                                 stroke_color = torch.tensor([0.4, 0.7, 0.5, 0.5]))
shape_groups = [path_group_one]
shape_groups.extend([path_group_two]) 

scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)



render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), 'results/single_stroke/split.png', gamma=2.2)
split = img.clone()



# Move the path to produce initial guess
# normalize points for easier learning rate
points_n = torch.tensor([[100.0/256.0,  40.0/256.0], # base
                         [155.0/256.0,  65.0/256.0], # control point
                         [100.0/256.0, 180.0/256.0], # control point
                         [ 65.0/256.0, 238.0/256.0]], # base
                        requires_grad = True) 
stroke_color = torch.tensor([0.4, 0.7, 0.5, 0.5], requires_grad=True)
stroke_width_n = torch.tensor(10.0 / 100.0, requires_grad=True)
path.points = points_n * 256
path.stroke_width = stroke_width_n * 100
path_group.stroke_color = stroke_color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), 'results/single_stroke/init.png', gamma=2.2)

# Optimize
optimizer = torch.optim.Adam([points_n, stroke_color, stroke_width_n], lr=1e-2)
# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    path.points = points_n * 256
    path.stroke_width = stroke_width_n * 100
    path_group.stroke_color = stroke_color
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(256,   # width
                 256,   # height
                 2,     # num_samples_x
                 2,     # num_samples_y
                 t+1,   # seed
                 None, # background_image
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/single_stroke/iter_{}.png'.format(t), gamma=2.2)
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('points_n.grad:', points_n.grad)
    print('stroke_color.grad:', stroke_color.grad)
    print('stroke_width.grad:', stroke_width_n.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current params.
    print('points:', path.points)
    print('stroke_color:', path_group.stroke_color)
    print('stroke_width:', path.stroke_width)

# Render the final result.
path.points = points_n * 256
path.stroke_width = stroke_width_n * 100
path_group.stroke_color = stroke_color
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256,   # width
             256,   # height
             2,     # num_samples_x
             2,     # num_samples_y
             202,    # seed
             None, # background_image
             *scene_args)
# Save the images and differences.
pydiffvg.imwrite(img.cpu(), 'results/single_stroke/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/single_stroke/iter_%d.png", "-vb", "20M",
    "results/single_stroke/out.mp4"])
