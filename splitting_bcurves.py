import numpy as np
import torch

def split_bezier(control_points, tValues):
    """
    split_bezier splits a Bézier curve at multiple parameter values.
    Inputs:
      - control_points: An n x 2 matrix where each row represents a control point.
      - t_values: A vector of parameter values at which to split the curve.
    Outputs:
      - segments: A list containing the control points of the resulting curve segments.
    """

    # Sort the parameter values to ensure correct sequential splitting
    tValues = np.sort(tValues)

    # Initialize a list to hold the control points of the split segments
    segments = []
    remaining_points = torch.clone(control_points)

    for k in range(len(tValues)):
        t=tValues[k]
        # Initialize matrices for left and right segments
        left = torch.zeros_like(remaining_points)
        right = torch.zeros_like(remaining_points)

        # Initialize points for de Casteljau's algorithm
        points = torch.clone(remaining_points)
        n = points.shape[0]

        for r in range(0,n-2):
            left[r, :] = points[0, :]
            for i in range(0,n - r-1):
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

controlPoints = torch.tensor([[120.0,  30.0], # base
                              [150.0,  60.0], # control point
                              [ 90.0, 198.0], # control point
                              [ 60.0, 218.0]    

])
# Define the parameter values at which to split the curve
tValues = [0.1,0.3, 0.5, 0.7, 0.9]


segments = (split_bezier(controlPoints, tValues))

i=0

for segment in segments:
    i+=1
    print (f"Segment {i}:")
    print (segment)
