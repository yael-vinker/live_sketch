import torch
import pydiffvg
import os
import argparse

def get_center_of_mass(shapes):
    all_points = []
    for shape in shapes:
        all_points.append(shape.points)
    points_vars = torch.vstack(all_points)
    center = points_vars.mean(dim=0)
    return center

def resize_canvas(svg_path, target_width=256, target_height=256, obj_scale=1, target_stroke_width=1.5):
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg_path)
    width_factor_center = canvas_width / target_width
    height_factor_center = canvas_height / target_height
    if canvas_width > canvas_height:
        width_factor = canvas_width / target_width
        height_factor = width_factor
    else:
        height_factor = canvas_height / target_height # preserve height
        width_factor = height_factor
    center_of_orig_sketch = get_center_of_mass(shapes_init)
    
    new_canvas_orig = torch.tensor((target_width / 2, target_height / 2))
    new_center = (center_of_orig_sketch[0] / width_factor_center, center_of_orig_sketch[1] / height_factor_center)
    for j,path in enumerate(shapes_init):
        # locate in (0,0)
        path.points[:, 0] -= center_of_orig_sketch[0]
        path.points[:, 1] -= center_of_orig_sketch[1]
        
        # fix height and width
        path.points[:, 1] /= height_factor
        path.points[:, 0] /= width_factor
        path.points[:, 1] *= obj_scale
        path.points[:, 0] *= obj_scale
        
        # back to original location
        path.points[:, 0] += new_center[0]
        path.points[:, 1] += new_center[1]
        
        path.stroke_width = torch.tensor((target_stroke_width))   
        
    scene_args = pydiffvg.RenderFunction.serialize_scene(target_width, target_height, shapes_init, shape_groups_init)
    render = pydiffvg.RenderFunction.apply
    pydiffvg.save_svg(f"svg_input/{os.path.splitext(os.path.basename(svg_path))[0]}_scaled.svg", target_width, target_height, shapes_init, shape_groups_init)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="svg_input/ballerina2.svg", help="path to sketch SVG file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pydiffvg.set_device(device) 
    resize_canvas(args.target, target_width=256, target_height=256, obj_scale=1, target_stroke_width=1.5)