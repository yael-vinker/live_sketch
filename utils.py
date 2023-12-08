import torch
import pydiffvg
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import cairosvg

# ==================================
# ====== video realted utils =======
# ==================================
def frames_to_vid(video_frames, output_vid_path):
    """
    Saves an mp4 file from the given frames
    """
    writer = imageio.get_writer(output_vid_path, fps=8)
    for im in video_frames:
        writer.append_data(im)
    writer.close()

def render_frames_to_tensor(frames_shapes, frames_shapes_grous, w, h, render, device):
    """
    Given a list with the points parameters, render them frame by frame and return a tensor of the rasterized frames ([16, 256, 256, 3])
    """
    frames_init = []
    for i in range(len(frames_shapes)):
        shapes = frames_shapes[i]
        shape_groups = frames_shapes_grous[i]
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        cur_im = render(w, h, 2, 2, 0, None, *scene_args)
    
        cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
               torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=device) * (1 - cur_im[:, :, 3:4])
        cur_im = cur_im[:, :, :3]
        frames_init.append(cur_im)
    return torch.stack(frames_init)

def save_mp4_from_tensor(frames_tensor, output_vid_path):
    # input is a [16, 256, 256, 3] video
    frames_copy = frames_tensor.clone()
    frames_output = []
    for i in range(frames_copy.shape[0]):
        cur_im = frames_copy[i]
        cur_im = cur_im[:, :, :3].detach().cpu().numpy()
        cur_im = (cur_im * 255).astype(np.uint8)
        frames_output.append(cur_im)
    frames_to_vid(frames_output, output_vid_path=output_vid_path)
    
def save_vid_svg(frames_svg, output_folder, step, w, h):
    if not os.path.exists(f"{output_folder}/svg_step{step}"):
        os.mkdir(f"{output_folder}/svg_step{step}")
    for i in range(len(frames_svg)):
        pydiffvg.save_svg(f"{output_folder}/svg_step{step}/frame{i:03d}.svg", w, h, frames_svg[i][0], frames_svg[i][1])

def svg_to_png(path_to_svg_files, dest_path):
    svgs = sorted(os.listdir(path_to_svg_files))
    filenames = [k for k in svgs if "svg" in k]
    for filename in filenames:        
        dest_path_ = f"{dest_path}/{os.path.splitext(filename)[0]}.png"
        cairosvg.svg2png(url=f"{path_to_svg_files}/{filename}", write_to=dest_path_, scale=4, background_color="white")
  
def save_gif_from_pngs(path_to_png_files, gif_dest_path):
    pngs = sorted(os.listdir(path_to_png_files))
    filenames = [k for k in pngs if "png" in k]
    images = []
    for filename in filenames:
        im = imageio.imread(f"{path_to_png_files}/{filename}")
        images.append(im)
    imageio.mimsave(f"{gif_dest_path}", images, 'GIF', loop=4, fps=8)

def save_hq_video(path_to_outputs, iter_=1000):
    dest_path_png = f"{path_to_outputs}/png_files_ite{iter_}"
    os.makedirs(dest_path_png, exist_ok=True)

    svg_to_png(f"{path_to_outputs}/svg_logs/svg_step{iter_}", dest_path_png)

    gif_dest_path = f"{path_to_outputs}/HQ_gif.gif"
    save_gif_from_pngs(dest_path_png, gif_dest_path)
    print(f"GIF saved to [{gif_dest_path}]")


def normalize_tensor(tensor: torch.Tensor, canvas_size: int = 256):
    range_value = float(canvas_size)# / 2
    normalized_tensor = tensor / range_value
    return normalized_tensor

def get_caption(target):
    # captions used in the main paper
    files_to_captions = {
        "penguin": "The penguin is shuffling along the ice terrain, taking deliberate and cautious step with its flippers outstretched to maintain balance.",
        "dancers3": "The two dancers are passionately dancing the Cha-Cha, their bodies moving in sync with the infectious Latin rhythm.",
        "box2": "The boxer ducking and weaving to avoid his opponent's punches, and to punch him back.",
        "runner1": "The runner runs with rhythmic leg strides and synchronized arm swing propelling them forward while maintaining balance.",
        "sax_play1": "The jazz saxophonist performs on stage with a rhythmic sway, his upper body sways subtly to the rhythm of the music.",
        "yoga4": "The woman gracefully flowed through her yoga routine, moving from one pose to another with fluid and controlled motion.",
        "yoga3": "The woman gracefully flowed through her yoga routine, moving from one pose to another with fluid and controlled motion.",
        "fish": "The goldenfish is gracefully moving through the water, its fins and tail fin gently propelling it forward with effortless agility.",
        "crabdraw": "The crab scuttled sideways along the sandy beach, its pincers raised in a defensive stance.",
        "ballerina2": "The ballerina is dancing.",
        "ballet1_16": "The ballerina is dancing.",
        "ballet2_16": "The ballerina is dancing.",
        "horse": "A galloping horse.",
        "snake3_16": "The hypnotized cobra snake sways rhythmically, mesmerized, with a gentle and trance-like side-to-side motion, its hood slightly flared, under the influence of an entrancing stimulus.",
        "snake5_8": "The hypnotized cobra snake sways rhythmically, mesmerized, with a gentle and trance-like side-to-side motion, its hood slightly flared, under the influence of an entrancing stimulus.",
        "lizard1_16": "The lizard moves with a sinuous, undulating motion, gliding smoothly over surfaces using its agile limbs and tail for balance and propulsion.",
        "eagle_16": "The eagle soars majestically, with powerful wing beats and effortless glides, displaying precise control and keen vision as it maneuvers gracefully through the sky.",
        "kangaroo1_16": "The kangaroo is jumping, crouching, gathering energy, then propelling itself forward with incredible force and agility, tucking its legs mid-air before landing gracefully on its hind legs with remarkable balance and coordination.",
        "kangaroo2_16": "The kangaroo is jumping, crouching, gathering energy, then propelling itself forward with incredible force and agility, tucking its legs mid-air before landing gracefully on its hind legs with remarkable balance and coordination.",
        "kangaroo1_8": "The kangaroo is jumping, crouching, gathering energy, then propelling itself forward with incredible force and agility, tucking its legs mid-air before landing gracefully on its hind legs with remarkable balance and coordination.",
        "boat2": "The man sailing the boat, his hands deftly manipulate the oars, while his body shifts subtly to maintain balance, while his boat moves foraward in the river.",
        "cat5_8": "The cat is playing.",
        "cat6_8": "The cat is playing.",
        "cat2_16": "The cat is playing.",
        "cat2_8":"The cat is playing.",
        "cat3_16":"The cat is playing.",
        "cat4_16":"The cat is playing.",
        "biking": "The biker is pedaling, each leg pumping up and down as the wheels of the bicycle spin rapidly, propelling them forward.",
        "cheetah_2": "The cheetah is running at high speeds in pursuit of prey.",
        "hummingbird1": "A hummingbird hovers in mid-air and sucks nectar from a flower.",
        "hummingbird0": "A hummingbird hovers in mid-air and sucks nectar from a flower.",
        "hummingbird_3": "A hummingbird hovers in mid-air and sucks nectar from a flower.",
        "dolphin_1": "A dolphin swimming and leaping out of the water.",
        "dolphin_2": "A dolphin swimming and leaping out of the water.",
        "dolphin_3": "A dolphin swimming and leaping out of the water.",
        "butterfly_1": "A butterfly fluttering its wings and flying gracefully.",
        "butterfly_3": "A butterfly fluttering its wings and flying gracefully.",
        "butterfly_2": "A butterfly fluttering its wings and flying gracefully.",
        "gazelle_0": "A gazelle galloping and jumping to escape predators.",
        "gazelle_2": "A gazelle galloping and jumping to escape predators.",
        "gazelle_3": "A gazelle galloping and jumping to escape predators.",
        "gazelle2": "A gazelle galloping and jumping to escape predators.",
        "squirrel3": "The squirrel uses its dexterous front paws to hold and manipulate nuts, displaying meticulous and deliberate motions while eating.",
        "squirrel1": "The squirrel uses its dexterous front paws to hold and manipulate nuts, displaying meticulous and deliberate motions while eating.",
        "squirrel_3": "The squirrel uses its dexterous front paws to hold and manipulate nuts, displaying meticulous and deliberate motions while eating.",
        "squirrel0": "The squirrel uses its dexterous front paws to hold and manipulate nuts, displaying meticulous and deliberate motions while eating.",
        "squirrel_2": "The squirrel uses its dexterous front paws to hold and manipulate nuts, displaying meticulous and deliberate motions while eating.",
        "gymnast_0": "A gymnast flipping, tumbling, and balancing on various apparatuses.",
        "artist_2": "A martial artist executing precise and controlled movements in different forms of martial arts.",
        "artist_3": "A martial artist executing precise and controlled movements in different forms of martial arts.",
        "skater_0": "A figure skater gliding, spinning, and performing jumps on ice skates.",
        "skater_2": "A figure skater gliding, spinning, and performing jumps on ice skates.",
        "surfer3": "A surfer riding and maneuvering on waves on a surfboard.",
        "surfer1": "A surfer riding and maneuvering on waves on a surfboard.",
        "surfer0": "A surfer riding and maneuvering on waves on a surfboard.",
        "basketball_a0": "A basketball player dribbling and passing while playing basketball.",
        "basketball_a2": "A basketball player dribbling and passing while playing basketball.",
        "basketball_a3": "A basketball player dribbling and passing while playing basketball.",
        "basketball_a4": "A basketball player dribbling and passing while playing basketball.",
        "basketball2_0": "A basketball player dribbling and passing while playing basketball.",
        "flag_1": "A waving flag fluttering and rippling in the wind.",
        "flag_3": "A waving flag fluttering and rippling in the wind.",
        "parachute_2": "A parachute descending slowly and gracefully after being deployed.",
        "toy_car_1": "A wind-up toy car, moving forward or backward when wound up and released.",
        "toy_car_3": "A wind-up toy car, moving forward or backward when wound up and released.",
        "windmill_3": "A windmill spinning its blades in the wind to generate energy.",
        "ceiling_fan_0": "A ceiling fan rotating blades to circulate air in a room.",
        "ceiling_fan_2": "A ceiling fan rotating blades to circulate air in a room.",
        "ceiling_fan_4": "A ceiling fan rotating blades to circulate air in a room.",
        "clock2": "A clock hands ticking and rotating to indicate time on a clock face.",
        "wine2": "The wine in the wine glass sways from side to side.",
        "wine1": "The wine in the wine glass sways from side to side.",
        "plane": "The airplane moves swiftly and steadily through the air.",
        "sapceship1": "The spaceship accelerates rapidly during takeoff, utilizing powerful rocket engines.",
        "flower2": "The flower is moving and growing, swaying gently from side to side.",
        "flower1": "The flower is moving and growing, swaying gently from side to side.",
        "flower_a1": "The flower is moving and growing, swaying gently from side to side.",
        "flower_a2": "The flower is moving and growing, swaying gently from side to side.",
        "clock_a1": "A clock hands ticking and rotating to indicate time.",
        "clock_a2": "A clock hands ticking and rotating to indicate time."
        }
    return files_to_captions[os.path.basename(target).replace("_scaled1", "")]