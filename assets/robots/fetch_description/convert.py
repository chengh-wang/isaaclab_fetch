import pymeshlab

input_dir = 'src/momafetch_description/meshes/'
output_dir = 'src/momafetch_description/meshes-mujoco/'

ms = pymeshlab.MeshSet()


convert_list = ['base_link','elbow_flex_link','estop_link','forearm_roll_link','gripper_link','head_pan_link','head_tilt_link','shoulder_lift_link','shoulder_pan_link','torso_fixed_link','torso_lift_link',
                'upperarm_roll_link','wrist_flex_link','wrist_roll_link']

for link in convert_list:
    ms.load_new_mesh(input_dir+link+'.dae')
    ms.save_current_mesh(output_dir+link+'.stl',binary=True)