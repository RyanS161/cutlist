import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ckdtree
from design import Design, visualize, AssembledComponent, Box, Cylinder
from shapeNetHelper import get_shapenet_samples, get_and_transform_partnet_meshes
import pyvista as pv
from tqdm import tqdm
import os, copy


def visualize_txt_designs(designs):
    EXAMPLE_DESIGNS = {
        "X": "3 0 0 0 0 0 -45\n3 0 0 0 0 0 45",
        "T": "5 0 0 0 0 0 0\n5 0 0 0 0 0 90",
        "H": "0 -40 0 0 0 0 0\n0  40 0 0 0 0 0\n2   0 0 0 0 0 90",
        "Chair": "0 0 0 0 0 0 0\n0 60 0 0 0 0 0\n0 0 60 0 0 0 0\n0 60 60 0 0 0 0\n4 30 20 30 0 0 0\n3 30 20 120 0 0 0\n",
        "House": "0 0 0 60 0 0 0\n0 160 0 60 0 0 0\n0 0 80 60 0 0 0\n0 160 80 60 0 0 0\n6 80 40 5 0 0 0\n6 80 40 125 0 0 0\n5 80 0 65 90 0 0\n5 80 80 65 90 0 0\n4 40 40 65 90 90 0\n4 120 40 65 90 90 0\n8 80 0 65 0 0 0\n",
        "Box": "0 10 10 5 0 0 0\n0 150 10 5 0 0 0\n0 10 70 5 0 0 0\n0 150 70 5 0 0 0\n5 80 10 2.5 0 0 0\n5 80 70 2.5 0 0 0\n2 10 40 60 0 0 0\n2 150 40 60 0 0 0\n6 80 40 117.5 90 0 0\n6 80 40 2.5 90 0 0\n",
    }
    for letter, design_txt in designs.items():
        design = Design(
            assembledComponents=[],
            bounds=np.array([[-200, 200], [-200, 200], [0, 200]]),
        )
        design.from_txt(design_txt, from_file=False)
        design.visualize_design(filename=f"designs/design_{letter}")




import numpy as np


def score_points_to_cylinder(points, cylinder_axis, cylinder_radius, cylinder_height):

    distance_from_axis = np.linalg.norm(points - (points @ cylinder_axis[:, None]) * cylinder_axis[:, None].T, axis=1)

    half_height = cylinder_height / 2
    projections = np.abs(points @ cylinder_axis)

    inside_of_cylinder = (distance_from_axis <= cylinder_radius) & (projections <= half_height)

    above_cap = (distance_from_axis <= cylinder_radius) & (projections > half_height)

    # set distances for points inside the cylinder to the minimum of distance to side or cap
    inside_distances = np.minimum(cylinder_radius - distance_from_axis, half_height - projections)
    above_cap_distances = np.abs(projections - half_height)
    outside_distances = np.sqrt((distance_from_axis - cylinder_radius) ** 2 + np.maximum(0, projections - half_height) ** 2)

    distances = np.zeros(points.shape[0])
    distances[inside_of_cylinder] = inside_distances[inside_of_cylinder]
    distances[above_cap] = above_cap_distances[above_cap]
    distances[~(inside_of_cylinder | above_cap)] = outside_distances[~(inside_of_cylinder | above_cap)]

    return np.linalg.norm(distances)


def score_points_to_box(points, box_x, box_y, box_z):

    # Idea one - minimum distance to box surface
    # Problem: boxes can be arbitrarily long
    # half_extents = np.array([box_x/2, box_y/2, box_z/2])
    # d = np.abs(points) - half_extents
    # outside = np.linalg.norm(np.maximum(d, 0), axis=1)
    # inside = np.minimum(np.max(d, axis=1), 0)
    # return np.linalg.norm(outside + np.abs(inside))
    
    # Idea two - distance to vertices
    # Penalizes larger shapes because they have vertices further away
    box_vertices = np.array([
        [box_x / 2, box_y / 2, box_z / 2],
        [box_x / 2, box_y / 2, -box_z / 2],
        [box_x / 2, -box_y / 2, box_z / 2],
        [box_x / 2, -box_y / 2, -box_z / 2],
        [-box_x / 2, box_y / 2, box_z / 2],
        [-box_x / 2, box_y / 2, -box_z / 2],
        [-box_x / 2, -box_y / 2, box_z / 2],
        [-box_x / 2, -box_y / 2, -box_z / 2],
    ])

    # calculate the distance from each point to each vertex
    distances = np.linalg.norm(points[:, None, :] - box_vertices[None, :, :], axis=2)
    return np.mean(distances.flatten())

    # Idea three - compare the dimensions with the ideal dimensions of a fitted primitive


# def ransac_for_pointcloud(sample):
#     N_SAMPLED_POINTS = 100
#     K_ITERATIONS = 250
#     SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
#     original_pc = sample["point_cloud"]
    
#     # visualize(scaled_pcs, colors=['red', 'green', 'blue', 'purple'], filename=f"designs/pointcloud_scaled_{sample['category']}_{sample['object_id']}.png")


#     best_pc, best_part, best_avg_distance = None, None, np.inf
#     for scale in tqdm(SCALES, desc=f"Scaling for {sample['category']} {sample['object_id']}"):
#         target_pc = original_pc.scale(scale, point=(0, 0, 0))
#         points = target_pc.points
#         for _ in range(K_ITERATIONS):
#             # Pick N random points
#             sampled_indices = np.random.choice(
#                 points.shape[0], N_SAMPLED_POINTS, replace=False
#             )
#             shape_id = np.random.randint(0, len(Design.PART_LIBRARY))
#             shape = Design.PART_LIBRARY[shape_id]
#             sampled_points = points[sampled_indices]

#             centroid, rotation = pca_from_points(sampled_points)

#             # Create design part aligned with principal components
#             part = AssembledComponent(
#                 part_id=shape_id,
#                 translation=centroid,
#                 rotation=rotation,
#             )

#             axis_aligned_pc = points_centered @ np.linalg.inv(eigenvectors)

#             if type(shape) is Box:
#                 box_x, box_y, box_z = shape.x_length, shape.y_length, shape.z_length
#                 avg_distance = np.mean(distance_points_to_box(axis_aligned_pc, box_x, box_y, box_z))
#             elif type(shape) is Cylinder:
#                 cylinder_axis = shape.axis
#                 cylinder_radius = shape.radius
#                 cylinder_height = shape.height
#                 avg_distance = np.average(distance_points_to_cylinder(axis_aligned_pc, cylinder_axis, cylinder_radius, cylinder_height))
#                 # avg_distance = N_SAMPLED_POINTS - np.sum(avg_distance < 5)
#             else:
#                 print("Unknown shape type")

#             # Calculate distances from points to mesh surface
#             # mesh_points = part.mesh.triangulate().points
#             # kdtree = ckdtree.cKDTree(mesh_points)
#             # distances, _ = kdtree.query(points)

#             # avg_distance = np.mean(distances)
#             if avg_distance < best_avg_distance:
#                 best_avg_distance = avg_distance
#                 best_part = part
#                 best_pc = target_pc
#                 # visualize(
#                 #     [best_pc, best_part.mesh],
#                 #     colors=["red", "tan"],
#                 #     filename=f"designs/best_part_{sample['category']}_{sample['object_id']}.png",
#                 # )
#                 # time.sleep(0.5)

#     if best_part is not None:
#         # Visualize the best part
#         visualize(
#             [best_pc, best_part.mesh],
#             colors=["red", "tan"],
#             filename=f"designs/best_part_{sample['category']}_{sample['object_id']}.png",
#         )
#     else:
#         print("No part found")


def arbitrary_primitives_strategy(sample):
    meshes = sample['meshes']
    fitted_meshes = []
    for mesh in meshes:
        # Fit a primitive shape to the mesh
        centroid, rotation, bounds = fit_with_rotation(mesh.points)

        # bounding_box = mesh.bounds
        # dims = [bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2], bounding_box[5] - bounding_box[4]]
        # dims.sort(reverse=True)

        options = [
            Box(x_length=bounds[0], y_length=bounds[1], z_length=bounds[2]),
        ]

        option_scores = [score_mesh_fit(option, mesh.points, rotation) for option in options]
        best_option = options[np.argmin(option_scores)]
        fitted_part = AssembledComponent(part_id=-1, translation=centroid, rotation=rotation, custom_component=best_option)
        #### do comparison to see the best mesh
        fitted_meshes.append(fitted_part.mesh)
    return fitted_meshes


def arbitrary_length_strategy(sample):
    meshes = sample['meshes']

    # scales = np.linspace(0.1, 3.0, 30)
    scales = [1.0,]
    best_scale_score = np.inf
    best_scale_meshes = None
    best_scale = None
    for scale in scales:
        scale_score = 0
        scaled_meshes = [mesh.scale(scale) for mesh in meshes]
        scale_meshes = []
        for mesh in scaled_meshes:
            centroid, rotation, bounds = fit_with_rotation(mesh.points)
            best_part, best_avg_distance = None, np.inf
            for shape_id, shape in Design.PART_LIBRARY.items():
                base_shape = Design.PART_LIBRARY[shape_id]
                modified_shape = copy.deepcopy(base_shape)
                modified_shape.z_length = bounds[0]

                avg_distance = score_mesh_fit(modified_shape, mesh.points, rotation)
                # print(f"Shape ID: {shape_id}, Avg Distance: {avg_distance}")
                if avg_distance < best_avg_distance:
                    best_avg_distance = avg_distance
                    best_part = AssembledComponent(part_id=-1,
                                                    translation=centroid,
                                                    rotation=rotation,
                                                    custom_component=modified_shape)
            scale_score += best_avg_distance
            scale_meshes.append(best_part.mesh)
        if scale_score < best_scale_score:
            best_scale_score = scale_score
            best_scale_meshes = scale_meshes
            best_scale = scale

    print(f"Best scale: {best_scale}")

    return best_scale_meshes, best_scale

def our_primitives_strategy(sample):
    meshes = sample['meshes']

    # scales = np.linspace(0.1, 3.0, 30)
    scales = [1.0,]
    best_scale_score = np.inf
    best_scale_meshes = None
    best_scale = None
    for scale in scales:
        scale_score = 0
        scaled_meshes = [mesh.scale(scale) for mesh in meshes]
        scale_meshes = []
        for mesh in scaled_meshes:
            centroid, rotation, bounds = fit_with_rotation(mesh.points)
            best_part, best_avg_distance = None, np.inf
            for shape_id, shape in Design.PART_LIBRARY.items():
                avg_distance = score_mesh_fit(shape, mesh.points, rotation)
                # print(f"Shape ID: {shape_id}, Avg Distance: {avg_distance}")
                if avg_distance < best_avg_distance:
                    best_avg_distance = avg_distance
                    best_part = AssembledComponent(
                        part_id=shape_id,
                        translation=centroid,
                        rotation=rotation,
                    )
            scale_score += best_avg_distance
            scale_meshes.append(best_part.mesh)
        if scale_score < best_scale_score:
            best_scale_score = scale_score
            best_scale_meshes = scale_meshes
            best_scale = scale

    print(f"Best scale: {best_scale}")

    return best_scale_meshes, best_scale




def score_mesh_fit(centered_mesh, points, rotation):
    point_centroid = np.mean(points, axis=0)
    centered_points = points - point_centroid
    aligned_points = centered_points @ rotation.as_matrix()

    if type(centered_mesh) is Box:
        box_x, box_y, box_z = centered_mesh.x_length, centered_mesh.y_length, centered_mesh.z_length
        score = score_points_to_box(aligned_points, box_x, box_y, box_z)
    elif type(centered_mesh) is Cylinder:
        cylinder_direction = centered_mesh.direction
        cylinder_radius = centered_mesh.radius
        cylinder_height = centered_mesh.height
        score = score_points_to_cylinder(aligned_points, cylinder_direction, cylinder_radius, cylinder_height)
        # avg_distance = N_SAMPLED_POINTS - np.sum(avg_distance < 5)
    else:
        print(f"Unknown shape type: {type(centered_mesh)}")

    return score



# Approach: iteratively rotate around each axis to minimize bounding box volume
def fit_with_rotation(points, coarse_steps=20, fine_steps=10):
    MAX_ANGLE = np.pi/4
    pts = np.asarray(points).astype(float)
    center = pts.mean(axis=0)
    pts_c = pts - center
    
    best_vol = np.inf
    pts_r = pts_c.copy()
    R_total = np.eye(3)
    
    for axis in range(3):
        # Stage 1: Coarse search across full range
        coarse_thetas = np.linspace(-MAX_ANGLE, MAX_ANGLE, coarse_steps)
        best_coarse_theta = 0
        best_coarse_vol = np.inf
        
        for theta in coarse_thetas:
            R_search = get_rotation_matrix(axis, theta)
            pts_candidate = (R_search @ pts_r.T).T
            
            extents = pts_candidate.max(axis=0) - pts_candidate.min(axis=0)
            vol = extents[0] * extents[1] * extents[2]
            
            if vol < best_coarse_vol:
                best_coarse_vol = vol
                best_coarse_theta = theta
        
        # Stage 2: Fine search around best coarse result
        theta_range = (np.pi/2) / coarse_steps  # Range around best coarse angle
        fine_start = max(-MAX_ANGLE, best_coarse_theta - theta_range)
        fine_end = min(MAX_ANGLE, best_coarse_theta + theta_range)
        fine_thetas = np.linspace(fine_start, fine_end, fine_steps)
        
        best_rot = np.eye(3)
        best_fine_vol = best_coarse_vol
        
        for theta in fine_thetas:
            R_search = get_rotation_matrix(axis, theta)
            pts_candidate = (R_search @ pts_r.T).T
            
            extents = pts_candidate.max(axis=0) - pts_candidate.min(axis=0)
            vol = extents[0] * extents[1] * extents[2]
            
            if vol < best_fine_vol:
                best_fine_vol = vol
                best_rot = R_search
        
        # Update for next axis iteration
        if best_fine_vol < best_vol:
            best_vol = best_fine_vol
        
        pts_r = (best_rot @ pts_r.T).T
        R_total = best_rot @ R_total
    
    # Final alignment and bounds calculation
    aligned = (R_total @ pts_c.T).T
    x_length, y_length, z_length = aligned.max(axis=0) - aligned.min(axis=0)
    rotation = R.from_matrix(R_total)
    
    return center, rotation, (x_length, y_length, z_length)


def get_rotation_matrix(axis, theta):
    ct, st = np.cos(theta), np.sin(theta)
    
    if axis == 0:  # x-axis rotation
        return np.array([[1, 0,   0],
                        [0, ct, -st],
                        [0, st,  ct]])
    elif axis == 1:  # y-axis rotation
        return np.array([[ ct, 0, st],
                        [ 0,  1, 0],
                        [-st, 0, ct]])
    else:  # z-axis rotation
        return np.array([[ct, -st, 0],
                        [st,  ct, 0], 
                        [0,   0,  1]])


if __name__ == "__main__":

    # points = np.array([[0,0,0],[0,0,-3],[2,0,0],[0,2,0], [0,-2,3], [-1, -1, 0], [-3,0,1], [1,1,10]])
    # correct_distances = np.array([2, 0, 0, 0, 0, (2-np.sqrt(2)), 1, 7])
    # print(np.allclose(distance_points_to_cylinder(points, np.array([0,0,1]), 2, 6), correct_distances))
    # print(distance_points_to_cylinder(points, np.array([0,0,1]), 2, 6))
    # print(np.abs(distance_points_to_cylinder(points, np.array([0,0,1]), 2, 6) - correct_distances))
    
    # exit()

    # Design.visualize_part_library()

    # create_random_designs()
    # visualize_txt_designs(GPT_DESIGNS)

    # samples = get_shapenet_samples()
    # for sample in samples:
    #     visualize(
    #         [sample["transformed_mesh"], sample["point_cloud"]],
    #         colors=["tan", "red"],
    #         filename=f"designs/shapenet_{sample['category']}_{sample['object_id']}.png",
    #     )
    #     ransac_for_pointcloud(sample)


    desired_models = [
        "1299",
        "10027",
        "18258",
        "18740",
    ]


    # Need to investigate how to integrate the part hierarchy as well
    for num in desired_models:
        original_meshes = get_and_transform_partnet_meshes(f"/Users/ryanslocum/Downloads/cutlist/scratch/{num}")

        arbitrary_meshes = arbitrary_primitives_strategy({"meshes": original_meshes})
        length_meshes, best_scale = arbitrary_length_strategy({"meshes": original_meshes})
        our_meshes, best_scale = our_primitives_strategy({"meshes": original_meshes})

        original_point_cloud = pv.PolyData(np.vstack([mesh.points[::10] for mesh in original_meshes]))

        off_screen = False
        visualize([original_point_cloud] + arbitrary_meshes, colors=['red'] + ['tan']*len(arbitrary_meshes), filename=f"designs/arbitrary_fitted_meshes_{num}.png", axis_length=25, off_screen=off_screen)
        # visualize([original_point_cloud] + length_meshes, colors=['red'] + ['tan']*len(length_meshes), filename=f"designs/length_fitted_meshes_{num}.png", axis_length=25, off_screen=off_screen)
        # visualize([original_point_cloud] + our_meshes, colors=['red'] + ['tan']*len(our_meshes), filename=f"designs/our_fitted_meshes_{num}.png", axis_length=25, off_screen=off_screen)
