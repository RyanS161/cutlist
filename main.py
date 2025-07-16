import numpy as np
from scipy.spatial.transform import Rotation as R
import pyvista as pv
from itertools import combinations

    
class Part:
    def __init__(self, function, **kwargs):
        self.function = function
        self.kwargs = kwargs
    
    def create(self):
        return self.function(**self.kwargs)


class DesignParameters:
    MAX_PARTS = 32
    PART_LIBRARY = {
        0: Part(pv.Cube, x_length=19, y_length=19, z_length=100),
        1: Part(pv.Cube, x_length=19, y_length=19, z_length=100),
        2: Part(pv.Cube, x_length=100, y_length=100, z_length=3.5),
    }
    assembledComponents: list
    def __init__(self, assembledComponents, bounds):
        self.assembledComponents=assembledComponents
        self.bounds = bounds

    def to_vector(self):
        list_of_vectors = [component.to_vector() for component in self.assembledComponents]
        return np.concatenate(list_of_vectors)
    
    def to_txt(self, filename):
        with open(filename, 'w') as f:
            for component in self.assembledComponents:
                euler_angles = component.rotation.as_euler('xyz')
                euler_strs = [str(angle) for angle in euler_angles]
                f.write(f"{component.part_id} {component.translation[0]} {component.translation[1]} {component.translation[2]} {' '.join(euler_strs)}\n")

    def from_txt(self, filename):
        assembledComponents = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                part_id = int(parts[0])
                translation = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                rotation = R.from_euler('xyz', [float(x) for x in parts[4:]])
                assembledComponents.append(AssembledComponent(part_id, translation, rotation))
        self.assembledComponents = assembledComponents

    # def from_vector(self, vector):
    #     assembledComponents = []
    #     offset = 0
    #     for part in self.assembledComponents:
    #         part_size = len(part.to_vector())
    #         part_vector = vector[offset:offset + part_size]
    #         new_part = AssembledComponent(part.name, RigidTransform())
    #         new_part.from_vector(part_vector)
    #         assembledComponents.append(new_part)
    #         offset += part_size
    #     self.assembledComponents = assembledComponents

    def visualize_design(self, filename="design"):
        plotter = pv.Plotter(off_screen=True)
        for component in self.assembledComponents:
            plotter.add_mesh(component.mesh, color='tan')
        
        # Add bounding box
        bounds = self.bounds.flatten()
        bounding_box = pv.Box(bounds)
        plotter.add_mesh(bounding_box, color='red', style='wireframe', line_width=2)

        plotter.set_background('white')
        plotter.camera_position = 'iso'  # Choose a nice view angle
        plotter.screenshot(filename=filename)
        # image = plotter.screenshot(return_img=True)

    def verify(self):

        # FIRST CHECK: MAKE SURE ALL PARTS ARE WITHIN BOUNDS
        # Make a bounding box from the bounds
        bounds_min = self.bounds[:, 0]
        bounds_max = self.bounds[:, 1]
        for component in self.assembledComponents:
            c_bound = component.mesh.bounds
            component_min_bounds = [c_bound.x_min, c_bound.y_min, c_bound.z_min]
            component_max_bounds = [c_bound.x_max, c_bound.y_max, c_bound.z_max]
            if np.any(component_min_bounds < bounds_min) or np.any(component_max_bounds > bounds_max):
                print(f"Component {component.part_id} is out of bounds. Verification Failed.")
                return False
        return True

class AssembledComponent:
    def __init__(self, part_id:int, translation:np.ndarray, rotation:R):
        self.part_id = part_id
        self.translation = translation
        self.rotation = rotation

        part = DesignParameters.PART_LIBRARY[self.part_id]
        part_mesh = part.create()
        part_mesh_copy = part_mesh.copy().triangulate()
        
        # Convert rotation to 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = self.rotation.as_matrix()
        transform_matrix[:3, 3] = self.translation
        
        self.mesh = part_mesh_copy.transform(transform_matrix)

    def to_vector(self):
        return np.array(self.translation + self.rotation.as_euler())

    def from_vector(self, vector):
        self.translation = vector[:3]
        self.rotation = R.from_euler(vector[3:])

    @staticmethod
    def random_component(max_id, bounds: np.ndarray):
        if bounds.shape != (3, 2):
            raise ValueError("bounds must be a 3x2 ndarray")
        translation = np.array([np.random.uniform(low, high) for low, high in bounds])
        # Generate random Euler angles, each snapped to pi/6 increments
        rotation_increments = np.pi/6
        random_angles = np.random.randint(0, 12, size=3) * rotation_increments  # 12 increments in 2*pi
        rotation = R.from_euler('xyz', random_angles)
        part_id = np.random.randint(0, max_id)
        return AssembledComponent(part_id, translation, rotation)









def main():
    NUM_COMPONENTS = 5
    NUM_DESIGNS = 100
    bounds = np.array([[-200, 200], [-200, 200], [0, 200]])
    for i in range(NUM_DESIGNS):
        components = [AssembledComponent.random_component(max_id=3, bounds=bounds) for _ in range(NUM_COMPONENTS)]
        design = DesignParameters(assembledComponents=components, bounds=bounds)
        if design.verify():
            design.visualize_design(filename=f"designs/valid_design_{i}")
            design.to_txt(f"designs/valid_design_{i}.txt")
        else:
            # print(f"Design {i} is invalid and will not be visualized.")
            design.visualize_design(filename=f"designs/invalid_design_{i}")

    # # Create a pair that does touch but does not overlap
    # part1 = AssembledComponent(0, np.array([0, 0, 0]), R.from_euler('xyz', np.array([0, 0, 0])))
    # part2 = AssembledComponent(0, np.array([20, 0, 0]), R.from_euler('xyz', np.array([0, 0, 0])))
    # design = DesignParameters(assembledComponents=[part1, part2])
    # if design.verify():
    #     visualize_design(design, name="touching_but_not_overlapping")
    # else:
    #     print("Design with touching but not overlapping parts is invalid and will not be visualized.")

if __name__ == "__main__":
    main()