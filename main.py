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
    # PART_LIBRARY = {
    #     0: Part(pv.Cube, x_length=20, y_length=20, z_length=100),
    #     1: Part(pv.Cube, x_length=20, y_length=20, z_length=50),
    #     2: Part(pv.Cube, x_length=5, y_length=100, z_length=100),
    # }
    PART_LIBRARY = {
        # Cubes / rectangular blocks
        0:  Part(pv.Cube, x_length=20, y_length=20, z_length=100),   # Tall post
        1:  Part(pv.Cube, x_length=20, y_length=20, z_length=50),    # Short post
        2:  Part(pv.Cube, x_length=5,  y_length=100, z_length=100),  # Thin panel
        3:  Part(pv.Cube, x_length=40, y_length=40, z_length=40),    # Large cube
        4:  Part(pv.Cube, x_length=80, y_length=20, z_length=20),    # Beam
        5:  Part(pv.Cube, x_length=100, y_length=5,  z_length=20),   # Shelf plank
        6:  Part(pv.Cube, x_length=60, y_length=60, z_length=10),    # Flat plate
        7:  Part(pv.Cube, x_length=30, y_length=120, z_length=20),   # Slat
        8:  Part(pv.Cube, x_length=25, y_length=25, z_length=200),   # Long leg
        9:  Part(pv.Cube, x_length=15, y_length=15, z_length=60),    # Stick

        # Cylinders (good for dowels, rods, posts)
        10: Part(pv.Cylinder, radius=10, height=80, direction=(0,0,1)),
        11: Part(pv.Cylinder, radius=50, height=10, direction=(0,0,1)),  # Thick disk
    }


    assembledComponents: list

    def __init__(self, assembledComponents, bounds):
        self.assembledComponents = assembledComponents
        self.bounds = bounds

    def to_vector(self):
        list_of_vectors = [
            component.to_vector() for component in self.assembledComponents
        ]
        return np.concatenate(list_of_vectors)

    def to_txt(self, filename):
        with open(filename, "w") as f:
            for component in self.assembledComponents:
                euler_angles = component.rotation.as_euler("xyz")
                euler_strs = [str(angle) for angle in euler_angles]
                f.write(
                    f"{component.part_id} {component.translation[0]} {component.translation[1]} {component.translation[2]} {' '.join(euler_strs)}\n"
                )

    def from_txt(self, filename):
        assembledComponents = []
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                part_id = int(parts[0])
                translation = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
                rotation = R.from_euler("xyz", [float(x) for x in parts[4:]])
                assembledComponents.append(
                    AssembledComponent(part_id, translation, rotation)
                )
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
            plotter.add_mesh(component.mesh, color="tan")

        # Add bounding box
        bounds = self.bounds.flatten()
        bounding_box = pv.Box(bounds)
        plotter.add_mesh(bounding_box, color="red", style="wireframe", line_width=2)

        plotter.set_background("white")
        plotter.camera_position = "iso"  # Choose a nice view angle
        plotter.screenshot(filename=filename)
        # image = plotter.screenshot(return_img=True)

    @staticmethod
    def visualize_part_library(filename="part_library"):
        """Visualize all parts in the part library in a grid layout"""
        plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))

        # Calculate grid dimensions for layout
        num_parts = len(DesignParameters.PART_LIBRARY)
        cols = int(np.ceil(np.sqrt(num_parts)))
        rows = int(np.ceil(num_parts / cols))

        # Spacing between parts
        spacing = 200

        for i, (part_id, part) in enumerate(DesignParameters.PART_LIBRARY.items()):
            # Calculate grid position
            row = i // cols
            col = i % cols

            # Create the part mesh
            part_mesh = part.create()

            # Position in grid
            x_offset = col * spacing
            y_offset = row * spacing
            z_offset = 0

            # Translate the mesh to its grid position
            translated_mesh = part_mesh.translate([x_offset, y_offset, z_offset])

            # Add to plotter with different colors for variety
            colors = [
                "lightblue",
                "lightgreen",
                "lightcoral",
                "lightyellow",
                "lightpink",
                "lightgray",
                "lightcyan",
                "wheat",
                "lavender",
                "mistyrose",
            ]
            color = colors[part_id % len(colors)]
            plotter.add_mesh(
                translated_mesh, color=color, show_edges=True, edge_color="black"
            )

            # Add text label for part ID - position it below each part
            bounds = translated_mesh.bounds

        plotter.set_background("white")
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.5)  # Zoom out to see all parts
        plotter.screenshot(filename=filename + ".png")
        print(f"Part library visualization saved as {filename}.png")

    def verify(self):

        # FIRST CHECK: MAKE SURE ALL PARTS ARE WITHIN BOUNDS
        # Make a bounding box from the bounds
        bounds_min = self.bounds[:, 0]
        bounds_max = self.bounds[:, 1]
        for component in self.assembledComponents:
            c_bound = component.mesh.bounds
            component_min_bounds = [c_bound.x_min, c_bound.y_min, c_bound.z_min]
            component_max_bounds = [c_bound.x_max, c_bound.y_max, c_bound.z_max]
            if np.any(component_min_bounds < bounds_min) or np.any(
                component_max_bounds > bounds_max
            ):
                print(
                    f"Component {component.part_id} is out of bounds. Verification Failed."
                )
                return False
        return True


class AssembledComponent:
    def __init__(self, part_id: int, translation: np.ndarray, rotation: R):
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
        rotation_increments = np.pi / 6
        random_angles = (
            np.random.randint(0, 12, size=3) * rotation_increments
        )  # 12 increments in 2*pi
        rotation = R.from_euler("xyz", random_angles)
        part_id = np.random.randint(0, max_id)
        return AssembledComponent(part_id, translation, rotation)

    @staticmethod
    def random_component_on_surface(
        existing_component, new_part_id, offset_distance=0.1
    ):
        """Place a new component on the surface of an existing one by selecting a random face"""
        mesh = existing_component.mesh

        # Get all faces (cells) of the mesh
        num_faces = mesh.n_cells

        # Choose a random face
        face_idx = np.random.randint(num_faces)

        # Get the face as a separate mesh
        face_mesh = mesh.extract_cells([face_idx])
        face_normal = mesh.cell_normals[face_idx]  # Get the normal of the face

        # Get the face center point and normal
        face_center = face_mesh.center

        # Alternative: Sample a random point on the face instead of using center
        # You can also generate a random barycentric coordinate for triangular faces
        face_points = face_mesh.points
        if len(face_points) == 3:  # Triangle face
            # Random barycentric coordinates
            r1, r2 = np.random.rand(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            r3 = 1 - r1 - r2
            placement_point = (
                r1 * face_points[0] + r2 * face_points[1] + r3 * face_points[2]
            )
        elif len(face_points) == 4:  # Quad face (like cube faces)
            # Random point on quad using bilinear interpolation
            u, v = np.random.rand(2)
            placement_point = (
                (1 - u) * (1 - v) * face_points[0]
                + u * (1 - v) * face_points[1]
                + u * v * face_points[2]
                + (1 - u) * v * face_points[3]
            )
        else:
            # Fallback to face center for other face types
            placement_point = face_center

        # Create the new part to get its dimensions
        new_part = DesignParameters.PART_LIBRARY[new_part_id].create()
        bounds = new_part.bounds

        # Calculate offset based on the new part's size
        part_size = max(
            bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
        )
        offset = face_normal * (part_size / 2 + offset_distance)

        translation = placement_point + offset

        # Align the new part with the surface normal
        try:
            rotation = R.align_vectors([face_normal], [[0, 0, 1]])[0]
        except:
            # If alignment fails, use identity rotation
            rotation = R.from_euler("xyz", [0, 0, 0])

        return AssembledComponent(new_part_id, translation, rotation)


def main():
    NUM_COMPONENTS = 4
    NUM_DESIGNS = 100
    STRATEGY = "surface"  #  "surface", "random"
    bounds = np.array([[-200, 200], [-200, 200], [0, 200]])
    for i in range(NUM_DESIGNS):
        if STRATEGY == "surface":
            # Create a design with components placed on the surface of existing components
            components = []
            for j in range(NUM_COMPONENTS):
                if j == 0:
                    # First component is random
                    component = AssembledComponent.random_component(
                        max_id=len(DesignParameters.PART_LIBRARY), bounds=bounds
                    )
                else:
                    # Subsequent components are placed on the surface of the previous one
                    component = AssembledComponent.random_component_on_surface(
                        components[j - 1], new_part_id=np.random.randint(0, 3)
                    )
                components.append(component)
        else:
            # Create a design with random components
            components = [
                AssembledComponent.random_component(max_id=3, bounds=bounds)
                for _ in range(NUM_COMPONENTS)
            ]
        design = DesignParameters(assembledComponents=components, bounds=bounds)
        if design.verify():
            design.visualize_design(filename=f"designs/valid_design_{i}")
            # design.to_txt(f"designs/valid_design_{i}.txt")
        else:
            # print(f"Design {i} is invalid and will not be visualized.")
            design.visualize_design(filename=f"designs/invalid_design_{i}")
            # design.to_txt(f"designs/invalid_design_{i}.txt")

    # # Create a pair that does touch but does not overlap
    # part1 = AssembledComponent(0, np.array([0, 0, 0]), R.from_euler('xyz', np.array([0, 0, 0])))
    # part2 = AssembledComponent(0, np.array([20, 0, 0]), R.from_euler('xyz', np.array([0, 0, 0])))
    # design = DesignParameters(assembledComponents=[part1, part2])
    # if design.verify():
    #     visualize_design(design, name="touching_but_not_overlapping")
    # else:
    #     print("Design with touching but not overlapping parts is invalid and will not be visualized.")


if __name__ == "__main__":

    # DesignParameters.visualize_part_library()

    main()
