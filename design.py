import numpy as np
from scipy.spatial.transform import Rotation as R
import pyvista as pv
import os


POSSIBLE_COLORS = [
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


def visualize(
    meshes,
    colors=None,
    opacities=None,
    bounds=None,
    axis_length=0,
    camera_position="iso",
    filename="visualization",
    off_screen=True,
    text=None,
):
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(off_screen=off_screen)
    if colors is None:
        # Random colors for each mesh
        colors = [POSSIBLE_COLORS[i % len(POSSIBLE_COLORS)] for i in range(len(meshes))]
    if opacities is None:
        opacities = [1.0 for _ in range(len(meshes))]
    for mesh, color, opacity in zip(meshes, colors, opacities):
        plotter.add_mesh(mesh, color=color, opacity=opacity)

    # Add bounding box
    if bounds is not None:
        bounds = bounds.flatten()
        bounding_box = pv.Box(bounds)
        plotter.add_mesh(bounding_box, color="red", style="wireframe", line_width=2)
    if axis_length > 0:
        x_axis = pv.Arrow(start=(0, 0, 0), direction=(10, 0, 0), scale=axis_length)
        y_axis = pv.Arrow(start=(0, 0, 0), direction=(0, 10, 0), scale=axis_length)
        z_axis = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 10), scale=axis_length)
        plotter.add_mesh(x_axis, color="red")
        plotter.add_mesh(y_axis, color="green")
        plotter.add_mesh(z_axis, color="blue")

    if text is not None:
        plotter.add_text(text, position="upper_edge", font_size=12, color="black")

    plotter.set_background("white")
    plotter.camera_position = camera_position
    # Create parent directory if it doesn't exist
    if off_screen:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plotter.screenshot(filename=filename)

        # image = plotter.screenshot(return_img=True)
    else:
        plotter.show()


class Part:
    def __init__(self, function, **kwargs):
        self.function = function
        self.kwargs = kwargs

    def create(self):
        return self.function(**self.kwargs)


class Box(Part):
    def __init__(self, x_length, y_length, z_length):
        self.x_length = x_length
        self.y_length = y_length
        self.z_length = z_length
        super().__init__(
            pv.Cube, x_length=x_length, y_length=y_length, z_length=z_length
        )


class Cylinder(Part):
    def __init__(self, radius, height, direction):
        self.radius = radius
        self.height = height
        self.direction = np.array(direction)
        super().__init__(pv.Cylinder, radius=radius, height=height, direction=direction)


class Design:
    # PART_LIBRARY = {
    #     # Cubes / rectangular blocks
    #     0: Part(pv.Cube, x_length=20, y_length=20, z_length=120),  # Medium post
    #     1: Part(pv.Cube, x_length=20, y_length=20, z_length=160),  # Tall post
    #     2: Part(pv.Cube, x_length=20, y_length=40, z_length=120),  # Medium plank
    #     3: Part(pv.Cube, x_length=20, y_length=40, z_length=160),  # Tall plank
    #     4: Part(pv.Cube, x_length=80, y_length=40, z_length=5),  # Small square plate
    #     5: Part(pv.Cube, x_length=160, y_length=40, z_length=5),  # Small rectangle plate
    #     6: Part(pv.Cube, x_length=160, y_length=80, z_length=5),  # Large rectangle plate
    #     # Cylinders (good for dowels, rods, posts)
    #     7: Part(pv.Cylinder, radius=10, height=80, direction=(0, 0, 1)),  # Dowel
    #     8: Part(pv.Cylinder, radius=80, height=10, direction=(0, 0, 1)),  # Thick disk
    # }
    PART_LIBRARY = {
        # Cubes / rectangular blocks
        0: Box(x_length=120, y_length=20, z_length=20),  # Medium post
        1: Box(x_length=160, y_length=20, z_length=20),  # Tall post
        2: Box(x_length=120, y_length=40, z_length=20),  # Medium plank
        3: Box(x_length=160, y_length=40, z_length=20),  # Tall plank
        4: Box(x_length=80, y_length=40, z_length=5),  # Small square plate
        5: Box(x_length=160, y_length=40, z_length=5),  # Small rectangle plate
        6: Box(x_length=160, y_length=80, z_length=5),  # Large rectangle plate
        # Cylinders (good for dowels, rods, posts)
        # 7: Cylinder(radius=10, height=80, direction=(1, 0, 0)),  # Dowel
        # 8: Cylinder(radius=80, height=10, direction=(0, 0, 1)),  # Thick disk
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

    def from_txt(self, txt, from_file=True):
        if from_file:
            filename = txt
            txt = ""
            with open(filename, "r") as f:
                txt = f.read()

        txt_lines = txt.strip().split("\n")
        assembledComponents = []
        for line in txt_lines:
            parts = line.strip().split()
            part_id = int(parts[0])
            translation = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
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
        meshes = [component.mesh for component in self.assembledComponents]
        visualize(meshes, bounds=self.bounds, filename=filename + ".png")

    @staticmethod
    def visualize_part_library(filename="designs/part_library", spacing=200):
        # Calculate grid dimensions for layout
        num_parts = len(Design.PART_LIBRARY)
        cols = int(np.ceil(np.sqrt(num_parts)))
        meshes, colors = [], []

        for i, (part_id, part) in enumerate(Design.PART_LIBRARY.items()):
            # Calculate grid position
            row = i // cols
            col = i % cols
            part_mesh = part.create()

            # Position in grid
            x_offset = (col + 1) * spacing
            y_offset = (row + 1) * spacing

            # Translate the mesh to its grid position
            translated_mesh = part_mesh.translate([x_offset, y_offset, 0])

            colors.append(POSSIBLE_COLORS[part_id % len(POSSIBLE_COLORS)])
            meshes.append(translated_mesh)

        visualize(meshes, colors=colors, axis_length=100, filename=filename)

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
                # print(f"Component {component.part_id} is out of bounds. Verification Failed.")
                return False
        return True


class AssembledComponent:
    def __init__(
        self,
        part_id: int,
        translation: np.ndarray,
        rotation: R,
        custom_component: Part = None,
    ):
        if custom_component is not None:
            self.part_id = -1  # Indicate custom part
            part = custom_component
        else:
            self.part_id = part_id
            part = Design.PART_LIBRARY[self.part_id]
        self.translation = translation
        self.rotation = rotation

        part_mesh = part.create()
        part_mesh_copy = part_mesh.copy().triangulate()

        # Convert rotation to 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = self.rotation.as_matrix()
        transform_matrix[:3, 3] = self.translation

        self.mesh = part_mesh_copy.transform(transform_matrix, inplace=True)

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
    def random_component_on_floor(max_id, bounds: np.ndarray):
        # Create a new component with a random x, y position on the floor
        translation = np.array(
            [
                np.random.uniform(bounds[0, 0], bounds[0, 1]),  # x
                np.random.uniform(bounds[1, 0], bounds[1, 1]),  # y
                bounds[2, 0],  # z (on the floor)
            ]
        )
        rotation = R.from_euler("xyz", [0, 0, 0])  # No rotation

        part_id = np.random.randint(0, max_id)
        return AssembledComponent(part_id, translation, rotation)

    @staticmethod
    def random_component_on_surface(existing_component, max_id, offset_distance=0.1):
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
        part_id = np.random.randint(0, max_id)
        new_part = Design.PART_LIBRARY[part_id].create()
        bounds = new_part.bounds

        # Calculate offset based on the new part's size
        part_size = max(
            bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
        )
        offset = face_normal * (part_size / 2 + offset_distance)

        translation = placement_point + offset

        # Align the new part with the surface normal
        rotation = R.align_vectors([face_normal], [[0, 0, 1]])[0]

        return AssembledComponent(part_id, translation, rotation)


def create_random_designs():
    NUM_COMPONENTS = 4
    NUM_DESIGNS = 100
    STRATEGY = "surface"  #  "surface", "random"
    MAX_ID = len(Design.PART_LIBRARY)
    bounds = np.array([[-200, 200], [-200, 200], [0, 200]])
    for i in range(NUM_DESIGNS):
        if STRATEGY == "surface":
            # Create a design with components placed on the surface of existing components
            components = []
            for j in range(NUM_COMPONENTS):
                if j == 0:
                    # First component is randomly placed on the floor
                    component = AssembledComponent.random_component_on_floor(
                        MAX_ID, bounds=bounds
                    )
                else:
                    # Subsequent components are placed on the surface of the previous one
                    component = AssembledComponent.random_component_on_surface(
                        components[j - 1], MAX_ID
                    )
                components.append(component)
        else:
            # Create a design with random components
            components = [
                AssembledComponent.random_component(max_id=MAX_ID, bounds=bounds)
                for _ in range(NUM_COMPONENTS)
            ]
        design = Design(assembledComponents=components, bounds=bounds)
        if design.verify():
            print(f"Design {i} is valid and will be visualized.")
            design.visualize_design(filename=f"designs/valid_design_{i}")
            # design.to_txt(f"designs/valid_design_{i}.txt")
        else:
            print(f"Design {i} is invalid and will not be visualized.")
            design.visualize_design(filename=f"designs/invalid_design_{i}")
            # design.to_txt(f"designs/invalid_design_{i}.txt")


class WoodPart:
    def __init__(self, transform: np.ndarray):
        self.transform = transform

    # def create(self):
    #     return pv.Cube(x_length=self.x_len, y_length=self.y_len, z_length=self.z_len)


class ArbitraryCuboid(WoodPart):
    def __init__(self, dims: np.ndarray, transform: np.ndarray):
        super().__init__(transform)
        self.dims = dims

    def get_mesh(self):
        return pv.Cube(
            x_length=self.dims[0], y_length=self.dims[1], z_length=self.dims[2]
        ).transform(self.transform, inplace=True)

    def to_text(self):
        centroid = self.transform[:3, 3]
        euler_angles = R.from_matrix(self.transform[:3, :3]).as_euler("xyz")
        properties = [
            max(round(self.dims[0]), 1),
            max(round(self.dims[1]), 1),
            max(round(self.dims[2]), 1),
            round(centroid[0]),
            round(centroid[1]),
            round(centroid[2]),
            round(euler_angles[0]),
            round(euler_angles[1]),
            round(euler_angles[2]),
        ]

        return " ".join([str(prop) for prop in properties])


class WoodDesign:
    def __init__(self, wood_parts: list[WoodPart]):
        self.wood_parts = wood_parts

    def to_txt(self):
        sorted_parts = self.assembly_order()
        return "\n".join([part.to_text() for part in sorted_parts])

    def assembly_order(self):
        # Sort wood parts by their z-coordinate (height)
        return sorted(self.wood_parts, key=lambda part: part.centroid[2])
