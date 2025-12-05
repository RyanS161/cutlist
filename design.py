import numpy as np
from scipy.spatial.transform import Rotation as R
import pyvista as pv
import os
from typing import Union
from PIL import Image

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
    filename=None,
    show_image=False,
    text=None,
):
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(off_screen=filename is not None or not show_image)
    bg_color = [0.1, 0.1, 0.15]
    if colors is None:
        # Random colors for each mesh
        colors = [POSSIBLE_COLORS[i % len(POSSIBLE_COLORS)] for i in range(len(meshes))]
    if opacities is None:
        opacities = [1.0 for _ in range(len(meshes))]
    for mesh, color, opacity in zip(meshes, colors, opacities):
        plotter.add_mesh(mesh, color=color, opacity=opacity)

    # Add bounding box
    if bounds is not None:
        bounds = np.array(bounds).flatten()
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

    plotter.set_background(bg_color)
    plotter.camera_position = camera_position
    # Create parent directory if it doesn't exist
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plotter.screenshot(filename=filename)

    if show_image:
        plotter.show()

    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def visualize_four_panel_img(
    meshes,
    filename=None,
):
    pv.global_theme.allow_empty_mesh = True

    view_size = 400
    colors = ["tan" for i in range(len(meshes))]
    opacities = [1.0 for _ in range(len(meshes))]

    # Define 4 isometric camera positions (azimuth, elevation)
    views = [
        (0, 0),
        (180, 0),
        (135, -90),
        (215, -90),
    ]

    # Dark background color matching website
    bg_color = [0.1, 0.1, 0.15]

    images = []

    for azimuth, elevation in views:
        plotter = pv.Plotter(off_screen=True, window_size=[view_size, view_size])
        for mesh, color, opacity in zip(meshes, colors, opacities):
            plotter.add_mesh(mesh, color=color, opacity=opacity)

        plotter.set_background(bg_color)
        plotter.camera_position = "iso"
        plotter.camera.azimuth = azimuth
        plotter.camera.elevation = elevation
        plotter.reset_camera()
        plotter.camera.zoom(1.0)

        img_array = plotter.screenshot(return_img=True)
        images.append(Image.fromarray(img_array))
        plotter.close()

    # Create combined 2x2 grid image
    grid_size = view_size * 2
    combined = Image.new("RGB", (grid_size, grid_size), color=(26, 26, 38))

    # Positions for 2x2 grid: top-left, top-right, bottom-left, bottom-right
    positions = [(0, 0), (view_size, 0), (0, view_size), (view_size, view_size)]

    for i, img in enumerate(images):
        combined.paste(img, positions[i])

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        combined.save(filename)

    return combined


def normalize_cuboid_rotation_to_positive(rot_matrix):
    """
    Find an equivalent rotation for a cuboid that has all positive Euler angles.

    Strategy:
    1. Generate all 8 symmetry transformations of a cuboid
    2. Apply each symmetry to the rotation matrix
    3. Convert each result to Euler angles
    4. Find the one where all angles are in [0, 180)

    Args:
        rot_matrix: 3x3 rotation matrix

    Returns:
        euler_angles: (rx, ry, rz) all in range [0, 180)
        rot_matrix: equivalent rotation matrix with positive Euler angles
    """

    # Add early return for if all angles are already positive
    euler = R.from_matrix(rot_matrix).as_euler("xyz", degrees=True)
    if np.all(euler >= 0) and np.all(euler < 180):
        return euler, rot_matrix

    symmetries = [
        np.eye(3),
        R.from_euler("x", 180, degrees=True).as_matrix(),
        R.from_euler("y", 180, degrees=True).as_matrix(),
        R.from_euler("z", 180, degrees=True).as_matrix(),
        R.from_euler("xy", [180, 180], degrees=True).as_matrix(),
        R.from_euler("xz", [180, 180], degrees=True).as_matrix(),
        R.from_euler("yz", [180, 180], degrees=True).as_matrix(),
        R.from_euler("xyz", [180, 180, 180], degrees=True).as_matrix(),
    ]

    for sym in symmetries:
        # Apply symmetry transformation
        combined = rot_matrix @ sym
        # Convert to Euler angles
        euler = R.from_matrix(combined).as_euler("xyz", degrees=True)

        # Normalize to [0, 360) range
        euler = euler % 360

        # Check if all angles are in [0, 180)
        if np.all(euler < 180):
            return euler, combined

    # Fallback: if no perfect match, choose the one with most positive angles
    best_euler = None
    best_matrix = None
    best_score = -1

    for sym in symmetries:
        combined = rot_matrix @ sym
        euler = R.from_matrix(combined).as_euler("xyz", degrees=True)
        euler = euler % 360

        # Count how many angles are in [0, 180)
        score = np.sum(euler < 180)
        if score > best_score:
            best_score = score
            best_euler = euler % 180  # Force into [0, 180)
            best_matrix = combined

    return best_euler, best_matrix


# Function for testing the above function because it's finnicky and AI generated and I don't trust it
# def test_normalization():
#     """
#     Test that normalization preserves cuboid orientation.
#     Tests with random rotations and verifies the cuboid looks identical.
#     """
#     print("Testing normalization with random rotations...\n")

#     num_tests = 20
#     all_passed = True

#     for test_num in range(num_tests):
#         # Generate random Euler angles (including negative values)
#         random_angles = np.random.uniform(-180, 180, size=3)

#         # Create rotation matrix and apply to cuboid
#         original_rot = R.from_euler("xyz", random_angles, degrees=True).as_matrix()

#         # Normalize to positive angles
#         normalized_angles, normalized_rot = normalize_cuboid_rotation_to_positive(original_rot)

#         # Check if all angles are positive and in [0, 180)
#         angles_positive = np.all(normalized_angles >= 0) and np.all(normalized_angles < 180)

#         # Verify the cuboids are actually the same by comparing vertices
#         # Create cuboid with different dimensions to avoid cube symmetries
#         original_cuboid = pv.Cube(x_length=50, y_length=100, z_length=200)
#         original_points = original_cuboid.points @ original_rot.T
#         normalized_points = original_cuboid.points @ normalized_rot.T

#         # Check if points match (allowing small numerical error)
#         points_match = np.allclose(
#             np.sort(original_points.flatten()),
#             np.sort(normalized_points.flatten()),
#             atol=1e-10
#         )

#         test_passed = angles_positive and points_match

#         if not test_passed or test_num < 5:  # Show first 5 tests always
#             status = "✓ PASS" if test_passed else "✗ FAIL"
#             print(f"Test {test_num + 1}: {status}")
#             print(f"  Original angles:    [{random_angles[0]:7.2f}, {random_angles[1]:7.2f}, {random_angles[2]:7.2f}]")
#             print(f"  Normalized angles:  [{normalized_angles[0]:7.2f}, {normalized_angles[1]:7.2f}, {normalized_angles[2]:7.2f}]")
#             print(f"  All positive (< 180): {angles_positive}")
#             print(f"  Orientation preserved: {points_match}")
#             print()

#         if not test_passed:
#             all_passed = False

#     if all_passed:
#         print(f"✓ All {num_tests} tests passed!")
#     else:
#         print(f"✗ Some tests failed")

#     return all_passed


class WoodPart:
    def __init__(self, transform: np.ndarray):
        self.transform = transform

    # def create(self):
    #     return pv.Cube(x_length=self.x_len, y_length=self.y_len, z_length=self.z_len)

    def get_int_euler_angles(self):
        # I am really not confident in the correctness of the normalization here but it seems to work for 99 percent of cases
        # Would like to double check if we're having issues with bad data
        original_euler_angles = (
            R.from_matrix(self.transform[:3, :3])
            .as_euler("xyz", degrees=True)
            .round()
            .astype(int)
        )
        rounded_matrix = R.from_euler(
            "xyz", original_euler_angles, degrees=True
        ).as_matrix()
        # return R.from_matrix(self.transform[:3, :3]).as_euler("xyz", degrees=True).astype(int)
        euler_angles, _ = normalize_cuboid_rotation_to_positive(rounded_matrix)

        euler_angles = euler_angles.astype(int)

        return euler_angles


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
        euler_angles = self.get_int_euler_angles()
        properties = [
            round(self.dims[0]),
            round(self.dims[1]),
            round(self.dims[2]),
            round(centroid[0]),
            round(centroid[1]),
            round(centroid[2]),
            euler_angles[0],
            euler_angles[1],
            euler_angles[2],
        ]

        # ignore parts that are invalid (zero or negative dimensions)
        if properties[0] <= 0 or properties[1] <= 0 or properties[2] <= 0:
            return ""

        # TODO: Should I be changing for format from space separated to something else?
        # The brickgpt paper uses 1x2(2,3,1) or something like that.
        # I could do dims as LxWxH(centroid_x,centroid_y,centroid_z)(rot_x,rot_y,rot_z)
        # But I'm not sure what that would accomplish necessarily.

        return " ".join([str(prop) for prop in properties])

    @staticmethod
    def from_text(text: str):
        parts = text.strip().split()
        if len(parts) != 9:
            # Invalid text format for ArbitraryCuboid.
            return None

        if not np.all([part.isdecimal() for part in parts]):
            return None

        dims = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        centroid = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
        euler_angles = np.array([float(parts[6]), float(parts[7]), float(parts[8])])
        rotation = R.from_euler("xyz", euler_angles, degrees=True).as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = centroid

        return ArbitraryCuboid(dims=dims, transform=transform)


class LibraryPrimitive(WoodPart):
    # ORIGINAL PARTS:
    # PART_LIBRARY = {
    #     # Cubes / rectangular blocks
    #     0: (120, 20, 20),  # Medium post
    #     1: (160, 20, 20),  # Tall post
    #     2: (120, 40, 20),  # Medium plank
    #     3: (160, 40, 20),  # Tall plank
    #     4: (80, 40, 5),  # Small square plate
    #     5: (160, 40, 5),  # Small rectangle plate
    #     6: (160, 80, 5),  # Large rectangle plate
    # }
    # PARTS FROM MILLI:
    PART_LIBRARY = {
        # 48x24 with lengths 100 to 500
        0: (48, 24, 100),
        1: (48, 24, 150),
        2: (48, 24, 200),
        3: (48, 24, 250),
        4: (48, 24, 300),
        5: (48, 24, 350),
        6: (48, 24, 400),
        7: (48, 24, 450),
        8: (48, 24, 500),
        # 28x28 with lengths 100 to 500
        9: (28, 28, 100),
        10: (28, 28, 150),
        11: (28, 28, 200),
        12: (28, 28, 250),
        13: (28, 28, 300),
        14: (28, 28, 350),
        15: (28, 28, 400),
        16: (28, 28, 450),
        17: (28, 28, 500),
        # 7mm thick plywood with varying sizes
        18: (7, 100, 100),
        19: (7, 200, 100),
        20: (7, 300, 100),
        21: (7, 400, 100),
        22: (7, 500, 100),
        23: (7, 100, 200),
        24: (7, 300, 200),
        25: (7, 400, 200),
        26: (7, 500, 200),
    }

    def __init__(self, part_id: int, transform: np.ndarray):
        super().__init__(transform)
        self.part_id = part_id

    def get_mesh(self):
        dims = LibraryPrimitive.PART_LIBRARY[self.part_id]
        return pv.Cube(x_length=dims[0], y_length=dims[1], z_length=dims[2]).transform(
            self.transform, inplace=True
        )

    def to_text(self):
        centroid = self.transform[:3, 3]
        euler_angles = self.get_int_euler_angles()
        properties = [
            self.part_id,
            round(centroid[0]),
            round(centroid[1]),
            round(centroid[2]),
            euler_angles[0],
            euler_angles[1],
            euler_angles[2],
        ]

        return " ".join([str(prop) for prop in properties])

    @staticmethod
    def from_text(text: str):
        parts = text.strip().split()
        if len(parts) != 7:
            raise ValueError("Invalid text format for LibraryPrimitive.")
        part_id = int(parts[0])
        centroid = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        euler_angles = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
        rotation = R.from_euler("xyz", euler_angles, degrees=True).as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = centroid

        return LibraryPrimitive(part_id=part_id, transform=transform)

    @staticmethod
    def visualize_part_library(filename="designs/part_library", spacing=200):
        # Calculate grid dimensions for layout
        num_parts = len(LibraryPrimitive.PART_LIBRARY)
        cols = int(np.ceil(np.sqrt(num_parts)))
        meshes, colors = [], []

        for i, part_id in enumerate(LibraryPrimitive.PART_LIBRARY):
            # Calculate grid position
            row = i // cols
            col = i % cols
            part_mesh = LibraryPrimitive(
                part_id=part_id, transform=np.eye(4)
            ).get_mesh()

            # Position in grid
            x_offset = (col + 1) * spacing
            y_offset = (row + 1) * spacing

            # Translate the mesh to its grid position
            translated_mesh = part_mesh.translate([x_offset, y_offset, 0])

            colors.append(POSSIBLE_COLORS[part_id % len(POSSIBLE_COLORS)])
            meshes.append(translated_mesh)

        visualize(meshes, colors=colors, axis_length=100, filename=filename)


class FootprintPrimitive(WoodPart):
    # ORIGINAL FOOTPRINTS:
    # FOOTPRINTS = {
    #     0: (20, 20),
    #     1: (40, 20),
    #     2: (40, 5),
    #     3: (80, 5),
    # }
    # FOOTPRINTS FROM MILLI:
    FOOTPRINTS = {
        # 48x24 with arbitrary length
        0: (48, 24),
        # 28x28 with arbitrary length
        1: (28, 28),
        # 7mm thick plywood with varying widths
        2: (
            7,
            50,
        ),  # It would be better if we could just say anything with thickness 7 is fine, but I need to fix the code for that
        3: (7, 100),
        4: (7, 150),
        5: (7, 200),
        6: (7, 250),
        7: (7, 300),
        8: (7, 350),
        9: (7, 400),
        10: (7, 450),
        11: (7, 500),
    }

    def __init__(self, part_id: int, length: float, transform: np.ndarray):
        super().__init__(transform)
        self.part_id = part_id
        self.length = length

    def get_mesh(self):
        dims = FootprintPrimitive.FOOTPRINTS[self.part_id]
        return pv.Cube(
            x_length=dims[0], y_length=dims[1], z_length=self.length
        ).transform(self.transform, inplace=True)

    def to_text(self):
        centroid = self.transform[:3, 3]
        euler_angles = self.get_int_euler_angles()
        properties = [
            self.part_id,
            self.length,
            round(centroid[0]),
            round(centroid[1]),
            round(centroid[2]),
            euler_angles[0],
            euler_angles[1],
            euler_angles[2],
        ]

        return " ".join([str(prop) for prop in properties])

    @staticmethod
    def from_text(text: str):
        parts = text.strip().split()
        if len(parts) != 8:
            raise ValueError("Invalid text format for FootprintPrimitive.")
        part_id = int(parts[0])
        length = float(parts[1])
        centroid = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
        euler_angles = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
        rotation = R.from_euler("xyz", euler_angles, degrees=True).as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = centroid

        return FootprintPrimitive(part_id=part_id, length=length, transform=transform)


class WoodDesign:
    def __init__(
        self,
        parts: list[WoodPart],
        design_type: Union[FootprintPrimitive, LibraryPrimitive, ArbitraryCuboid],
    ):
        self.parts = parts
        self.design_type = design_type
        self.text = None

    def add_part(self, part: WoodPart):
        self.parts.append(part)
        self.text = None  # Invalidate cached text representation

    def to_txt(self):
        if self.text is not None:
            return self.text

        sorted_parts = self.assembly_order()
        part_texts = [part.to_text() for part in sorted_parts]
        self.text = "\n".join([text for text in part_texts if text != ""])
        return self.text

    @staticmethod
    def from_txt(
        txt: str,
        design_type: Union[FootprintPrimitive, LibraryPrimitive, ArbitraryCuboid],
    ):
        if design_type is None:
            raise ValueError(
                "design_type must be specified to parse the text representation."
            )
        txt_lines = txt.strip().split("\n")
        parts = []
        for line in txt_lines:
            part = design_type.from_text(line)
            if part is None:
                return None
            parts.append(part)

        design = WoodDesign(parts=parts, design_type=design_type)
        design.text = txt

        return design

    def assembly_order(self):
        # Sort wood parts by their z-coordinate (height)
        return sorted(self.parts, key=lambda part: part.transform[2, 3])

    def clean_design(self):
        pass

    def visualize_img(self, **kwargs):
        meshes = [part.get_mesh() for part in self.parts]
        return visualize(meshes, **kwargs)

    def visualize_four_panel_img(self, filename, **kwargs):
        meshes = [part.get_mesh() for part in self.parts]
        return visualize_four_panel_img(meshes, filename=filename, **kwargs)

    def visualize_gif(self, filename, fps=4):
        meshes = [part.get_mesh() for part in self.parts]
        images = []

        for i in range(len(meshes)):
            opacities = [1.0 if j <= i else 0.1 for j in range(len(meshes))]
            image = visualize(
                meshes,
                colors=["tan"] * len(meshes),
                opacities=opacities,
            )
            images.append(image)
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        frames = [Image.fromarray(img) for img in images]
        frame_one = frames[0]
        frame_one.save(
            filename,
            format="GIF",
            append_images=frames,
            save_all=True,
            duration=(1000 // fps),
            loop=0,
        )

    def get_stl(self, filename):
        if not self.parts:
            return

        merged = pv.PolyData()
        for part in self.parts:
            merged = merged.merge(part.get_mesh())

        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            merged.save(filename)
