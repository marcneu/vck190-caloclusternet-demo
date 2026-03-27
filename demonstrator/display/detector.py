import numpy as np
import pyvista as pv
import pandas as pd

tc_mapping_kart = pd.read_csv("data/crystals.csv", index_col=0)
corners = tc_mapping_kart[["corner0_theta","corner0_phi","corner1_theta","corner1_phi","corner2_theta","corner2_phi","corner3_theta","corner3_phi"]].values

def create_crystal(radius3d, ll_corner, lr_corner, ul_corner, ur_corner, crystal_id, crystal_length=30, crate_id=1):
    """
    Create the 3D representation of a crystal using its corner coordinates and radius.
    """
    distance_to_back = radius3d + crystal_length
    if 1 <= crate_id<= 36:
        # Define the points in 3D space based on the corners
        p1 = np.array([radius3d * np.sin(ll_corner[0]) * np.cos(ll_corner[1]), radius3d * np.sin(ll_corner[0]) * np.sin(ll_corner[1]), radius3d * np.cos(ll_corner[0])])
        p2 = np.array([radius3d * np.sin(lr_corner[0]) * np.cos(lr_corner[1]), radius3d * np.sin(lr_corner[0]) * np.sin(lr_corner[1]), radius3d * np.cos(lr_corner[0])])
        p0 = np.array([radius3d * np.sin(ul_corner[0]) * np.cos(ul_corner[1]), radius3d * np.sin(ul_corner[0]) * np.sin(ul_corner[1]), radius3d * np.cos(ul_corner[0])])
        p3 = np.array([radius3d * np.sin(ur_corner[0]) * np.cos(ur_corner[1]), radius3d * np.sin(ur_corner[0]) * np.sin(ur_corner[1]), radius3d * np.cos(ur_corner[0])])
        back_p5 = np.array([distance_to_back * np.sin(ll_corner[0]) * np.cos(ll_corner[1]), distance_to_back * np.sin(ll_corner[0]) * np.sin(ll_corner[1]), distance_to_back * np.cos(ll_corner[0])])
        back_p6 = np.array([distance_to_back * np.sin(lr_corner[0]) * np.cos(lr_corner[1]), distance_to_back * np.sin(lr_corner[0]) * np.sin(lr_corner[1]), distance_to_back * np.cos(lr_corner[0])])
        back_p4 = np.array([distance_to_back * np.sin(ul_corner[0]) * np.cos(ul_corner[1]), distance_to_back * np.sin(ul_corner[0]) * np.sin(ul_corner[1]), distance_to_back * np.cos(ul_corner[0])])
        back_p7 = np.array([distance_to_back * np.sin(ur_corner[0]) * np.cos(ur_corner[1]), distance_to_back * np.sin(ur_corner[0]) * np.sin(ur_corner[1]), distance_to_back * np.cos(ur_corner[0])])
    elif 37 <= crate_id <= 44:
        p0 = np.array([radius3d * np.sin(ll_corner[0]) * np.cos(ll_corner[1]), radius3d * np.sin(ll_corner[0]) * np.sin(ll_corner[1]), radius3d * np.cos(ll_corner[0])])
        p1 = np.array([radius3d * np.sin(lr_corner[0]) * np.cos(lr_corner[1]), radius3d * np.sin(lr_corner[0]) * np.sin(lr_corner[1]), radius3d * np.cos(lr_corner[0])])
        p2 = np.array([radius3d * np.sin(ul_corner[0]) * np.cos(ul_corner[1]), radius3d * np.sin(ul_corner[0]) * np.sin(ul_corner[1]), radius3d * np.cos(ul_corner[0])])
        p3 = np.array([radius3d * np.sin(ur_corner[0]) * np.cos(ur_corner[1]), radius3d * np.sin(ur_corner[0]) * np.sin(ur_corner[1]), radius3d * np.cos(ur_corner[0])])
        back_p4 = np.array([distance_to_back * np.sin(ll_corner[0]) * np.cos(ll_corner[1]), distance_to_back * np.sin(ll_corner[0]) * np.sin(ll_corner[1]), distance_to_back * np.cos(ll_corner[0])])
        back_p5 = np.array([distance_to_back * np.sin(lr_corner[0]) * np.cos(lr_corner[1]), distance_to_back * np.sin(lr_corner[0]) * np.sin(lr_corner[1]), distance_to_back * np.cos(lr_corner[0])])
        back_p6 = np.array([distance_to_back * np.sin(ul_corner[0]) * np.cos(ul_corner[1]), distance_to_back * np.sin(ul_corner[0]) * np.sin(ul_corner[1]), distance_to_back * np.cos(ul_corner[0])])
        back_p7 = np.array([distance_to_back * np.sin(ur_corner[0]) * np.cos(ur_corner[1]), distance_to_back * np.sin(ur_corner[0]) * np.sin(ur_corner[1]), distance_to_back * np.cos(ur_corner[0])])    
    else: 
        p0 = np.array([radius3d * np.sin(ll_corner[0]) * np.cos(ll_corner[1]), radius3d * np.sin(ll_corner[0]) * np.sin(ll_corner[1]), radius3d * np.cos(ll_corner[0])])
        p1 = np.array([radius3d * np.sin(lr_corner[0]) * np.cos(lr_corner[1]), radius3d * np.sin(lr_corner[0]) * np.sin(lr_corner[1]), radius3d * np.cos(lr_corner[0])])
        p2 = np.array([radius3d * np.sin(ul_corner[0]) * np.cos(ul_corner[1]), radius3d * np.sin(ul_corner[0]) * np.sin(ul_corner[1]), radius3d * np.cos(ul_corner[0])])
        p3 = np.array([radius3d * np.sin(ur_corner[0]) * np.cos(ur_corner[1]), radius3d * np.sin(ur_corner[0]) * np.sin(ur_corner[1]), radius3d * np.cos(ur_corner[0])])
        back_p4 = np.array([distance_to_back * np.sin(ll_corner[0]) * np.cos(ll_corner[1]), distance_to_back * np.sin(ll_corner[0]) * np.sin(ll_corner[1]), distance_to_back * np.cos(ll_corner[0])])
        back_p5 = np.array([distance_to_back * np.sin(lr_corner[0]) * np.cos(lr_corner[1]), distance_to_back * np.sin(lr_corner[0]) * np.sin(lr_corner[1]), distance_to_back * np.cos(lr_corner[0])])
        back_p6 = np.array([distance_to_back * np.sin(ul_corner[0]) * np.cos(ul_corner[1]), distance_to_back * np.sin(ul_corner[0]) * np.sin(ul_corner[1]), distance_to_back * np.cos(ul_corner[0])])
        back_p7 = np.array([distance_to_back * np.sin(ur_corner[0]) * np.cos(ur_corner[1]), distance_to_back * np.sin(ur_corner[0]) * np.sin(ur_corner[1]), distance_to_back * np.cos(ur_corner[0])])
    points = np.array([p0, p1 , p2, p3, back_p4,  back_p5, back_p6, back_p7]) 

    #All cell types follow the same connectivity array format:
    # [Number of points, Point 1, Point 2, ...] 
    cells = np.array([[4, 0, 1, 3, 2],[4, 4, 5, 7, 6],[4, 4, 5, 7, 6],[4, 1, 3, 7, 5],[4, 2, 3, 7, 6],[4, 2, 6, 4, 0],[4, 1, 0, 4, 5]])
    
    # Create a PolyData object
    crystal = pv.PolyData(points, cells )
    return crystal


def create_detector():
    crystal_meshes = pv.MultiBlock()
    for i in range(len(tc_mapping_kart)):
        crystal_meshes.append(
            create_crystal(
                radius3d=tc_mapping_kart.iloc[i]["R_3D cm"],
                ll_corner=tc_mapping_kart.iloc[i][["corner0_theta", "corner0_phi"]].values,
                lr_corner=tc_mapping_kart.iloc[i][["corner1_theta", "corner1_phi"]].values,
                ul_corner=tc_mapping_kart.iloc[i][["corner2_theta", "corner2_phi"]].values,
                ur_corner=tc_mapping_kart.iloc[i][["corner3_theta", "corner3_phi"]].values,
                crystal_id=tc_mapping_kart.iloc[i]["Crystal ID"],
                crate_id=tc_mapping_kart.iloc[i]["CrateID"]
            ),
            name=f"crystal_{tc_mapping_kart.iloc[i]['Crystal ID']}"
        ) 
    crystal_meshes.save("/tmp/detector_mesh.vtmb")

def read_detector_mesh(file_path):
    """
    Load the detector mesh from a given file path. Assuming the individual crystal meshes are saved as multiblock data.
    """
    try:
        detector_multiblock_data = pv.read(file_path,progress_bar=True)
        print(f"Loaded {len(detector_multiblock_data)} crystal meshes from {file_path}", flush=True)
        return detector_multiblock_data
    except Exception as e:
        print("Create detector mesh from scratch...",flush=True)
        create_detector()
        print("Created detector mesh successfully",flush=True)
        detector_multiblock_data = pv.read(file_path,progress_bar=True)
        print(f"Loaded {len(detector_multiblock_data)} crystal meshes from {file_path}", flush=True)
        return detector_multiblock_data

class Detector:
    def __init__(self, path: str):
        self.crystals = read_detector_mesh(path)
        self.mesh = pv.merge(self.crystals)


if __name__ == "__main__":
    """
    Create all crystal meshes of the detector.
    """
    create_detector()