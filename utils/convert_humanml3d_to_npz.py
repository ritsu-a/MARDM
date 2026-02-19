"""
Convert humanml3d pickle files to npz format with fps interpolation.
Converts from 20fps to 60fps (3x interpolation).
"""

import argparse
import pathlib
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy import signal
from tqdm import tqdm
import sys

# Output qpos dimension: same as G1_brainco (semi_synthetic format). 3 root_pos + 4 root_rot + 53 dof = 60.
# G1_brainco_hands.xml joint order (after freejoint): 0-21 body (legs, waist, left arm), 22-33 left hand, 34-40 right arm, 41-52 right hand.
# So in qpos[7:60]: indices 22-33 (qpos 29-40) = left hand, indices 41-52 (qpos 48-59) = right hand -> set to 0.
QPOS_DIM = 60
# Hand DOF indices in qpos (0-based). Left hand: 29-40, Right hand: 48-59 (from G1_brainco_hands.xml joint order).
G1_BRAINCO_HAND_QPOS_INDICES = list(range(29, 41)) + list(range(48, 60))

# Try to import torch and vec6d_to_quat, fallback to numpy implementation
try:
    import torch
    sys.path.append(str(pathlib.Path(__file__).parent.parent / "external" / "GMR" / "scripts"))
    from diff_quat import vec6d_to_quat as vec6d_to_quat_torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not available, using numpy implementation for 6D rotation conversion")


def normalize_vec6d_numpy(x):
    """
    Normalize 6D rotation representation to get orthonormal basis.
    Input shape: (..., 3, 2)
    """
    # Normalize first column
    x = x / np.linalg.norm(x, axis=-2, keepdims=True)
    
    first_col = x[..., 0]  # (..., 3)
    second_col = x[..., 1]  # (..., 3)
    
    # Compute third column as cross product
    last_col = np.cross(first_col, second_col, axis=-1)
    last_col = last_col / np.linalg.norm(last_col, axis=-1, keepdims=True)
    
    # Recompute second column to ensure orthonormality
    second_col = np.cross(-first_col, last_col, axis=-1)
    second_col = second_col / np.linalg.norm(second_col, axis=-1, keepdims=True)
    
    return first_col, second_col, last_col


def vec6d_to_matrix_numpy(x):
    """
    Convert 6D rotation representation to rotation matrix.
    Input shape: (..., 3, 2)
    Output shape: (..., 3, 3)
    """
    first_col, second_col, last_col = normalize_vec6d_numpy(x)
    # Stack columns to form rotation matrix
    mat = np.stack([first_col, second_col, last_col], axis=-1)
    return mat


def vec6d_to_quat_numpy(x):
    """
    Convert 6D rotation representation to quaternion (xyzw format).
    Input shape: (..., 3, 2)
    Output shape: (..., 4)
    """
    # Convert to rotation matrix
    mat = vec6d_to_matrix_numpy(x)
    
    # Reshape to (N, 3, 3) for scipy
    original_shape = x.shape[:-2]
    mat_flat = mat.reshape(-1, 3, 3)
    
    # Convert rotation matrices to quaternions using scipy
    rotations = R.from_matrix(mat_flat)
    quat_xyzw = rotations.as_quat()  # (N, 4) in xyzw format
    
    # Reshape back to original shape
    quat_xyzw = quat_xyzw.reshape(*original_shape, 4)
    
    return quat_xyzw


def vec6d_to_quat(x):
    """
    Wrapper function that uses torch if available, otherwise numpy.
    """
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return vec6d_to_quat_torch(x).numpy()
    elif HAS_TORCH:
        return vec6d_to_quat_torch(torch.tensor(x)).numpy()
    else:
        return vec6d_to_quat_numpy(np.array(x))


def quat_slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions.
    
    Args:
        q1: First quaternion (w, x, y, z)
        q2: Second quaternion (w, x, y, z)
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated quaternion (w, x, y, z)
    """
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.dot(q1, q2)
    
    # If dot product is negative, negate one quaternion to take shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Clamp dot product to avoid numerical errors
    dot = np.clip(dot, -1.0, 1.0)
    
    # If quaternions are very close, use linear interpolation
    if abs(dot) > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Compute angle
    theta = np.arccos(abs(dot))
    sin_theta = np.sin(theta)
    
    # SLERP formula
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    result = w1 * q1 + w2 * q2
    return result / np.linalg.norm(result)


def low_pass_filter(data, cutoff_freq=0.15, order=4, fps=60):
    """
    Apply a low-pass Butterworth filter to the data to reduce jitter.
    
    Parameters:
        data (numpy.ndarray): The input data to be filtered, shape (N, D).
        cutoff_freq (float): Cutoff frequency in Hz (default: 0.15 Hz for 60fps).
        order (int): The order of the Butterworth filter (default: 4).
        fps (float): Frame rate of the data (default: 60).
        
    Returns:
        numpy.ndarray: The filtered data with same shape.
    """
    # Convert Hz to normalized frequency (0~1) for scipy.signal.butter
    # Nyquist frequency = fps / 2
    nyquist = fps / 2.0
    normalized_cutoff = cutoff_freq / nyquist
    
    # Ensure cutoff is less than 1.0 (Nyquist)
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99
    
    b, a = signal.butter(order, normalized_cutoff, 'low')
    filtered_data = data.copy()
    
    # Apply filter to each dimension independently
    for idx in range(filtered_data.shape[1]):
        filtered_data[:, idx] = signal.filtfilt(b, a, data[:, idx])
    
    return filtered_data


def interpolate_qpos(qpos_original, src_fps=20, tgt_fps=60):
    """
    Interpolate qpos data from src_fps to tgt_fps.
    
    Args:
        qpos_original: numpy array of shape (N, D) where D = 3 (root_pos) + 4 (root_rot) + num_dofs
        src_fps: Source frame rate
        tgt_fps: Target frame rate
    
    Returns:
        qpos_interpolated: numpy array of shape (M, D) where M = N * (tgt_fps / src_fps)
    """
    num_frames_original = qpos_original.shape[0]
    num_frames_target = int(num_frames_original * tgt_fps / src_fps)
    
    # Create time arrays
    original_time = np.arange(num_frames_original)
    target_time = np.linspace(0, num_frames_original - 1, num_frames_target)
    
    # Extract components
    root_pos = qpos_original[:, :3]  # (N, 3)
    root_rot = qpos_original[:, 3:7]  # (N, 4) in wxyz format
    dof_pos = qpos_original[:, 7:]  # (N, num_dofs)
    
    # Interpolate root position (linear interpolation)
    root_pos_interp = np.zeros((num_frames_target, 3))
    for i in range(3):
        interp_func = interp1d(original_time, root_pos[:, i], kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        root_pos_interp[:, i] = interp_func(target_time)
    
    # Interpolate root rotation (SLERP)
    root_rot_interp = np.zeros((num_frames_target, 4))
    for i in range(num_frames_target):
        t = target_time[i]
        idx1 = int(np.floor(t))
        idx2 = min(idx1 + 1, num_frames_original - 1)
        alpha = t - idx1
        
        q1 = root_rot[idx1]
        q2 = root_rot[idx2]
        root_rot_interp[i] = quat_slerp(q1, q2, alpha)
    
    # Interpolate DOF positions (linear interpolation)
    num_dofs = dof_pos.shape[1]
    dof_pos_interp = np.zeros((num_frames_target, num_dofs))
    for i in range(num_dofs):
        interp_func = interp1d(original_time, dof_pos[:, i], kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        dof_pos_interp[:, i] = interp_func(target_time)
    
    # Combine interpolated components
    qpos_interpolated = np.zeros((num_frames_target, qpos_original.shape[1]))
    qpos_interpolated[:, :3] = root_pos_interp
    qpos_interpolated[:, 3:7] = root_rot_interp
    qpos_interpolated[:, 7:] = dof_pos_interp
    
    return qpos_interpolated


def load_humanml3d_pickle(pickle_path):
    """
    Load humanml3d pickle data and convert to 60-dim qpos format for G1_brainco.
    Body DOFs are mapped according to G1_brainco_hands.xml joint order; hand DOFs are set to 0.
    
    Args:
        pickle_path: Path to the pickle file
        
    Returns:
        qpos: numpy array of shape (N, 60)
        fps: frame rate
        
    Raises:
        ValueError: If the pickle file contains NaN or Inf values
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    
    global_translation = data["global_translation"][:, :, 0]  # (N, 3)
    global_rotation = data["global_rotation"]  # (N, 3, 2) - 6D rotation
    angles = data["angles"]  # (N, 29) for g1_29
    fps = data.get("fps", 20)
    
    # Check for NaN or Inf values in input data
    if np.isnan(global_translation).any() or np.isinf(global_translation).any():
        raise ValueError(f"NaN or Inf values found in global_translation")
    if np.isnan(global_rotation).any() or np.isinf(global_rotation).any():
        raise ValueError(f"NaN or Inf values found in global_rotation")
    if np.isnan(angles).any() or np.isinf(angles).any():
        raise ValueError(f"NaN or Inf values found in angles")
    
    quat_xyzw = vec6d_to_quat(global_rotation)
    quat_wxyz = np.zeros_like(quat_xyzw)
    quat_wxyz[:, 0] = quat_xyzw[:, 3]
    quat_wxyz[:, 1] = quat_xyzw[:, 0]
    quat_wxyz[:, 2] = quat_xyzw[:, 1]
    quat_wxyz[:, 3] = quat_xyzw[:, 2]
    
    num_frames = global_translation.shape[0]
    # Output 60-dim qpos: root(7) + 53 joints. Map 29 body angles to G1_brainco order, hand = 0.
    # G1_brainco dof order (after root): left_leg(6), right_leg(6), waist(3), left_arm(7), left_hand(12), right_arm(7), right_hand(12)
    qpos = np.zeros((num_frames, QPOS_DIM))
    qpos[:, :3] = global_translation
    qpos[:, 3:7] = quat_wxyz
    # angles: assumed same order as G1 body = left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)
    qpos[:, 7:22] = angles[:, 0:15]   # legs + waist
    qpos[:, 22:29] = angles[:, 15:22] # left arm
    qpos[:, 29:41] = 0.0              # left hand (G1_brainco hand DOF indices)
    qpos[:, 41:48] = angles[:, 22:29]  # right arm
    qpos[:, 48:60] = 0.0              # right hand
    
    return qpos, fps


def convert_pickle_to_npz(pickle_path, output_path, src_fps=20, tgt_fps=60, filter_cutoff=0.15, filter_order=4):
    """
    Convert humanml3d pickle file to npz format with fps interpolation and low-pass filtering.
    
    Args:
        pickle_path: Path to input pickle file
        output_path: Path to output npz file
        src_fps: Source frame rate (default: 20)
        tgt_fps: Target frame rate (default: 60)
        filter_cutoff: Low-pass filter cutoff frequency in Hz (default: 0.15)
        filter_order: Low-pass filter order (default: 4)
    """
    try:
        qpos, data_fps = load_humanml3d_pickle(pickle_path)
        
        # Check for NaN/Inf in loaded qpos (shouldn't happen if input is clean, but double-check)
        if np.isnan(qpos).any() or np.isinf(qpos).any():
            print(f"Warning: NaN/Inf detected in qpos after loading {pickle_path}, skipping")
            return False
        
        # Use fps from data if available, otherwise use src_fps
        if data_fps != src_fps:
            src_fps = data_fps
        
        # Interpolate qpos
        qpos_interpolated = interpolate_qpos(qpos, src_fps=src_fps, tgt_fps=tgt_fps)
        
        # Check for NaN/Inf after interpolation
        if np.isnan(qpos_interpolated).any() or np.isinf(qpos_interpolated).any():
            print(f"Warning: NaN/Inf detected after interpolation for {pickle_path}, skipping")
            return False
        
        # Apply low-pass filter to reduce jitter
        qpos_filtered = low_pass_filter(qpos_interpolated, cutoff_freq=filter_cutoff, order=filter_order, fps=tgt_fps)
        
        # Final check before saving
        if np.isnan(qpos_filtered).any() or np.isinf(qpos_filtered).any():
            print(f"Warning: NaN/Inf detected after filtering for {pickle_path}, skipping")
            return False
        
        # Save as npz
        output_path_obj = pathlib.Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, qpos=qpos_filtered)
        
        return True
    except ValueError as e:
        # ValueError indicates NaN/Inf in input data - skip this file
        print(f"Skipping {pickle_path}: {e}")
        return False
    except Exception as e:
        print(f"Error converting {pickle_path}: {e}")
        return False


def batch_convert_pickle_to_npz(input_dir, output_dir, src_fps=20, tgt_fps=60, filter_cutoff=0.15, filter_order=4):
    """
    Batch convert all pickle files in a directory to npz format.
    
    Args:
        input_dir: Directory containing pickle files
        output_dir: Directory to save npz files
        src_fps: Source frame rate (default: 20)
        tgt_fps: Target frame rate (default: 60)
        filter_cutoff: Low-pass filter cutoff frequency in Hz (default: 0.15)
        filter_order: Low-pass filter order (default: 4)
    """
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all pickle files
    pickle_files = sorted(input_path.glob("*.pickle"))
    
    if len(pickle_files) == 0:
        print(f"No pickle files found in {input_dir}")
        return
    
    print(f"Found {len(pickle_files)} pickle files to convert")
    
    # Process each file
    success_count = 0
    skipped_count = 0
    for pickle_file in tqdm(pickle_files, desc="Converting"):
        # Generate output filename
        output_file = output_path / (pickle_file.stem + ".npz")
        
        if convert_pickle_to_npz(
            str(pickle_file),
            str(output_file),
            src_fps=src_fps,
            tgt_fps=tgt_fps,
            filter_cutoff=filter_cutoff,
            filter_order=filter_order
        ):
            success_count += 1
    
    print(f"\nBatch conversion complete! Successfully processed {success_count}/{len(pickle_files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert humanml3d pickle files to npz format with fps interpolation")
    
    parser.add_argument(
        "--pickle_path",
        type=str,
        default=None,
        help="Path to input pickle file (for single file conversion)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output npz file (for single file conversion)"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing pickle files (for batch conversion)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save npz files (for batch conversion)"
    )
    
    parser.add_argument(
        "--src_fps",
        type=int,
        default=20,
        help="Source frame rate (default: 20)"
    )
    
    parser.add_argument(
        "--tgt_fps",
        type=int,
        default=60,
        help="Target frame rate (default: 60)"
    )
    
    parser.add_argument(
        "--filter_cutoff",
        type=float,
        default=2.0,
        help="Low-pass filter cutoff frequency in Hz (default: 2.0)"
    )
    
    parser.add_argument(
        "--filter_order",
        type=int,
        default=4,
        help="Low-pass filter order (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Check if batch mode or single file mode
    if args.input_dir and args.output_dir:
        # Batch conversion mode
        batch_convert_pickle_to_npz(
            args.input_dir,
            args.output_dir,
            src_fps=args.src_fps,
            tgt_fps=args.tgt_fps,
            filter_cutoff=args.filter_cutoff,
            filter_order=args.filter_order
        )
    elif args.pickle_path and args.output_path:
        # Single file conversion mode
        convert_pickle_to_npz(
            args.pickle_path,
            args.output_path,
            src_fps=args.src_fps,
            tgt_fps=args.tgt_fps,
            filter_cutoff=args.filter_cutoff,
            filter_order=args.filter_order
        )
    else:
        parser.error("Either provide --pickle_path and --output_path (single file) or --input_dir and --output_dir (batch)")
