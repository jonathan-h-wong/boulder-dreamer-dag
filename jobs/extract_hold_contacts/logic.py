"""
extract_hold_contacts logic — on-route and volume mounting from geometry, not the agent.

Inputs: poses.npy (GVHMR, full sequence), intrinsics.json (shared static K), bkg_depth.png,
holds/proposals/masks.npz, holds/hold_descriptions.csv.

Steps (thresholds from config hold_contacts):
1. Contacts: posed SMPL mesh per frame -> hand and foot vertex regions (not just wrist /
   ankle joints) -> project with the shared K. A contact is a run of >= min_contact_frames
   frames where the point moves less than max_contact_speed_px per frame AND its depth is
   within wall_depth_tol_m of the wall depth at that pixel.
2. Each contact touches the hold whose mask (dilated by mask_dilate_px) contains the
   projected point. on_route = touched at least once.
3. Mounting: for each non-volume hold, overlap with each volume's bbox measured as
   intersection / hold-bbox area (containment; symmetric IoU is tiny for a small hold on a
   large volume). Containment >= mount_min_containment -> mounted_on = that volume's
   hold_id (largest containment wins).
4. A volume is on_route if touched directly or if any hold mounted on it is.
5. Upload processed/{video_id}/holds/hold_contacts.csv:
     hold_id, on_route, mounted_on, n_contacts, contact_limbs
   (joined with hold_descriptions.csv on hold_id by downstream stages)
"""


def run(video_id: str, config: dict):
    # TODO: implement
    raise NotImplementedError
