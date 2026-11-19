"""Lossless setter for a Scene's saved native frame/subframe pair."""


def set_scene_frame(scene, frame, subframe=0.0):
    """Reapply native RNA time, including negative fractional frames.

    Blender reads negative times as a truncated integer plus a negative
    subframe, but frame_set only accepts a nonnegative subframe. Normalize
    that native pair before setting it; do not round through their sum.
    This helper does not change timeline ownership or capture policy.
    """
    if subframe < 0.0:
        frame -= 1
        subframe += 1.0
    scene.frame_set(frame, subframe=subframe)
