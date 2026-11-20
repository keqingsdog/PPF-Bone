"""Display-only palette for the existing sixteen collision membership slots."""

# Fixed colors avoid Python's randomized hash and stay stable across sessions.
COLLISION_COLORS = (
    (0.95, 0.38, 0.32), (0.25, 0.67, 0.96), (0.35, 0.78, 0.42), (0.98, 0.69, 0.22),
    (0.72, 0.43, 0.90), (0.20, 0.78, 0.73), (0.98, 0.48, 0.68), (0.67, 0.65, 0.28),
    (0.48, 0.52, 0.94), (0.89, 0.55, 0.29), (0.44, 0.75, 0.58), (0.82, 0.43, 0.55),
    (0.42, 0.69, 0.82), (0.73, 0.57, 0.35), (0.60, 0.48, 0.72), (0.64, 0.76, 0.31),
)


def collision_group_color(settings, fallback):
    """Use the lowest selected membership; disabled/empty retains legacy tint."""
    if not getattr(settings, 'collision_filter_enabled', False):
        return fallback
    for index, enabled in enumerate(getattr(settings, 'collision_group_layers', ())):
        if enabled and index < len(COLLISION_COLORS):
            return COLLISION_COLORS[index]
    return fallback
