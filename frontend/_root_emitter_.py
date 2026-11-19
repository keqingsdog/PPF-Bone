"""One scene-bound producer choice for both ordinary and mixed roots."""
def emit_roots(fixed, roots, body_controls=False):
    from ._build_contract_ import contract_for
    version = contract_for(fixed).producer_version
    if body_controls:
        setter = fixed.set_pdrd_body_controls_v3 if version == 3 else fixed.set_pdrd_body_controls_v1
    else:
        setter = fixed.set_pdrd_root_motion_v3 if version == 3 else fixed.set_pdrd_root_motion_v2
    setter({'roots': roots})
