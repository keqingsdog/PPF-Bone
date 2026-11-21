"""Proxy material starting points; calibration metadata is deliberately bounded."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BoneMaterialPreset:
    label: str
    young_pa: float
    density_kg_m3: float
    poisson: float
    deformation_damping_s: float
    evidence: str


PRESETS = {
    'STIFF_RIGID': BoneMaterialPreset('高硬度软体', 1e9, 1000., .3, .002,
        '仅静态根、两个软体 BOX、1 秒重力夹具的有限证据；长链和胶囊尚未标定'),
    'LIGHT_HAIR': BoneMaterialPreset('轻质发片（未标定起点）', 1e8, 50., .3, .002, '未标定的代理效果起点'),
    'SOFT_CLOTH': BoneMaterialPreset('柔软布片（未标定起点）', 1e7, 100., .3, .002, '未标定的代理效果起点'),
    'STIFF_STRAP': BoneMaterialPreset('硬挺织带（未标定起点）', 1e8, 200., .3, .002, '未标定的代理效果起点'),
    'HARD_WEBBING': BoneMaterialPreset('硬质织带（未标定起点）', 2e8, 200., .3, .002, '未标定的代理效果起点'),
}
PRESET_ITEMS = [(key, value.label, '待应用的软体材质；选择不等于已应用，不切换物理类型、形状或关节') for key, value in PRESETS.items()]


def preset_values(identifier, mode, target_mass_kg, volume_m3=None):
    from ..core.bone_mass_math import density_for_mass, valid_density
    try: preset = PRESETS[identifier]
    except KeyError: raise ValueError('未知的骨骼材质预设') from None
    density = density_for_mass(target_mass_kg, volume_m3) if mode == 'MASS' else valid_density(preset.density_kg_m3)
    if mode not in ('MASS', 'DENSITY'): raise ValueError('未知的质量输入模式')
    return {'solid_model': 'STABLE_NEOHOOKEAN', 'solid_young_modulus': preset.young_pa,
            'solid_poisson_ratio': preset.poisson, 'density': density,
            'solid_deformation_damping': preset.deformation_damping_s}


def material_preset_matches(settings, group):
    """Read-only current material comparison, never an Apply-history claim."""
    import math
    preset=PRESETS.get(settings.material_preset)
    if preset is None or group is None or group.object_type!='SOLID':return False
    young=float(group.solid_young_modulus)
    if group.young_mod_density_normalized:young*=float(group.solid_density)
    pairs=[(young,preset.young_pa),(float(group.solid_poisson_ratio),preset.poisson),
           (float(group.deformation_damping),preset.deformation_damping_s)]
    if settings.mass_mode=='DENSITY':pairs.append((float(group.solid_density),preset.density_kg_m3))
    return (group.solid_model=='STABLE_NEOHOOKEAN'
            and all(math.isclose(a,b,rel_tol=2.e-6,abs_tol=1.e-8) for a,b in pairs))
