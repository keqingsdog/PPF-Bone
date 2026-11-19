"""Explicit, undoable material preset Apply; enum selection is read-only."""
from .. import i18n as _ppf_i18n
import bpy
from ..i18n.props import EnumProperty
from ..models.bone_bindings import active_armature, validate_proxy_owner
from ..models.bone_material_presets import PRESETS, preset_values
from ..models.new_binding_defaults import new_binding_plan
from ..core.bone_mass import _owned, _measure, material_write_guard, mark_settings
from .bone_proxy import SCOPES, selected_bones, _owner_animation_paths


def apply_material_preset(context, scope='SELECTED'):
    from . import bone_proxy as proxy
    proxy._facade_communicator().assert_bone_body_switch_idle()
    if proxy.stale_simulation_output(context):
        raise ValueError('请先 Clear 已有模拟帧，再应用材质预设')
    arm, bones = selected_bones(context, scope=scope)
    planned = []
    for bone in bones:
        validate_proxy_owner(arm, bone)
        s = bone.ppf_bone_physics
        model = new_binding_plan(s).get('physical_model', s.physical_model)
        if model != 'SOFT': raise ValueError(f'{bone.name}：材质预设需要软体 SOLID 物理体')
        group = _owned(context.scene, arm, bone) if s.proxy else None
        paths = [path for block, label in ((context.scene, '场景'), (arm.data, '骨架数据'), (arm, '骨架对象'))
                 for path in _owner_animation_paths(block, label)]
        prefix = bone.path_from_id()+'.ppf_bone_physics.'
        if any(path.startswith(prefix) for path in paths):
            raise ValueError(f'{bone.name}：带动画或驱动的骨骼材质不能应用预设')
        volume = _measure(context.scene, s)['volume_m3'] if s.mass_mode == 'MASS' else None
        values = preset_values(s.material_preset, s.mass_mode, s.target_mass_kg, volume)
        # An explicit preset replaces E and density together. The density-only
        # path deliberately rejects stale authors and must not reinterpret this
        # newly requested material as an accidental author/group drift.
        material_plan = (proxy._plan_copied_soft_material(
            s, group, values, values['density'], group.young_mod_density_normalized)
            if group is not None else None)
        planned.append((s, group, values, material_plan))
    # Snapshot before any target changes so a failure restores the entire batch.
    from .restoring_curve_operators import snapshot_rna, restore_rna
    snapshots = [(s, snapshot_rna(s)) for s, _group, _values, _plan in planned]
    snapshots += [(group, snapshot_rna(group)) for _s, group, _values, _plan in planned if group is not None]
    try:
        with material_write_guard():
            for s, group, values, material_plan in planned:
                for key, value in values.items(): setattr(s, key, value)
                if group is not None: proxy._apply_copied_soft_material(s, group, material_plan)
    except Exception:
        with material_write_guard():
            for owner, snapshot in reversed(snapshots): restore_rna(owner, snapshot)
        raise
    for s, _group, _values, _plan in planned: mark_settings(s, context)
    return len(planned)


class PPF_OT_BoneMaterialPresetApply(bpy.types.Operator):
    bl_idname = 'ppf.bone_material_preset_apply'
    bl_label = '应用骨骼材质预设'
    bl_description = '应用所选材质预设；固定目标质量模式保留目标质量，关节参数独立设置'
    bl_options = {'REGISTER', 'UNDO'}
    scope: EnumProperty(name='骨骼范围', items=SCOPES, default='SELECTED')

    @classmethod
    def poll(cls, context):
        return active_armature(context) is not None and context.mode in {'OBJECT', 'POSE'}

    def execute(self, context):
        try: count = apply_material_preset(context, self.scope)
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error)); return {'CANCELLED'}
        self.report({'INFO'}, f'已应用 {count} 个骨骼的材质预设；关节参数独立设置')
        return {'FINISHED'}


def register(): _ppf_i18n.register_class(PPF_OT_BoneMaterialPresetApply)
def unregister(): bpy.utils.unregister_class(PPF_OT_BoneMaterialPresetApply)
