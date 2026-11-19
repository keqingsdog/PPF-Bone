# Fork modification: changed from the upstream baseline.
# File: profile_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Profile operators for scene and material parameter profiles.

from ... import i18n as _ppf_i18n
import bpy  # pyright: ignore
from ...i18n import pgettext_iface as iface_  # pyright: ignore
from ...i18n.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ...models.material_locks import locked_props
from ...core.param_introspect import (
    MATERIAL_CLIPBOARD_EXCLUDE,
    PIN_OP_CLIPBOARD_EXCLUDE,
    copy_scalar_props,
    material_param_applies,
)
from ...core.utils import redraw_all_areas
from ...models.groups import get_addon_data, invalidate_overlays
from .utils import get_group_from_index

_SAVE_MODES = [('AUTO', 'Auto', ''), ('BOUND', 'Bound', ''), ('PATH', 'Path', '')]

def _invoke_profile_save(operator, context, owner, kind, default_filename):
    from ...core.profile_identity import save_mode
    if operator.save_mode == 'AUTO' and not operator.properties.is_property_set('filepath'):
        if owner and getattr(owner, kind + '_profile_path') and getattr(owner, kind + '_profile_selection') != 'NONE':
            operator.save_mode = 'BOUND'
            return operator.execute(context)
        operator.save_mode = 'PATH'
        operator.filepath = default_filename
        context.window_manager.fileselect_add(operator)
        return {'RUNNING_MODAL'}
    if save_mode(operator) == 'BOUND':
        return operator.execute(context)
    operator.save_mode = 'PATH'
    if operator.properties.is_property_set('filepath'):
        return operator.execute(context)
    operator.filepath = default_filename
    context.window_manager.fileselect_add(operator)
    return {'RUNNING_MODAL'}


class SCENE_OT_OpenSceneProfile(Operator):
    """Open a TOML scene parameter profile file."""

    bl_idname = "scene.open_scene_profile"
    bl_label = "Open Scene Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "scene_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ...core.profile import load_profiles, validate_scene_profile

        state = get_addon_data(context.scene).state
        abs_path = bpy.path.abspath(self.filepath)
        try:
            profiles = load_profiles(abs_path, strict=True)
            names = sorted(profiles)
            from ...core.profile_identity import validate_name
            for name in names:validate_name(name)
            if names:
                validate_scene_profile(profiles[names[0]], state)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        state.scene_profile_path = self.filepath
        state.scene_profile_selection = names[0] if names else 'NONE'
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_ClearSceneProfile(Operator):
    """Clear the loaded scene parameter profile."""

    bl_idname = "scene.clear_scene_profile"
    bl_label = "Clear Scene Profile"

    def execute(self, context):
        state = get_addon_data(context.scene).state
        state.scene_profile_path = ""
        state.scene_profile_selection = "NONE"
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_OpenMaterialProfile(Operator):
    """Open a TOML material parameter profile file."""

    bl_idname = "object.open_material_profile"
    bl_label = "Open Material Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "material_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ...core.profile import load_profiles, validate_material_profile

        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        if group is None:
            return {"CANCELLED"}
        abs_path = bpy.path.abspath(self.filepath)
        profiles = load_profiles(abs_path)
        names = sorted(profiles)
        if not names:
            self.report({"ERROR"}, iface_("No readable material profiles found"))
            return {"CANCELLED"}
        try:
            from ...core.profile_identity import validate_name
            for name in names:validate_name(name)
            validate_material_profile(profiles[names[0]], group)
        except (ValueError, TypeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        group.material_profile_path = self.filepath
        group.material_profile_selection = names[0]
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_ClearMaterialProfile(Operator):
    """Clear the loaded material parameter profile."""

    bl_idname = "object.clear_material_profile"
    bl_label = "Clear Material Profile"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        if group is None:
            return {"CANCELLED"}
        group.material_profile_path = ""
        group.material_profile_selection = "NONE"
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_ReloadSceneProfile(Operator):
    """Re-apply the currently selected scene profile."""

    bl_idname = "scene.reload_scene_profile"
    bl_label = "Reload Scene Profile"

    def execute(self, context):
        from ...core.profile import apply_scene_profile, load_profiles

        state = get_addon_data(context.scene).state
        from ...core.profile_identity import selected_profile
        try:
            profile = selected_profile(state, 'scene')
            apply_scene_profile(profile, state)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        from ...models.groups import invalidate_overlays
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_SaveSceneProfile(Operator):
    """Save current scene parameters to a profile."""

    bl_idname = "scene.save_scene_profile"
    bl_label = "Save Scene Profile"

    filepath: StringProperty(subtype="FILE_PATH", options={'SKIP_SAVE'})  # pyright: ignore
    save_mode: EnumProperty(items=_SAVE_MODES, default='AUTO', options={'HIDDEN', 'SKIP_SAVE'})
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    entry_name: StringProperty(name="Entry Name", default="Default")  # pyright: ignore

    def invoke(self, context, event):
        state = get_addon_data(context.scene).state
        return _invoke_profile_save(self, context, state, 'scene', 'scene_profile.toml')

    def execute(self, context):
        from ...core.profile import read_scene_profile, save_profile_entry

        state = get_addon_data(context.scene).state
        data = read_scene_profile(state)
        from ...core.profile_identity import save_selected, save_mode, bind_saved
        try:
            name, use_bound = save_selected(state, 'scene', data, self.filepath, self.entry_name, save_mode(self))
            if not use_bound:
                bind_saved(state, 'scene', self.filepath, name)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, iface_("Saved to '{name}'").format(name=name))
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_ReloadMaterialProfile(Operator):
    """Re-apply the currently selected material profile."""

    bl_idname = "object.reload_material_profile"
    bl_label = "Reload Material Profile"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        from ...core.profile import apply_material_profile, load_profiles

        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        from ...core.profile_identity import selected_profile
        try:
            profile = selected_profile(group, 'material')
            apply_material_profile(profile, group)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        from ...models.groups import invalidate_overlays
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_SaveMaterialProfile(Operator):
    """Save current material parameters to a profile."""

    bl_idname = "object.save_material_profile"
    bl_label = "Save Material Profile"

    filepath: StringProperty(subtype="FILE_PATH", options={'SKIP_SAVE'})  # pyright: ignore
    save_mode: EnumProperty(items=_SAVE_MODES, default='AUTO', options={'HIDDEN', 'SKIP_SAVE'})
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    entry_name: StringProperty(name="Entry Name", default="Default")  # pyright: ignore
    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        return _invoke_profile_save(self, context, group, 'material', 'material_profile.toml')

    def execute(self, context):
        from ...core.profile import read_material_profile, save_profile_entry

        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        data = read_material_profile(group)
        from ...core.profile_identity import save_selected, save_mode, bind_saved
        try:
            name, use_bound = save_selected(group, 'material', data, self.filepath, self.entry_name, save_mode(self))
            if not use_bound:
                bind_saved(group, 'material', self.filepath, name)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, iface_("Saved to '{name}'").format(name=name))
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_CopyMaterialParams(Operator):
    """Copy material parameters to clipboard."""

    bl_idname = "object.copy_material_params"
    bl_label = "Copy Material Params"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        wm = context.window_manager
        copy_scalar_props(group, wm.material_clipboard, exclude=MATERIAL_CLIPBOARD_EXCLUDE)
        wm.material_clipboard_src_type = group.object_type
        wm.material_clipboard_valid = True
        self.report({"INFO"}, iface_("Material params copied"))
        return {"FINISHED"}


class OBJECT_OT_PasteMaterialParams(Operator):
    """Paste material parameters from clipboard."""

    bl_idname = "object.paste_material_params"
    bl_label = "Paste Material Params"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return getattr(context.window_manager, "material_clipboard_valid", False)

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        wm = context.window_manager
        # Filter model-specific params by the source's object_type so
        # cross-type pastes don't overwrite the target's unused
        # shell_*/solid_*/rod_* fields with the source's defaults.
        src_type = wm.material_clipboard_src_type or group.object_type
        # A locked parameter is skipped, which is what the padlock beside it
        # promises: the lock guards the value from the tools that overwrite a
        # whole group at once, exactly as Transform > Location's lock guards
        # against transform tools. It does not mute an F-curve.
        locked = locked_props(group)
        copy_scalar_props(
            wm.material_clipboard,
            group,
            exclude=MATERIAL_CLIPBOARD_EXCLUDE,
            filter_fn=lambda n: material_param_applies(n, src_type)
            and n not in locked,
        )
        redraw_all_areas(context)
        if locked:
            # Say what was NOT pasted. A silent skip looks identical to a
            # paste that did nothing, and the artist cannot see which values
            # the clipboard held.
            self.report(
                {"INFO"},
                iface_("Material params pasted; %d locked value(s) kept")
                % len(locked),
            )
        else:
            self.report({"INFO"}, iface_("Material params pasted"))
        return {"FINISHED"}


def _get_selected_pin(context, group_index):
    """Get the selected pin vertex group item for a given object group."""
    group = get_group_from_index(context.scene, group_index)
    if group is None or group.pin_vertex_groups_index < 0:
        return None, None
    idx = group.pin_vertex_groups_index
    if idx >= len(group.pin_vertex_groups):
        return None, None
    return group, group.pin_vertex_groups[idx]


class OBJECT_OT_OpenPinProfile(Operator):
    """Open a TOML pin operations profile file."""

    bl_idname = "object.open_pin_profile"
    bl_label = "Open Pin Operations Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "pin_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ...core.profile import load_profiles, validate_pin_operations

        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        if group is None:
            return {"CANCELLED"}
        abs_path = bpy.path.abspath(self.filepath)
        try:
            profiles = load_profiles(abs_path, strict=True)
            names = sorted(profiles)
            from ...core.profile_identity import validate_name
            for name in names:validate_name(name)
            if names:
                # A library may be opened before a Pin is selected or created.
                # Validate its schema without creating temporary author RNA.
                pin_rna = group.bl_rna.properties["pin_vertex_groups"].fixed_type
                validate_pin_operations(profiles[names[0]], pin_rna)
                _, target = _get_selected_pin(context, self.group_index)
                if target is not None:
                    from ...core.profile import validate_pin_source_replacement
                    validate_pin_source_replacement(profiles[names[0]], target)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        group.pin_profile_path = self.filepath
        group.pin_profile_selection = names[0] if names else 'NONE'
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_ClearPinProfile(Operator):
    """Clear the loaded pin operations profile."""

    bl_idname = "object.clear_pin_profile"
    bl_label = "Clear Pin Profile"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        if group is None:
            return {"CANCELLED"}
        group.pin_profile_path = ""
        group.pin_profile_selection = "NONE"
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_ReloadPinProfile(Operator):
    """Re-apply the currently selected pin operations profile."""

    bl_idname = "object.reload_pin_profile"
    bl_label = "Reload Pin Profile"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        from ...core.profile import apply_pin_operations, load_profiles

        group, pin_item = _get_selected_pin(context, self.group_index)
        if pin_item is None:
            return {"CANCELLED"}
        from ...core.profile_identity import selected_profile
        try:
            profile = selected_profile(group, 'pin')
            apply_pin_operations(profile, pin_item)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        from ...models.groups import invalidate_overlays
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_SavePinProfile(Operator):
    """Save current pin operations to a profile."""

    bl_idname = "object.save_pin_profile"
    bl_label = "Save Pin Operations Profile"

    filepath: StringProperty(subtype="FILE_PATH", options={'SKIP_SAVE'})  # pyright: ignore
    save_mode: EnumProperty(items=_SAVE_MODES, default='AUTO', options={'HIDDEN', 'SKIP_SAVE'})
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    entry_name: StringProperty(name="Entry Name", default="Default")  # pyright: ignore
    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        return _invoke_profile_save(self, context, group, 'pin', 'pin_profile.toml')

    def execute(self, context):
        from ...core.profile import read_pin_operations, save_profile_entry

        group, pin_item = _get_selected_pin(context, self.group_index)
        if pin_item is None:
            self.report({"WARNING"}, iface_("No pin vertex group selected"))
            return {"CANCELLED"}
        data = read_pin_operations(pin_item)
        from ...core.profile_identity import save_selected, save_mode, bind_saved
        try:
            name, use_bound = save_selected(group, 'pin', data, self.filepath, self.entry_name, save_mode(self))
            if not use_bound:
                bind_saved(group, 'pin', self.filepath, name)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, iface_("Saved to '{name}'").format(name=name))
        redraw_all_areas(context)
        return {"FINISHED"}


def _copy_operations_collection(src_ops, dst_ops):
    """Replace *dst_ops* with a deep copy of *src_ops*. Field set is
    discovered from ``PinOperation.bl_rna.properties`` — new fields
    added to PinOperation flow through automatically.
    """
    dst_ops.clear()
    for src_op in src_ops:
        new_op = dst_ops.add()
        copy_scalar_props(src_op, new_op, exclude=PIN_OP_CLIPBOARD_EXCLUDE)


class OBJECT_OT_CopyPinOps(Operator):
    """Copy pin operations to clipboard."""

    bl_idname = "object.copy_pin_ops"
    bl_label = "Copy Pin Operations"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        _, pin_item = _get_selected_pin(context, self.group_index)
        if pin_item is None:
            self.report({"WARNING"}, iface_("No pin vertex group selected"))
            return {"CANCELLED"}
        wm = context.window_manager
        _copy_operations_collection(pin_item.operations, wm.pin_ops_clipboard.operations)
        wm.pin_ops_clipboard_valid = True
        self.report({"INFO"}, iface_("Copied {count} operations").format(count=len(pin_item.operations)))
        return {"FINISHED"}


class OBJECT_OT_PastePinOps(Operator):
    """Paste pin operations from clipboard."""

    bl_idname = "object.paste_pin_ops"
    bl_label = "Paste Pin Operations"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return getattr(context.window_manager, "pin_ops_clipboard_valid", False)

    def execute(self, context):
        _, pin_item = _get_selected_pin(context, self.group_index)
        if pin_item is None:
            self.report({"WARNING"}, iface_("No pin vertex group selected"))
            return {"CANCELLED"}
        wm = context.window_manager
        from ...core.profile import read_pin_operations, validate_pin_source_replacement
        try:
            validate_pin_source_replacement(read_pin_operations(wm.pin_ops_clipboard), pin_item)
        except (ValueError, TypeError, AttributeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _copy_operations_collection(wm.pin_ops_clipboard.operations, pin_item.operations)
        invalidate_overlays()
        redraw_all_areas(context)
        self.report({"INFO"}, iface_("Pin operations pasted"))
        return {"FINISHED"}


classes = [
    SCENE_OT_OpenSceneProfile,
    SCENE_OT_ClearSceneProfile,
    SCENE_OT_ReloadSceneProfile,
    SCENE_OT_SaveSceneProfile,
    OBJECT_OT_OpenMaterialProfile,
    OBJECT_OT_ClearMaterialProfile,
    OBJECT_OT_ReloadMaterialProfile,
    OBJECT_OT_SaveMaterialProfile,
    OBJECT_OT_CopyMaterialParams,
    OBJECT_OT_PasteMaterialParams,
    OBJECT_OT_OpenPinProfile,
    OBJECT_OT_ClearPinProfile,
    OBJECT_OT_ReloadPinProfile,
    OBJECT_OT_SavePinProfile,
    OBJECT_OT_CopyPinOps,
    OBJECT_OT_PastePinOps,
]


def register():
    for cls in classes:
        _ppf_i18n.register_class(cls)
    # Clipboard storage: PointerProperty to the same PropertyGroup
    # classes the source uses. Field-by-field copy is driven by
    # bl_rna introspection at copy/paste time, so the clipboard
    # schema auto-tracks any new ObjectGroup / PinOperation params.
    # WindowManager scope = session-only (not saved to .blend),
    # which matches clipboard semantics.
    from ..state_types import PinVertexGroupItem
    from ..object_group import ObjectGroup
    bpy.types.WindowManager.material_clipboard = PointerProperty(type=ObjectGroup)
    bpy.types.WindowManager.material_clipboard_valid = BoolProperty(default=False)
    bpy.types.WindowManager.material_clipboard_src_type = StringProperty(default="")
    bpy.types.WindowManager.pin_ops_clipboard = PointerProperty(type=PinVertexGroupItem)
    bpy.types.WindowManager.pin_ops_clipboard_valid = BoolProperty(default=False)


def unregister():
    for attr in (
        "material_clipboard",
        "material_clipboard_valid",
        "material_clipboard_src_type",
        "pin_ops_clipboard",
        "pin_ops_clipboard_valid",
    ):
        if hasattr(bpy.types.WindowManager, attr):
            delattr(bpy.types.WindowManager, attr)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
