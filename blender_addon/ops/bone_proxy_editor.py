"""Opt-in bone-selected collision shape editing; never transform the rig."""
from .. import i18n as _ppf_i18n
import math
import bpy
from ..i18n.props import BoolProperty,FloatVectorProperty,EnumProperty
from ..i18n import pgettext_iface as iface_
from mathutils import Vector, Matrix
from ..core.compat import pose_bone_selected,set_pose_bone_selected

_editor=None
_scaler=None
SCOPES=[('SELECTED','选中骨骼',''),('CHAIN','当前骨链',''),('ALL','整个骨架','')]
CATEGORIES=[('PARAMETERS','常用物理参数','保留目标形状、碰撞、控制和身份'),('SHAPE','形状参数','仅复制形状设置；已有网格须显式重建'),('COLLISION','碰撞设置','仅复制碰撞层和相邻骨碰撞设置'),('ALL','全部设置','复制可编辑设置，保留目标身份')]
EDITOR_HEADER='代理编辑：T循环形状 · C添加 · S缩放 · E网格 · X删除 · P参与骨骼物理 · Esc退出'


def transform_owns_input(context, window=None):
    """Numeric input and Esc belong to the child transform until it finishes."""
    if _scaler is not None:return True
    manager=getattr(context,'window_manager',None)
    for candidate in getattr(manager,'windows',()):
        if window is not None:
            if candidate!=window:continue
        elif candidate.scene!=context.scene:continue
        for operator in getattr(candidate,'modal_operators',()):
            identifiers=(getattr(operator,'bl_idname',''),getattr(getattr(operator,'bl_rna',None),'identifier',''))
            if any(name.startswith('TRANSFORM_OT_') for name in identifiers):return True
    return False


def _cycle_shapes(context,bones):
    """CUSTOM is a batch option only when every source passes read-only preflight."""
    from .bone_proxy import validate_closed_mesh
    shapes=['CAPSULE','SPHERE','BOX']
    checked=set()
    for bone in bones:
        source=bone.ppf_bone_physics.custom_object
        if (source is None or source.type!='MESH' or source.get('ppf_bone_proxy_owned')
                or context.scene.objects.get(source.name)!=source):return shapes
        key=source.as_pointer()
        if key in checked:continue
        mesh=None
        try:
            depsgraph=context.evaluated_depsgraph_get()
            mesh=bpy.data.meshes.new_from_object(source.evaluated_get(depsgraph),depsgraph=depsgraph)
            mesh.transform(source.matrix_world)
            validate_closed_mesh(mesh)
        except (ValueError,RuntimeError,TypeError,ReferenceError):return shapes
        finally:
            if mesh is not None and mesh.users==0:bpy.data.meshes.remove(mesh)
        checked.add(key)
    if bones:shapes.append('CUSTOM')
    return shapes


def shape_hotkey(context,event):
    """BoneX-compatible T cycle; all numeric input remains with Blender."""
    if (event.type!='T' or event.value!='PRESS' or event.ctrl or event.alt or event.shift or getattr(event,'oskey',False)
            or transform_owns_input(context)):return None
    from .bone_proxy import selected_bones
    arm,bones=selected_bones(context,scope=editor_scope(context))
    shapes=_cycle_shapes(context,bones)
    active=arm.data.bones.active
    current=(active if active in bones else bones[0]).ppf_bone_physics.shape
    return shapes[(shapes.index(current)+1)%len(shapes)] if current in shapes else shapes[0]


def _restore_editor_header(area=None, force=False):
    """Restore only after a child operation or eligibility transition."""
    editor=_editor
    if editor is None or editor._stop:return
    if force:editor._header_dirty=True
    try:
        if area is not None and area!=editor._area:return
        transforming=transform_owns_input(bpy.context, window=editor._window)
        eligible=(editor._window.scene==editor._scene and
                  editor._window.view_layer.objects.active==editor._arm and editor._arm.mode=='POSE')
        if transforming or not eligible:
            editor._header_dirty=True
            return
        if editor._header_dirty:
            editor._area.header_text_set(EDITOR_HEADER)
            editor._header_dirty=False
    except (ReferenceError,RuntimeError,AttributeError):pass


def _refresh_editor_header():
    if _editor is None or _editor._stop:return None
    _restore_editor_header()
    return .2


def _stop_header_refresh():
    if bpy.app.timers.is_registered(_refresh_editor_header):
        bpy.app.timers.unregister(_refresh_editor_header)


def _prepare_editor_pose(context):
    """Allow an object-only entry without changing the general update contract."""
    from ..core.bone_geometry_edit import preflight
    from .bone_proxy import enter_proxy_pose
    preflight(context)
    arm=context.object
    if arm is None or arm.type!='ARMATURE':raise ValueError('请先选择一个骨架')
    if sum(obj.type=='ARMATURE' for obj in context.selected_objects)>1:
        raise ValueError('请选择一个骨架，避免跨骨架选择歧义')
    original=context.mode
    bones=arm.data.edit_bones if original=='EDIT_ARMATURE' else arm.data.bones
    active=bones.active
    selected=tuple(b.name for b in bones if (b.select if original=='EDIT_ARMATURE' else pose_bone_selected(arm.pose.bones[b.name])))
    before=(arm,original,selected,active.name if active else None)
    try:
        if original=='OBJECT' and not selected:
            candidates=[b for b in arm.data.bones if not b.hide and not b.hide_select]
            target=active if active in candidates else next((b for b in candidates if b.parent is None),next(iter(candidates),None))
            if target is None:raise ValueError('骨架没有可选骨骼，请先显示或解锁骨骼')
            set_pose_bone_selected(arm.pose.bones[target.name],True)
            arm.data.bones.active=target
        enter_proxy_pose(context)
    except BaseException:
        _restore_entry_context(before)
        raise
    return before


def _restore_entry_context(before):
    arm,mode,selected,active_name=before
    if bpy.context.object!=arm:return
    try:
        if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        if mode=='EDIT_ARMATURE':
            bpy.ops.object.mode_set(mode='EDIT')
            for bone in arm.data.edit_bones:bone.select=bone.name in selected
            arm.data.edit_bones.active=arm.data.edit_bones.get(active_name) if active_name else None
        else:
            for pb in arm.pose.bones:set_pose_bone_selected(pb,pb.name in selected)
            arm.data.bones.active=arm.data.bones.get(active_name) if active_name else None
            if mode=='POSE':bpy.ops.object.mode_set(mode='POSE')
    except (ReferenceError,RuntimeError):pass


def editor_active(context):
    return _editor is not None and not _editor._stop and _editor._scene==context.scene


def editor_scope(context):
    return getattr(context.window_manager,'ppf_bone_editor_scope','SELECTED')


def in_editor_window(editor,context,event):
    """Do not intercept sidebar typing, other editors, or modified shortcuts."""
    if context.area!=editor._area:return False
    return any(r.type=='WINDOW' and r.x<=event.mouse_x<r.x+r.width and r.y<=event.mouse_y<r.y+r.height for r in context.area.regions)

def pending_mesh_proxy(editor):
    from ..core.bone_geometry_edit import leased_proxies
    leased = leased_proxies(kind='MESH')
    if not leased:
        return None
    try:
        candidates = {proxy.as_pointer(): proxy for proxy in leased}
        for bone in editor._arm.data.bones:
            proxy=bone.ppf_bone_physics.proxy
            if proxy is not None and proxy.as_pointer() in candidates:
                return proxy
    except ReferenceError:pass
    return None


def pending_mesh_message(editor):
    return f'请返回场景「{editor._scene.name}」的「{editor._arm.name}」完成当前代理网格编辑（Tab 返回或 Esc 退出）'


def close_editor(editor):
    global _editor
    # A scene/object switch is not a lifecycle reset: preserve the owner of
    # real mesh edits until Return can commit them in the original context.
    if pending_mesh_proxy(editor) is not None:return False
    if _scaler is not None:
        try:_scaler.cancel(None)
        except (ReferenceError,RuntimeError):pass
    editor._stop=True
    try:
        # The viewport display RNA may already be removed during unregister.
        # Only skip restoration in that lifecycle case; ordinary close restores it.
        display=getattr(editor._scene,'ppf_bone_display',None)
        if display is not None:
            for key,value in editor._display_before.items():setattr(display,key,value)
    except (ReferenceError,RuntimeError):pass
    try:editor._area.header_text_set(None);editor._area.tag_redraw()
    except (ReferenceError,RuntimeError):pass
    if _editor is editor:_editor=None
    _stop_header_refresh()
    return True

def scale_factors(values):
    values=tuple(values)
    if len(values)!=3 or any(not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('缩放比例必须是大于零的有限数值')
    return values

def scaled_coordinates(coordinates,factors):
    factors=scale_factors(factors)
    x,y,z=factors
    result=[(float(co[0])*x,float(co[1])*y,float(co[2])*z) for co in coordinates]
    if any(not math.isfinite(v) for co in result for v in co):raise ValueError('缩放后代理坐标超出有限范围')
    return result

def targets(context):
    from ..core.bone_geometry_edit import preflight
    preflight(context)
    from .bone_proxy import selected_bones,validate_closed_mesh
    from ..models.bone_bindings import validate_proxy_owner
    from ..models.bone_bindings import validate_final_binding_ids
    arm,bones=selected_bones(context,scope=editor_scope(context))
    validate_final_binding_ids(context.scene,arm,bones)
    result=[]
    for bone in bones:
        validate_proxy_owner(arm,bone)
        proxy=bone.ppf_bone_physics.proxy
        if not proxy:raise ValueError('选中的骨骼还没有代理；请先按 C 添加')
        # Recovery records hold additional ID references to the same mesh.
        # Only another object using it makes geometry editing non-independent.
        owners=sum(1 for obj in bpy.data.objects if obj.data == proxy.data)
        if proxy.library or proxy.data.library or owners != 1:
            raise ValueError('代理网格必须独立且可编辑')
        if proxy.data.shape_keys is not None:
            raise ValueError('代理含形态键，当前代理缩放不支持；未修改网格')
        result.append(proxy)
    if len({p.as_pointer() for p in result})!=len(result):raise ValueError('选中的骨骼共享了代理')
    # The production validator triangulates and recalculates normals. Validate
    # disposable copies only, after every target's ownership is checked, so a
    # cancelled scale or invalid later target cannot change authored topology.
    for proxy in result:
        candidate=proxy.data.copy()
        try:validate_closed_mesh(candidate)
        finally:bpy.data.meshes.remove(candidate)
    return result

def restore_geometry(snapshots):
    for proxy,coords in snapshots:
        for vertex,co in zip(proxy.data.vertices,coords):vertex.co=co
        proxy.data.update()
        proxy.update_tag(refresh={'DATA'})
    # Flush evaluated modifiers as well as the hidden original mesh before
    # drawing the next modal step. Restore/cancel uses this same path.
    for layer in bpy.context.scene.view_layers:
        layer.update()
    from ..core.bone_mass import invalidate_all
    from ..models.groups import invalidate_overlays
    invalidate_all();invalidate_overlays()

def apply_geometry(snapshots,factors,orientation='LOCAL',matrices=None,*,preview=False,view_layer=None,preview_arrays=None):
    # Precompute the complete batch before any vertex mutation.
    factors=scale_factors(factors)
    if preview and preview_arrays is not None:
        import numpy as np
        packed=[]
        for index,((proxy,_),coords) in enumerate(zip(snapshots,preview_arrays)):
            if orientation=='WORLD':
                a=(matrices[index] if matrices is not None else proxy.matrix_world.to_3x3()).copy()
                if any(not math.isfinite(v) for row in a for v in row):raise ValueError('代理世界矩阵无效')
                try:transform=a.inverted() @ Matrix.Diagonal(Vector(factors)) @ a
                except ValueError:raise ValueError('代理世界矩阵不可逆')
                with np.errstate(over='ignore',invalid='ignore'):
                    values=np.asarray(coords,dtype=np.float32) @ np.asarray(transform,dtype=np.float32).T
            else:
                with np.errstate(over='ignore',invalid='ignore'):
                    values=np.asarray(coords * np.asarray(factors,dtype=np.float64),dtype=np.float32)
            if not np.isfinite(values).all():raise ValueError('缩放后代理坐标超出有限范围')
            packed.append((proxy,np.ascontiguousarray(values).ravel()))
        try:
            for proxy,values in packed:
                proxy.data.vertices.foreach_set('co',values)
                proxy.data.update()
                proxy.update_tag(refresh={'DATA'})
            if view_layer is not None:view_layer.update()
            from ..core.bone_mass import invalidate_all
            from ..models.groups import invalidate_overlays
            invalidate_all();invalidate_overlays()
        except Exception:
            restore_geometry(snapshots)
            raise
        return
    candidates=[]
    for index,(proxy,coords) in enumerate(snapshots):
        if orientation=='WORLD':
            a=(matrices[index] if matrices is not None else proxy.matrix_world.to_3x3()).copy()
            if any(not math.isfinite(v) for row in a for v in row):raise ValueError('代理世界矩阵无效')
            try:transform=a.inverted() @ Matrix.Diagonal(Vector(factors)) @ a
            except ValueError:raise ValueError('代理世界矩阵不可逆')
            result=[tuple(transform @ Vector(co)) for co in coords]
            if any(not math.isfinite(v) for co in result for v in co):raise ValueError('缩放后代理坐标超出有限范围')
        else:result=scaled_coordinates(coords,factors)
        candidates.append((proxy,result))
    if preview:
        # Convert the entire batch before touching authored meshes: float32
        # overflow and non-finite results must not leave a partial edit.
        from array import array
        packed=[]
        for proxy,coords in candidates:
            try: values=array('f',(value for co in coords for value in co))
            except (OverflowError,ValueError):raise ValueError('缩放后代理坐标超出有限范围')
            if any(not math.isfinite(v) for v in values):raise ValueError('缩放后代理坐标超出有限范围')
            packed.append((proxy,values))
        try:
            for proxy,values in packed:
                proxy.data.vertices.foreach_set('co',values)
                proxy.data.update()
                proxy.update_tag(refresh={'DATA'})
            if view_layer is not None:view_layer.update()
            from ..core.bone_mass import invalidate_all
            from ..models.groups import invalidate_overlays
            invalidate_all();invalidate_overlays()
        except Exception:
            restore_geometry(snapshots)
            raise
        return
    from .bone_proxy import validate_closed_mesh
    for proxy,coords in candidates:
        candidate=proxy.data.copy()
        try:
            for vertex,co in zip(candidate.vertices,coords):vertex.co=co
            candidate.update();validate_closed_mesh(candidate)
        finally:bpy.data.meshes.remove(candidate)
    try:restore_geometry(candidates)
    except Exception:
        restore_geometry(snapshots)
        raise

class PPF_OT_BoneProxyScale(bpy.types.Operator):
    bl_idname='ppf.bone_proxy_scale';bl_label='缩放选中骨骼的代理';bl_options={'REGISTER','UNDO'}
    bl_description='鼠标或数字设比例；X/Y/Z 循环本地轴、世界轴和自由缩放，Shift 加轴键约束平面；左键/Enter 确认，右键/Esc 取消'
    factors:FloatVectorProperty(name='代理缩放',size=3,default=(1.,1.,1.),min=.000001)
    orientation:EnumProperty(name='缩放坐标系',items=[('LOCAL','本地',''),('WORLD','世界','')],default='LOCAL')
    @classmethod
    def poll(cls,context):return context.mode=='POSE' and context.object is not None and context.object.type=='ARMATURE'
    def execute(self,context):
        from ..core.bone_geometry_edit import acquire
        self._lease=None
        try:
            snapshots=[(p,[tuple(v.co) for v in p.data.vertices]) for p in targets(context)]
            self._snapshots=snapshots
            self._lease=acquire(context,[proxy for proxy,_coords in snapshots],kind='SCALE')
            apply_geometry(snapshots,self.factors,self.orientation)
            self._finish(commit=True)
        except (ValueError,RuntimeError) as exc:
            if self._lease is not None:self.cancel(context)
            self.report({'ERROR'},iface_(str(exc)));return {'CANCELLED'}
        except BaseException:
            if self._lease is not None:self.cancel(context)
            raise
        return {'FINISHED'}
    def invoke(self,context,event):
        global _scaler
        if not context.region_data:return self.execute(context)
        self._lease=None
        try:self._snapshots=[(p,[tuple(v.co) for v in p.data.vertices]) for p in targets(context)]
        except (ValueError,RuntimeError) as exc:self.report({'ERROR'},iface_(str(exc)));return {'CANCELLED'}
        import numpy as np
        self._preview_arrays=[np.asarray(coords,dtype=np.float64) for _proxy,coords in self._snapshots]
        if any(not np.isfinite(coords).all() for coords in self._preview_arrays):
            self.report({'ERROR'},iface_('代理坐标超出有限范围'));return {'CANCELLED'}
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        # Use the same placement that draws the proxy. Hidden proxy object
        # matrices can lag behind a posed or animated bone.
        from ..ui import bone_overlay
        depsgraph=context.evaluated_depsgraph_get()
        pose_matrices=bone_overlay._pose_display_matrices(context,depsgraph)
        centers=[]
        for proxy,coords in self._snapshots:
            token=pose_matrices.get(proxy.as_pointer())
            if token is None:
                token=bone_overlay._model_matrix(context,proxy,depsgraph)
            matrix=Matrix(tuple(token[i:i+4] for i in range(0,16,4)))
            local_center=sum((Vector(co) for co in coords),Vector())/len(coords) if coords else Vector()
            centers.append(matrix @ local_center)
        center=sum(centers,Vector())/len(centers)
        self._center=location_3d_to_region_2d(context.region,context.region_data,center)
        if self._center is None:self.report({'ERROR'},iface_('请将选中骨骼移到视图内'));return {'CANCELLED'}
        self._origin=Vector((context.region.x,context.region.y))
        self._distance=(Vector((event.mouse_x,event.mouse_y))-self._origin-self._center).length or 1.
        self._factor=1.;self.factors=(1.,1.,1.);self._axes={0,1,2};self._axis_key=None;self.orientation='LOCAL';self._matrices=[p.matrix_world.to_3x3().copy() for p,_ in self._snapshots];self._numeric='';self._area=context.area;self._cancelled=False
        from ..core.bone_geometry_edit import acquire
        try:
            self._lease=acquire(context,[proxy for proxy,_coords in self._snapshots],kind='SCALE')
            context.window_manager.modal_handler_add(self)
        except (ValueError,RuntimeError) as exc:
            if self._lease is not None:self.cancel(context)
            self.report({'ERROR'},iface_(str(exc)));return {'CANCELLED'}
        _scaler=self;return {'RUNNING_MODAL'}
    def _finish(self,commit=False):
        global _scaler
        from ..core.bone_geometry_edit import finish
        token=getattr(self,'_lease',None)
        # Keep the token and snapshots live if committing fails. The caller can
        # restore author geometry, then release without retiring the old build.
        if token is not None:finish(token,commit=commit)
        self._lease=None
        if _scaler is self:_scaler=None
        self._cancelled=True
        try:
            if getattr(self,'_area',None):
                self._area.header_text_set(None)
                _restore_editor_header(self._area, force=True)
        except (ReferenceError,RuntimeError):pass
    def cancel(self,context):
        try:restore_geometry(self._snapshots)
        finally:self._finish(commit=False)
    def modal(self,context,event):
        if self._cancelled:return {'CANCELLED'}
        if not in_editor_window(self,context,event):return {'PASS_THROUGH'}
        if event.type in {'MIDDLEMOUSE','WHEELUPMOUSE','WHEELDOWNMOUSE','TRACKPADPAN','TRACKPADZOOM'} or event.type.startswith('NDOF_'):
            return {'PASS_THROUGH'}
        if event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS':
            self.cancel(context);return {'CANCELLED'}
        if event.type in {'LEFTMOUSE','RET','NUMPAD_ENTER'} and event.value=='PRESS':
            try:
                apply_geometry(self._snapshots,self.factors,self.orientation,self._matrices)
                self._finish(commit=True)
            except (ValueError,RuntimeError) as exc:
                self.cancel(context);self.report({'ERROR'},iface_(str(exc)));return {'CANCELLED'}
            except BaseException:
                self.cancel(context);raise
            return {'FINISHED'}
        if event.value=='PRESS' and event.type in {'X','Y','Z'}:
            # A held key must not cycle through coordinate spaces on OS repeat.
            if getattr(event,'is_repeat',False):return {'RUNNING_MODAL'}
            axis={'X':0,'Y':1,'Z':2}[event.type]
            key=(axis,bool(event.shift))
            if self._axis_key != key:
                self._axis_key=key;self.orientation='LOCAL'
                self._axes={0,1,2}-{axis} if event.shift else {axis}
            elif self.orientation=='LOCAL':self.orientation='WORLD'
            else:self._axis_key=None;self._axes={0,1,2};self.orientation='LOCAL'
        elif event.value=='PRESS' and event.type=='BACK_SPACE':self._numeric=self._numeric[:-1]
        elif event.value=='PRESS' and event.ascii and event.ascii in '0123456789.':self._numeric+=event.ascii
        elif event.type=='MOUSEMOVE' and not self._numeric:
            self._factor=max(.000001,(Vector((event.mouse_x,event.mouse_y))-self._origin-self._center).length/self._distance)
        else:
            # Key releases and unrelated events do not change the requested
            # geometry. Avoid revalidating every mesh and flushing every view
            # layer for those events (including mouse moves during numeric input).
            return {'RUNNING_MODAL'}
        try:
            factor=float(self._numeric) if self._numeric else self._factor
            self.factors=scale_factors(tuple(factor if i in self._axes else 1. for i in range(3)))
            apply_geometry(self._snapshots,self.factors,self.orientation,self._matrices,preview=True,view_layer=context.view_layer,preview_arrays=self._preview_arrays)
        except ValueError as exc:
            if self.orientation=='WORLD' and '矩阵' in str(exc):
                self.cancel(context);self.report({'ERROR'},iface_(str(exc)));return {'CANCELLED'}
            return {'RUNNING_MODAL'}
        except Exception:
            self.cancel(context);raise
        axes=''.join('XYZ'[i] for i in sorted(self._axes)) if self._axis_key is not None else '自由'
        space=('世界' if self.orientation=='WORLD' else '本地') if self._axis_key is not None else ''
        context.area.header_text_set('代理缩放：%.3f  · %s%s · 重复轴切换本地/世界/自由 · 数字输入 · 确认/Enter · Esc取消'%(factor,space,axes))
        return {'RUNNING_MODAL'}

class PPF_OT_BoneProxyGenerate(bpy.types.Operator):
    bl_idname='ppf.bone_proxy_generate';bl_label='应用生成参数到范围';bl_options={'REGISTER','UNDO'}
    bl_description='按范围重建代理；替换已有手工网格，可撤销。快捷键只改形状，按钮同时复制活动骨骼的生成参数'
    scope:EnumProperty(name='范围',items=SCOPES,default='SELECTED')
    shape:EnumProperty(items=[(key,label,'') for key,label in [('ACTIVE','活动骨骼'),('CAPSULE','胶囊'),('SPHERE','球'),('BOX','盒'),('CUSTOM','自定义')]],default='ACTIVE')
    quality:EnumProperty(items=[(key,label,'') for key,label in [('ACTIVE','活动骨骼'),('AUTO','自动'),('VERY_LOW','极低'),('LOW','低'),('MEDIUM','中'),('HIGH','高'),('VERY_HIGH','极高')]],default='ACTIVE')
    face_mode:EnumProperty(items=[('ACTIVE','活动骨骼',''),('QUAD','四边面',''),('TRI','三角面','')],default='ACTIVE')
    copy_settings:BoolProperty(default=True,options={'HIDDEN'})
    reset_default_size:BoolProperty(name='重算默认尺寸',description='仅按各目标当前骨长重算默认尺寸；保留形状、LOD、面类型、材质、关节和作者偏移；不适用于自定义网格',default=False)
    @classmethod
    def poll(cls,context):
        from ..models.bone_bindings import active_armature
        return active_armature(context) is not None and context.mode in {'OBJECT','POSE','EDIT_ARMATURE'}
    def execute(self,context):
        from .bone_proxy import enter_proxy_pose,selected_bones,update_proxies
        from ..models.proxy_generation import effective_proxy_quality,effective_proxy_face_mode
        from ..core.bone_geometry_edit import preflight
        previous=[]
        original=context.mode
        try:
            preflight(context)
            # Bone RNA edited from the panel while in armature Edit mode is
            # replaced when Blender commits EditBones on the mode transition.
            # Keep the authored generation fields by name across that commit.
            edit_values={}
            if original=='EDIT_ARMATURE':
                from ..models.bone_bindings import active_armature
                edit_arm=active_armature(context)
                edit_bones=edit_arm.data.edit_bones
                active_edit=edit_bones.active
                if self.scope=='ALL':
                    names={bone.name for bone in edit_bones}
                elif self.scope=='CHAIN' and active_edit:
                    names={active_edit.name, *(bone.name for bone in active_edit.children_recursive)}
                else:
                    names={bone.name for bone in edit_bones if bone.select}
                if active_edit:names.add(active_edit.name)
                fields=('shape','proxy_quality','proxy_face_mode','length_ratio','radius','box_width','box_depth','custom_object')
                for name in names:
                    bone=edit_arm.data.bones.get(name)
                    if bone is not None:
                        settings=bone.ppf_bone_physics
                        edit_values[name]={key:(settings.is_property_set(key),getattr(settings,key)) for key in fields}
            enter_proxy_pose(context,self.scope)
            arm,bones=selected_bones(context,scope=self.scope)
            for name,snapshot in edit_values.items():
                bone=arm.data.bones.get(name)
                if bone is None:raise ValueError(name+'：切换模式后骨骼不存在')
                settings=bone.ppf_bone_physics
                for key,(was_set,value) in snapshot.items():
                    if was_set:
                        if not settings.is_property_set(key) or getattr(settings,key)!=value:setattr(settings,key,value)
                    elif settings.is_property_set(key):settings.property_unset(key)
            if self.reset_default_size:
                if self.shape!='ACTIVE' or self.quality!='ACTIVE' or self.face_mode!='ACTIVE':
                    raise ValueError('重算默认尺寸不能同时切换形状、LOD或面类型')
                if any(bone.ppf_bone_physics.shape=='CUSTOM' for bone in bones):
                    raise ValueError('重算默认尺寸不适用于自定义网格；保留全部选中代理和源网格')
                count=update_proxies(context,scope=self.scope,rebuild=True,reset_default_size=True)
                self.report({'INFO'},iface_(f'已按当前骨长重算 {count} 根骨骼的默认尺寸；可撤销'))
                return {'FINISHED'}
            active=arm.data.bones.active
            if active is None:raise ValueError('请先指定活动骨骼')
            source=active.ppf_bone_physics
            dimensions=('length_ratio','radius','box_width','box_depth','custom_object')
            fields=('shape','proxy_quality','proxy_face_mode')+dimensions
            # Snapshot field values AND unset state before any batch setters.
            for bone in bones:
                settings=bone.ppf_bone_physics
                previous.append((settings,{key:(settings.is_property_set(key),getattr(settings,key)) for key in fields}))
            shape=source.shape if self.shape=='ACTIVE' else self.shape
            quality=effective_proxy_quality(source) if self.quality=='ACTIVE' else self.quality
            face=effective_proxy_face_mode(source) if self.face_mode=='ACTIVE' else self.face_mode
            values={key:getattr(source,key) for key in dimensions}
            for bone in bones:
                settings=bone.ppf_bone_physics
                target_shape=settings.shape if self.shape=='ACTIVE' and not self.copy_settings else shape
                if target_shape=='CUSTOM' and (values['custom_object'] if self.copy_settings else settings.custom_object) is None:
                    raise ValueError(bone.name+'：自定义形状需要源网格；保留原代理')
            for bone in bones:
                settings=bone.ppf_bone_physics
                if self.copy_settings or self.shape!='ACTIVE':settings.shape=shape
                if self.copy_settings:
                    settings.proxy_quality=quality;settings.proxy_face_mode=face
                    for key,value in values.items():setattr(settings,key,value)
                else:
                    if self.quality!='ACTIVE':settings.proxy_quality=quality
                    if self.face_mode!='ACTIVE':settings.proxy_face_mode=face
            shape_changed_from={bone.name:snapshot['shape'][1]
                for bone,(_settings,snapshot) in zip(bones,previous)
                if snapshot['shape'][1]!=bone.ppf_bone_physics.shape
                and bone.ppf_bone_physics.shape=='CAPSULE'}
            count=update_proxies(context,scope=self.scope,rebuild=True,shape_changed_from=shape_changed_from)
            self.report({'INFO'},iface_(f'已更新 {count} 根骨骼的代理形状；可撤销'))
            return {'FINISHED'}
        except Exception as exc:
            for settings,snapshot in previous:
                for key,(was_set,value) in snapshot.items():
                    setattr(settings,key,value)
                    if not was_set:settings.property_unset(key)
            self.report({'ERROR'},iface_(str(exc)));return {'CANCELLED'}
        finally:
            if original!='POSE' and context.object is not None:
                try:bpy.ops.object.mode_set(mode='EDIT' if original=='EDIT_ARMATURE' else 'OBJECT')
                except RuntimeError:pass


class PPF_OT_BoneEditorAction(bpy.types.Operator):
    bl_idname='ppf.bone_editor_action';bl_label='代理编辑操作';bl_options={'REGISTER','UNDO'}
    bl_description='对统一范围操作代理；删除和重建需确认，均可撤销'
    action:EnumProperty(items=[(key,label,'') for key,label in [('UPDATE','添加 / 更新'),('DELETE','删除代理'),('ENABLE','启用物理'),('DISABLE','停用物理'),('TOGGLE','切换参与物理'),('REBUILD','按形状参数重建')]],default='UPDATE')
    scope:EnumProperty(name='范围',items=SCOPES,default='SELECTED')
    @classmethod
    def poll(cls,context):
        from ..models.bone_bindings import active_armature
        return active_armature(context) is not None and context.mode in {'OBJECT','POSE','EDIT_ARMATURE'}
    def invoke(self,context,event):
        if self.action in {'DELETE','REBUILD'}:return context.window_manager.invoke_props_dialog(self,width=320)
        return self.execute(context)
    def draw(self,context):
        self.layout.label(text='删除范围内的代理，可撤销' if self.action=='DELETE' else '重建将替换范围内的手工网格与变换',icon='QUESTION')
        self.layout.prop(self,'scope')
    def execute(self,context):
        try:
            if self.action in {'UPDATE','REBUILD'}:
                return bpy.ops.ppf.bone_proxy_update(scope=self.scope,rebuild=self.action=='REBUILD')
            if self.action=='DELETE':return bpy.ops.ppf.bone_proxy_delete(scope=self.scope)
            enabled=self.action=='ENABLE'
            if self.action=='TOGGLE':
                from .bone_proxy import selected_bones
                _arm,bones=selected_bones(context,scope=self.scope)
                enabled=not all(b.ppf_bone_physics.enabled for b in bones if b.ppf_bone_physics.proxy)
            return bpy.ops.ppf.bone_participation(scope=self.scope,enabled=enabled)
        except (RuntimeError,ValueError) as exc:self.report({'ERROR'},iface_(str(exc)));return {'CANCELLED'}


def _retire_forced_mesh():
    """Forced cancel cannot veto Blender; preserve fail-closed upload state."""
    from .bone_proxy import _facade_communicator
    communicator=_facade_communicator()
    try:
        communicator.bone_physical_body_changed()
        return []
    except Exception as error:
        runner=communicator._runner
        try:
            with runner._collision_lock:
                # Same retirement contract as the facade. Use the existing
                # refusal path to clear proofs and publish the actual failure.
                runner._body_retransfer=True;runner._body_required=True
                runner._collision_reject(error,runner._collision_ticket())
        except Exception as fallback:
            return [str(error),str(fallback)]
        return [str(error)]


class PPF_OT_BoneProxyEditor(bpy.types.Operator):
    bl_idname='ppf.bone_proxy_editor';bl_label='进入代理编辑';bl_options={'REGISTER'}
    bl_description='代理编辑：T循环胶囊/球/盒（源齐全时含自定义）；C添加、S缩放、E网格、X删除、P参与骨骼物理、Esc退出。变换时数字输入保留'
    @classmethod
    def poll(cls,context):return context.area is not None and context.area.type=='VIEW_3D' and context.object is not None and context.object.type=='ARMATURE' and context.mode in {'OBJECT','POSE','EDIT_ARMATURE'}
    def invoke(self,context,event):
        global _editor
        if _editor is not None:
            if not close_editor(_editor):
                self.report({'ERROR'},iface_(pending_mesh_message(_editor)));return {'CANCELLED'}
            return {'FINISHED'}
        before=None
        try:
            before=_prepare_editor_pose(context)
            self._stop=False;self._header_dirty=True;self._arm=context.object;self._area=context.area;self._scene=context.scene;self._window=context.window
            display=context.scene.ppf_bone_display
            self._display_before={'enabled':display.enabled,'proxy_enabled':display.proxy_enabled,'edges':display.edges}
            # Enabling the master must not reveal a spring layer that was
            # previously hidden by that master switch.
            if not display.enabled:
                self._display_before['cross_links']=display.cross_links
                display.cross_links=False
            display.proxy_enabled=True
            display.enabled=True
            if not display.faces and not display.edges:display.edges=True
            _editor=self
            context.window_manager.modal_handler_add(self)
            _restore_editor_header()
            if not bpy.app.timers.is_registered(_refresh_editor_header):
                bpy.app.timers.register(_refresh_editor_header,first_interval=.2)
        except (ValueError,RuntimeError) as exc:
            if _editor is self:close_editor(self)
            if before is not None:_restore_entry_context(before)
            self.report({'ERROR'},iface_(str(exc)));return {'CANCELLED'}
        return {'RUNNING_MODAL'}
    def modal(self,context,event):
        global _editor
        if self._stop:return {'FINISHED'}
        if context.scene!=self._scene:
            return {'FINISHED'} if close_editor(self) else {'PASS_THROUGH'}
        if not in_editor_window(self,context,event):return {'PASS_THROUGH'}
        if transform_owns_input(context):
            self._header_dirty=True
            return {'PASS_THROUGH'}
        pending=pending_mesh_proxy(self)
        if pending is not None and context.object!=pending:return {'PASS_THROUGH'}
        if event.type=='ESC' and event.value=='PRESS' and _scaler is None:
            if (context.mode=='EDIT_MESH' or pending is not None) and context.object and context.object.get('ppf_bone_proxy_owned'):
                try:result=bpy.ops.ppf.bone_proxy_return()
                except RuntimeError:return {'RUNNING_MODAL'}
                if result!={'FINISHED'}:return {'RUNNING_MODAL'}
            return {'FINISHED'} if close_editor(self) else {'RUNNING_MODAL'}
        if context.mode=='EDIT_MESH' or pending is not None:
            self._header_dirty=True
            if event.value=='PRESS' and event.type in {'TAB','E'} and not(event.ctrl or event.alt or event.shift):
                try:bpy.ops.ppf.bone_proxy_return()
                except RuntimeError:pass
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}
        if _scaler is not None:return {'PASS_THROUGH'}
        if context.object!=self._arm or context.mode!='POSE':
            return {'FINISHED'} if close_editor(self) else {'PASS_THROUGH'}
        if event.value=='PRESS' and event.type in {'G','R'}:self._header_dirty=True
        if event.value=='PRESS' and not(event.ctrl or event.alt or event.shift or getattr(event,'oskey',False)):
            try:
                shape=shape_hotkey(context,event)
                if shape is not None:
                    bpy.ops.ppf.bone_proxy_generate('EXEC_DEFAULT',shape=shape,scope=editor_scope(context),copy_settings=False)
                    return {'RUNNING_MODAL'}
                if event.type=='C':bpy.ops.ppf.bone_editor_action('INVOKE_DEFAULT',action='UPDATE',scope=editor_scope(context));return {'RUNNING_MODAL'}
                if event.type in {'X','DEL'}:bpy.ops.ppf.bone_editor_action('INVOKE_DEFAULT',action='DELETE',scope=editor_scope(context));return {'RUNNING_MODAL'}
                if event.type=='P':bpy.ops.ppf.bone_editor_action('INVOKE_DEFAULT',action='TOGGLE',scope=editor_scope(context));return {'RUNNING_MODAL'}
                if event.type=='S':
                    region=next(r for r in context.area.regions if r.type=='WINDOW')
                    with context.temp_override(region=region):bpy.ops.ppf.bone_proxy_scale('INVOKE_DEFAULT')
                    return {'RUNNING_MODAL'}
                if event.type=='E':
                    self._header_dirty=True
                    bpy.ops.ppf.bone_proxy_select();return {'RUNNING_MODAL'}
            except (RuntimeError,ValueError) as exc:self.report({'ERROR'},iface_(str(exc)));return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}
    def cancel(self,context):
        global _editor
        from ..core.bone_geometry_edit import token_for,finish
        owns_editor=_editor is self
        token=None;errors=[]
        try:
            if owns_editor:
                if _scaler is not None:
                    try:_scaler.cancel(context)
                    except Exception as error:errors.append(str(error))
                proxy=pending_mesh_proxy(self)
                if proxy is not None:
                    token=token_for(proxy,kind='MESH')
                    # Edit BMesh can be newer than Mesh RNA: conservatively
                    # retire instead of treating an unflushed edit as unchanged.
                    if token is not None:errors.extend(_retire_forced_mesh())
        finally:
            if token is not None:
                try:finish(token,commit=False)
                except Exception as error:errors.append(str(error))
            self._stop=True
            if owns_editor:
                try:
                    for key,value in self._display_before.items():setattr(self._scene.ppf_bone_display,key,value)
                except (ReferenceError,RuntimeError,AttributeError):pass
                try:self._area.header_text_set(None);self._area.tag_redraw()
                except (ReferenceError,RuntimeError,AttributeError):pass
                if _editor is self:_editor=None
                _stop_header_refresh()
            self._display_before={};self._arm=None;self._scene=None;self._area=None
        if errors:self.report({'WARNING'},iface_('代理编辑已结束，旧构建已禁止使用：'+'；'.join(errors)))
        return None

classes=(PPF_OT_BoneProxyScale,PPF_OT_BoneProxyGenerate,PPF_OT_BoneEditorAction,PPF_OT_BoneProxyEditor)
@bpy.app.handlers.persistent
def cancel_editor(*_args):
    global _editor,_scaler
    from ..core.bone_geometry_edit import release_all
    # Load/Undo/Redo and unregister are cancellation boundaries. In particular,
    # a later E Return must not commit a mesh edit while undo is restoring data.
    release_all()
    if _scaler is not None:
        try:_scaler.cancel(None)
        except (ReferenceError,RuntimeError):pass
    if _editor is not None:
        # Return only the mesh opened by this rig's editor. Never hide a user mesh.
        try:
            if bpy.context.mode=='EDIT_MESH' and bpy.context.object and any(b.ppf_bone_physics.proxy==bpy.context.object for b in _editor._arm.data.bones):bpy.ops.ppf.bone_proxy_return()
        except (ReferenceError,RuntimeError):pass
        close_editor(_editor)

def register():
    from ..core import bone_geometry_edit,bone_selection_parameters
    bone_geometry_edit.register()
    bone_selection_parameters.register()
    for cls in classes:_ppf_i18n.register_class(cls)
    bpy.types.WindowManager.ppf_bone_editor_scope=EnumProperty(name='操作范围',items=SCOPES,default='SELECTED')
    bpy.types.WindowManager.ppf_bone_editor_category=EnumProperty(name='批量内容',items=CATEGORIES,default='PARAMETERS')
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if cancel_editor not in handlers:handlers.append(cancel_editor)
def unregister():
    cancel_editor()
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if cancel_editor in handlers:handlers.remove(cancel_editor)
    del bpy.types.WindowManager.ppf_bone_editor_scope
    del bpy.types.WindowManager.ppf_bone_editor_category
    for cls in reversed(classes):bpy.utils.unregister_class(cls)
    from ..core import bone_geometry_edit,bone_selection_parameters
    bone_selection_parameters.unregister()
    bone_geometry_edit.unregister()
