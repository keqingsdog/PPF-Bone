"""Persistent author links. UUIDs identify bones; labels never resolve ownership."""
from .. import i18n as _ppf_i18n
import bpy
from ..i18n.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty


class PPF_CrossLink(bpy.types.PropertyGroup):
    link_uuid: StringProperty()
    enabled: BoolProperty(name='启用', default=True, options=set())
    selected: BoolProperty(name='选择', default=False)
    bone_a_uuid: StringProperty()
    bone_b_uuid: StringProperty()
    bone_a_name: StringProperty()
    bone_b_name: StringProperty()
    anchor_a_bone: FloatVectorProperty(size=3, options=set())
    anchor_b_bone: FloatVectorProperty(size=3, options=set())
    rest_length_m: FloatProperty(name='初始距离 (m)', min=0., precision=6, options=set())
    reference_armature_scale: FloatProperty(default=1., min=1.e-8, options=set())
    stiffness_N_per_m: FloatProperty(name='刚度 (N/m)', min=0., precision=4, options=set())
    damping_Ns_per_m: FloatProperty(name='阻尼 (N·s/m)', min=0., precision=4, options=set())
    exclude_collision: BoolProperty(name='排除连接骨对碰撞', default=False, options=set())
    tension_enabled: BoolProperty(name='启用拉伸力', default=True, options=set())
    compression_enabled: BoolProperty(name='启用压缩力', default=True, options=set())
    tension_stiffness_N_per_m: FloatProperty(name='拉伸刚度 (N/m)', min=0., options=set())
    compression_stiffness_N_per_m: FloatProperty(name='压缩刚度 (N/m)', min=0., options=set())
    preview_color_custom: BoolProperty(name='自定义连接颜色', default=False, options=set())
    preview_color: FloatVectorProperty(name='连接颜色', size=4, subtype='COLOR', min=0., max=1., default=(.2,.85,1.,1.), options=set())
    source: StringProperty(default='MASS_DEFAULT_2HZ')
    topology: StringProperty(default='PAIR')


class PPF_CrossLinkDraft(bpy.types.PropertyGroup):
    search_text: StringProperty(name='搜索连接', default='')
    mode: EnumProperty(name='连接', items=[('PAIR','骨对','选中两个不同的物理骨骼；同链及父子骨对可用；已有骨对保留原参数'),('PLANE','平面','至少两条等长、无分叉骨链'),('TUBE','筒形','至少三条等长、无分叉骨链')])
    diagonals: BoolProperty(name='添加双斜向', default=False)
    use_mass_defaults: BoolProperty(name='按质量初始化', default=True, description='2Hz、阻尼比0.7的未标定工程起点；写入后不自动覆盖手动参数')
    stiffness_N_per_m: FloatProperty(name='刚度 (N/m)', min=0., default=100.)
    damping_Ns_per_m: FloatProperty(name='阻尼 (N·s/m)', min=0., default=1.)
    exclude_collision: BoolProperty(name='排除连接骨对碰撞', default=False)
    exclude_direct_only: BoolProperty(name='仅排除直接相邻', description='只排除同层直接连接的碰撞，斜向连接仍参与碰撞', default=False)
    tension_enabled: BoolProperty(name='启用拉伸力', default=True)
    compression_enabled: BoolProperty(name='启用压缩力', default=True)
    tension_stiffness_N_per_m: FloatProperty(name='拉伸刚度 (N/m)', min=0., default=100.)
    compression_stiffness_N_per_m: FloatProperty(name='压缩刚度 (N/m)', min=0., default=100.)
    preview_color_custom: BoolProperty(name='自定义连接颜色', default=False)
    preview_color: FloatVectorProperty(name='连接颜色',size=4,subtype='COLOR',min=0.,max=1.,default=(.2,.85,1.,1.))
    active_index: IntProperty(default=0, min=-1)


ACTUAL_FIELDS=('enabled','tension_enabled','compression_enabled','tension_stiffness_N_per_m','compression_stiffness_N_per_m','damping_Ns_per_m','exclude_collision','preview_color_custom','preview_color')


def _actual_get(field):
    def get(self):
        from ..core import cross_links as author
        return author.actual_value(self.id_data,field)
    return get


def _actual_set(field):
    def set_value(self,value):
        from ..core import cross_links as author
        try:
            author.direct_set(bpy.context,self.id_data,field,value)
            self.last_error=''
        except (ValueError,RuntimeError,TypeError) as exc:
            self.last_error=str(exc)
    return set_value


class PPF_CrossLinkActual(bpy.types.PropertyGroup):
    last_error: StringProperty(options={'SKIP_SAVE'})


for _field in ACTUAL_FIELDS:
    _kw=dict(get=_actual_get(_field),set=_actual_set(_field),options={'SKIP_SAVE'})
    if _field=='preview_color':_prop=FloatVectorProperty(size=4,subtype='COLOR',min=0.,max=1.,**_kw)
    elif _field in {'enabled','tension_enabled','compression_enabled','exclude_collision','preview_color_custom'}:_prop=BoolProperty(**_kw)
    else:_prop=FloatProperty(min=0.,precision=4,**_kw)
    PPF_CrossLinkActual.__annotations__[_field]=_prop


def register():
    _ppf_i18n.register_class(PPF_CrossLink)
    _ppf_i18n.register_class(PPF_CrossLinkDraft)
    _ppf_i18n.register_class(PPF_CrossLinkActual)
    bpy.types.Object.ppf_cross_link_actual = PointerProperty(type=PPF_CrossLinkActual)
    bpy.types.Object.ppf_cross_links = CollectionProperty(type=PPF_CrossLink)
    bpy.types.Object.ppf_cross_link_draft = PointerProperty(type=PPF_CrossLinkDraft)


def unregister():
    del bpy.types.Object.ppf_cross_link_actual
    bpy.utils.unregister_class(PPF_CrossLinkActual)
    del bpy.types.Object.ppf_cross_link_draft
    del bpy.types.Object.ppf_cross_links
    bpy.utils.unregister_class(PPF_CrossLinkDraft)
    bpy.utils.unregister_class(PPF_CrossLink)
