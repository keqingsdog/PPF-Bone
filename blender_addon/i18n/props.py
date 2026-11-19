"""Local metadata factories; no bpy factory or runtime RNA is patched."""
import functools
import bpy
from . import CONTEXT, iface, tooltip

def original_factory(name):
    """Identity only: deferred properties retain Blender's original factory."""
    return getattr(bpy.props, name)

def _factory(name):
    original = getattr(bpy.props, name)
    @functools.wraps(original)
    def make(*args, **kwargs):
        kwargs['translation_context'] = CONTEXT
        if 'name' in kwargs: kwargs['name'] = iface(kwargs['name'])
        if kwargs.get('description'): kwargs['description'] = tooltip(kwargs['description'])
        if name == 'EnumProperty' and 'items' in kwargs:
            items = kwargs['items']
            def translate_items(source):
                # Identifiers, separators, icon and explicit stored numbers survive.
                return [None if item is None else tuple([item[0], iface(item[1]), tooltip(item[2])] + list(item[3:])) for item in source]
            if callable(items):
                # Preserve user labels and callback semantics; only descriptions
                # need protection from Blender's context-free tooltip translator.
                holder = []
                descriptions = {}
                @functools.wraps(items)
                def local_items(self, context):
                    source = items(self, context)
                    def retained_description(source):
                        if source not in descriptions:
                            descriptions[source] = tooltip(source)
                        return descriptions[source]
                    result = [None if item is None else tuple([item[0], item[1], retained_description(item[2])] + list(item[3:])) for item in source]
                    holder[:] = [result]
                    return result
                local_items._ppf_description_cache = descriptions
                local_items._ppf_local_items_holder = holder
                kwargs['items'] = local_items
            else:
                kwargs['items'] = translate_items(items)
        return original(*args, **kwargs)
    return make

for _name in dir(bpy.props):
    if _name.endswith('Property'):
        globals()[_name] = _factory(_name)
