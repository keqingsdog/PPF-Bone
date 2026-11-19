# Fork modification: changed from the upstream baseline.
"""Plugin-local language. RNA metadata is translated once at startup."""
import functools
import inspect
import json
import os
import re
import bpy

CONTEXT = "PPF_CONTACT_SOLVER"
_HERE = os.path.dirname(__file__)
_CHOICES = {"AUTO", "zh_HANS", "en_US"}
_registered = False
_syncing = False
_error = ""

def _settings_path():
    return os.path.join(bpy.utils.user_resource('CONFIG'), 'ppf_contact_solver', 'language.json')

def _read_choice():
    try:
        with open(_settings_path(), encoding='utf-8') as f:
            data = json.load(f)
        if data.get('schema') == 1 and data.get('choice') in _CHOICES:
            return data['choice']
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None

_SAVED_CHOICE = _read_choice()
ACTIVE_CHOICE = _SAVED_CHOICE or 'AUTO'
_view = bpy.context.preferences.view
ACTIVE_LOCALE = ACTIVE_CHOICE if ACTIVE_CHOICE != 'AUTO' else bpy.app.translations.locale
if ACTIVE_LOCALE in {'zh_CN', 'zh_SG'}:
    ACTIVE_LOCALE = 'zh_HANS'
if ACTIVE_CHOICE == 'AUTO' and not _view.use_translate_interface:
    ACTIVE_LOCALE = 'en_US'

def _load_locale(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return {k:v for k,v in data.items() if k != '_meta' and isinstance(v,str) and v}

_catalog = _load_locale(os.path.join(_HERE, 'en.json'))
if ACTIVE_LOCALE != 'en_US':
    path = os.path.join(_HERE, ACTIVE_LOCALE + '.json')
    if os.path.isfile(path):
        _catalog.update(_load_locale(path))

def pgettext_iface(message, *args, **kwargs):
    if message in _catalog:
        return _catalog[message]
    if ACTIVE_LOCALE == 'zh_HANS':
        message = re.sub(r'(\d+) self-intersections(?: \((\d+) tri-tri\))?', lambda m: m[1] + ' 处自相交' + ('（' + m[2] + ' 对三角面相交）' if m[2] else ''), message)
    # Catalog extraction folds Python docstring whitespace for stable keys.
    return _catalog.get(' '.join(message.split()), message)
pgettext_tip = pgettext_iface
iface = pgettext_iface

def tooltip(message):
    # Blender translates descriptions without a context. A plugin prefix avoids
    # accidental matches in Blender's built-in catalog, even with global flags on.
    return 'PPF: ' + iface(message) if message else ''

def _build_translations():
    # Never publish plugin entries in Blender's default or Operator contexts.
    return {}

def apply_language(choice, context=None):
    """Persist a pending choice. Never modify Blender language or translation flags."""
    global _SAVED_CHOICE, _error
    if choice not in _CHOICES:
        raise ValueError('Unsupported plugin language: ' + str(choice))
    path = _settings_path()
    temporary = path + '.tmp'
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(temporary, 'w', encoding='utf-8') as f:
            json.dump({'schema':1, 'choice':choice}, f)
        os.replace(temporary, path)
    except OSError as exc:
        _error = str(exc)
        raise
    _SAVED_CHOICE = choice
    _error = ''

def language_update(self, context):
    global _syncing
    if _syncing:
        return
    previous = _SAVED_CHOICE or ACTIVE_CHOICE
    try:
        apply_language(self.ui_language, context)
    except OSError as exc:
        _syncing = True
        try: self.ui_language = previous
        finally: _syncing = False
        print('[PPF language] Could not save plugin language:', exc)

def sync_preferences(preferences):
    global _syncing
    # Legacy zh/en selections become pending only; no global state is replayed.
    if _SAVED_CHOICE is None and preferences.ui_language != 'AUTO':
        language_update(preferences, bpy.context)
    elif _SAVED_CHOICE is not None:
        _syncing = True
        try: preferences.ui_language = _SAVED_CHOICE
        finally: _syncing = False

def language_property():
    from .props import EnumProperty
    return EnumProperty(name='Plugin Language',
        description='Save plugin language; restart Blender to apply',
        items=(('AUTO','Follow Blender','Use Blender language on next startup'),
               ('zh_HANS','中文','Use Chinese for this plugin'),
               ('en_US','English','Use English for this plugin')),
        default='AUTO', update=language_update)

def draw_language(layout, preferences):
    layout.prop(preferences, 'ui_language')
    layout.label(text=iface('Restart Blender to apply the saved plugin language.'), icon='INFO', translate=False)
    layout.label(text=iface('Active plugin language: {locale}').format(locale=ACTIVE_LOCALE), translate=False)
    if _error:
        layout.label(text=iface('Could not save plugin language: {error}').format(error=_error), icon='ERROR', translate=False)

class LocalLayout:
    """A per-draw adapter. Native bpy UI and global translators stay untouched."""
    def __init__(self, layout): object.__setattr__(self, '_layout', layout)
    def __setattr__(self, key, value): setattr(self._layout, key, value)
    def __getattr__(self, key):
        method = getattr(self._layout, key)
        if not callable(method): return method
        @functools.wraps(method)
        def call(*args, **kwargs):
            # UILayout RNA APIs must receive the real owner, never our draw adapter.
            args = tuple(a._target if isinstance(a, _Drawer) else a for a in args)
            kwargs = {k: (v._target if isinstance(v, _Drawer) else v) for k, v in kwargs.items()}
            if 'text' in kwargs and kwargs.get('translate', True):
                kwargs['text'] = iface(kwargs['text'])
            if key in {'prop','prop_search'} and 'text' not in kwargs and not kwargs.get('icon_only', False) and len(args) >= 2:
                try: kwargs['text'] = iface(args[0].bl_rna.properties[args[1]].name)
                except (AttributeError, KeyError): pass
            if key in {'label','operator','operator_menu_enum','menu','prop','prop_search','prop_enum','prop_menu_enum','popover'}:
                kwargs['translate'] = False
            if key in {'prop', 'prop_search'} and kwargs.get('icon_only', False):
                kwargs['text'] = ''
            result = method(*args, **kwargs)
            if isinstance(result, bpy.types.UILayout): return LocalLayout(result)
            if key == 'panel': return tuple(LocalLayout(x) if x is not None else None for x in result)
            return result
        return call

class _Drawer:
    def __init__(self, target):
        self._target = target
        self.layout = LocalLayout(target.layout)
    def __getattr__(self, key):
        descriptor = inspect.getattr_static(type(self._target), key, None)
        if inspect.isfunction(descriptor): return descriptor.__get__(self, type(self._target))
        return getattr(self._target,key)

def _draw_wrapper(fn):
    # Blender validates positional co_argcount; preserve the native signature.
    names = list(inspect.signature(fn).parameters)
    def invoke(self, *args):
        args = tuple(LocalLayout(a) if isinstance(a, bpy.types.UILayout) else a for a in args)
        drawer = self if isinstance(self, _Drawer) or not hasattr(self, 'layout') else _Drawer(self)
        return fn(drawer, *args)
    namespace = {'invoke': invoke}
    arguments = ', '.join(names)
    exec('def wrapped(' + arguments + '):\n    return invoke(' + arguments + ')', namespace)
    wrapped = functools.wraps(fn)(namespace['wrapped'])
    wrapped.__defaults__ = fn.__defaults__
    return wrapped

def register_class(cls):
    """Only addon-owned registration calls use this metadata/draw boundary."""
    if not cls.__dict__.get('_ppf_localized_class', False):
        cls.bl_translation_context = CONTEXT
        if issubclass(cls, bpy.types.Operator) and not cls.__dict__.get('bl_description') and cls.__doc__:
            cls.bl_description = inspect.getdoc(cls)
        for key in ('bl_label', 'bl_description'):
            value = getattr(cls,key,None)
            if value:
                setattr(cls,key, tooltip(value) if key == 'bl_description' else iface(value))
        for key in ('draw','draw_header','draw_header_preset','draw_item','draw_filter'):
            fn = cls.__dict__.get(key)
            if inspect.isfunction(fn): setattr(cls,key,_draw_wrapper(fn))
        cls._ppf_localized_class = True
    bpy.utils.register_class(cls)

def register():
    global _registered
    _registered = True

def unregister():
    global _registered
    _registered = False
