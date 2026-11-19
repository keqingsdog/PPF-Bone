> **Fork modification:** This document has been changed from the upstream version to describe the fork’s plugin-local language behavior.

# Plugin-local language

Preferences > PPF > Plugin Language saves `AUTO`, `zh_HANS`, or `en_US`.
Restart Blender normally to apply the saved choice. Both UI text and native
RNA property names, enum labels, descriptions, operator labels and tooltips
use the same startup language. No scene PropertyGroup is unregistered when
choosing a language. Blender language and all translation toggles are untouched.
AUTO resolves Blender's active interface locale on startup (English when
interface translation is disabled).

The pending choice is stored in Blender's user CONFIG directory under
`ppf_contact_solver/language.json` (schema 1, choice only). Missing/malformed
settings fall back to AUTO. A legacy addon preference is migrated after normal
registration and becomes pending until restart; prior global language changes
cannot be reversed without evidence of the old value. A failed write restores
the preference selection and shows an error in plugin preferences.

`i18n.props` wraps only addon-owned property creation, preserving identifiers,
static enum order, numbers, defaults and callbacks. Descriptions have a `PPF: `
prefix to avoid Blender's context-free built-in description translation.
`register_class` prepares only addon classes. `LocalLayout` wraps only a draw
invocation; it respects explicit `translate=False` data labels. No global bpy
translator, property factory or default translation context is overwritten.
Dynamic enum callbacks retain their original ownership and user-defined names.

## Catalogs and limits

`en.json` is the master. Chinese-authored sources may map to English; existing
untranslated Chinese entries remain visible in English mode. This change does
not claim complete English coverage. `<locale>.json` maps master source keys;
empty entries fall back to the master. Technical names, paths and user names
remain unchanged. Preserve runtime format placeholders exactly.

Add a source string to the master and relevant locale catalogs. Addon UI modules
import local `pgettext_iface`/`pgettext_tip`; property definitions import local
factories. These are explicit addon interfaces, never a patch to Blender globals.

The catalog covers static text. Arbitrary formatted Chinese source messages,
external server-generated operator schemas, and fixed words returned by existing
dynamic data callbacks require explicit translations at their source; they are
not silently declared fully translated. Other installed addons are unaffected.
