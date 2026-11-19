"""Plain-data dirty generations; no RNA references survive a timer tick."""
class DirtyMassCache:
    def __init__(self, throttle=.15):
        self.throttle = throttle
        self.entries = {}
        self.measurements = 0
        self.geometry_measurements = 0

    def retain(self, keys):
        live = set(keys)
        for key in set(self.entries)-live: del self.entries[key]

    def dirty(self, key, now, *, geometry=True):
        entry = self.entries.setdefault(key, {'generation': 0, 'handled': -1, 'due': now+self.throttle,
                                             'value': None, 'geometry': None, 'geometry_signature': None, 'geometry_dirty': True})
        entry['generation'] += 1
        entry['geometry_dirty'] = entry['geometry_dirty'] or geometry
        # Debounce bursts and keep repeated UI redraws completely independent.
        entry['due'] = now+self.throttle

    def ready(self, now):
        return [key for key, e in self.entries.items() if e['generation'] != e['handled'] and now >= e['due']]

    def store(self, key, value, *, geometry=None, geometry_signature=None):
        entry = self.entries[key]
        entry['handled'] = entry['generation']; entry['value'] = value
        if geometry is not None:
            entry['geometry'] = geometry; entry['geometry_signature'] = geometry_signature
            self.geometry_measurements += 1
        elif entry['geometry_dirty']:
            # An invalid edited mesh can never inherit the old shell proof.
            entry['geometry'] = None; entry['geometry_signature'] = None
        entry['geometry_dirty'] = False
        self.measurements += 1

    def clean(self, key):
        entry = self.entries.get(key)
        return entry is not None and entry['handled'] == entry['generation']
