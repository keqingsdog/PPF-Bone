"""Read-only RNA validation before profile setters clear authored collections."""
import math


def validate_fields(data, rna, mapping=None, path='profile'):
    if not isinstance(data, dict):
        raise ValueError(f'{path} must be a table')
    for key, value in data.items():
        name = mapping.get(key) if mapping is not None else key
        prop = rna.properties.get(name) if name else None
        if prop is None:
            continue  # Keep legacy behavior: unknown profile fields are ignored.
        label = f'{path}.{key}'
        if prop.type == 'COLLECTION':
            if not isinstance(value, list):
                raise ValueError(f'{label} must be an array of tables')
            for i, entry in enumerate(value):
                validate_fields(entry, prop.fixed_type, path=f'{label}[{i}]')
            continue
        length = getattr(prop, 'array_length', 0)
        values = value if length else [value]
        if length and (not isinstance(value, (list, tuple)) or len(value) != length):
            raise ValueError(f'{label} requires exactly {length} values')
        for scalar in values:
            if prop.type == 'FLOAT':
                valid = isinstance(scalar, (int, float)) and not isinstance(scalar, bool)
                if valid:
                    try:
                        valid = math.isfinite(scalar) and abs(scalar) <= 3.4028234663852886e38
                    except OverflowError:
                        valid = False
            elif prop.type == 'INT':
                valid = isinstance(scalar, int) and not isinstance(scalar, bool) and -(2**31) <= scalar < 2**31
            elif prop.type == 'BOOLEAN':
                valid = isinstance(scalar, bool)
            elif prop.type == 'STRING':
                valid = isinstance(scalar, str)
            elif prop.type == 'ENUM':
                valid = isinstance(scalar, str) and scalar in prop.enum_items.keys()
            else:
                continue
            if not valid:
                raise ValueError(f'{label} has an invalid {prop.type.lower()} value')
