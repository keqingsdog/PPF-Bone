"""Read the solver-owned kernel ABI order, preserving append-only IDs."""
from pathlib import Path
import argparse

def kernel_sources(root):
    root=Path(root).resolve()
    order=root/'seam/kernel_order.txt'
    if not order.is_file():
        return sorted(str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*.kernel.cpp'))
    names=[line.strip() for line in order.read_text(encoding='utf-8').splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if not names or len(names)!=len(set(names)):
        raise ValueError('Kernel ABI order must be nonempty and unique')
    for name in names:
        source=(root/name).resolve()
        if not source.is_relative_to(root) or not name.endswith('.kernel.cpp') or not source.is_file():
            raise ValueError('Invalid kernel source in ABI order: '+name)
    return names

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    print('\n'.join(kernel_sources(args.root)))
