"""Experimental local contour fitting for the generated capsule topology."""

from bisect import bisect_left
from math import cos, hypot, isfinite, pi


def _weighted_radius(bucket):
    """Weighted 90th percentile, not a maximum that a single weak point owns."""
    samples = sorted(bucket)
    cutoff = 0.9 * sum(weight for _, weight in samples)
    cumulative = 0.0
    for radius, weight in samples:
        cumulative += weight
        if cumulative >= cutoff:
            return radius
    return samples[-1][0]



def _concave_majorant(xs, lower):
    """Smallest concave polygonal profile above the per-ring lower bounds."""
    hull = []
    for i, x in enumerate(xs):
        while len(hull) >= 2:
            a, b = hull[-2:]
            if ((lower[b]-lower[a]) * (x-xs[b])
                    >= (lower[i]-lower[b]) * (xs[b]-xs[a])):
                break
            hull.pop()
        hull.append(i)
    result = lower.copy()
    for a, b in zip(hull, hull[1:]):
        for i in range(a + 1, b):
            t = (xs[i]-xs[a]) / (xs[b]-xs[a])
            # Avoid tiny roundoff undershooting an authored constraint.
            result[i] = max(lower[i], (1.0-t)*lower[a] + t*lower[b])
    return result


def fit_capsule_contour(vertices, samples, half_shaft, strength):
    """Taper existing rings without changing their axial stations or face indices.

    Samples are (axial X in metres, radial extent in metres, skin weight),
    centered like the stock proxy. A fixed axis and regular ring angles make
    the noncollapsed mesh a single-valued radial profile; sections cannot
    cross. This is *not* a new convex hull, a per-angle fit, or an analytic
    capsule at nonzero strength. Preserve conservative stock coverage for
    strong samples (at least half of the maximum selected weight).
    """
    if strength <= 0.0:
        return vertices
    if not isfinite(strength) or strength > 1.0:
        raise ValueError('Contour fit must be between zero and one')
    samples = [(x, r, w) for x, r, w in samples
               if isfinite(x) and isfinite(r) and isfinite(w) and r >= 0.0 and w > 0.0]
    if len(samples) < 6:
        return vertices
    radial_data = {}
    for x, y, z in vertices:
        r = hypot(y, z)
        if x not in radial_data:
            radial_data[x] = (r, 1)
        else:
            radius, count = radial_data[x]
            radial_data[x] = (max(radius, r), count + 1)
    xs = sorted(radial_data)
    stock = [radial_data[x][0] for x in xs]
    if len(xs) < 4 or max(stock) <= 0.0:
        return vertices
    rings = [i for i, r in enumerate(stock) if r > 0.0]
    tolerance = max(1.e-12, abs(half_shaft) * 1.e-9)
    shaft = [i for i in rings if abs(xs[i]) <= half_shaft + tolerance]
    if len(shaft) < 2:
        return vertices
    segment_count = min(radial_data[xs[i]][1] for i in rings)
    if segment_count < 3:
        return vertices

    # Adjacent rings have the same vertex directions. At any in-between X,
    # the inscribed radial coverage is cos(pi/n) times the linear ring radius.
    inradius = cos(pi / segment_count)
    buckets = {i: [] for i in shaft}
    shaft_x = [xs[i] for i in shaft]
    for x, radial, weight in samples:
        p = bisect_left(shaft_x, x)
        if p == len(shaft) or (p and x-shaft_x[p-1] < shaft_x[p]-x):
            p -= 1
        buckets[shaft[p]].append((radial, weight))

    # Avoid letting an isolated sample dictate a ring. Regions without local
    # evidence interpolate between supported sections, never extrapolate a
    # narrow section onto the opposite end without evidence there.
    supported = [i for i in shaft if len(buckets[i]) >= 3]
    if len(supported) < 2:
        return vertices
    sx = [xs[i] for i in supported]
    target_r = [_weighted_radius(buckets[i]) / inradius for i in supported]
    ratios = [1.0] * len(xs)
    for i in shaft:
        p = bisect_left(sx, xs[i])
        if p < len(sx) and xs[i] == sx[p]:
            fitted = target_r[p]
        elif p == 0 or p == len(sx):
            continue
        else:
            t = (xs[i]-sx[p-1]) / (sx[p]-sx[p-1])
            fitted = (1.0-t)*target_r[p-1] + t*target_r[p]
        ratios[i] = max(0.20, min(1.0, fitted / stock[i]))

    # A cap follows its shaft equator's scale. Doing this before enforcing
    # coverage avoids leaving a full-width bulge next to a tightened equator.
    for i in rings:
        if i < shaft[0]:
            ratios[i] = ratios[shaft[0]]
        elif i > shaft[-1]:
            ratios[i] = ratios[shaft[-1]]

    # Retain the stock radial lower bound for strongly weighted samples that
    # stock did not fully enclose (coarse faceting can already miss them).
    # Otherwise reserve enough inscribed radius to keep them inside.
    strong_weight = 0.5 * max(w for _, _, w in samples)
    for x, r, w in samples:
        if w < strong_weight or x < xs[0] or x > xs[-1] or r == 0.0:
            continue
        j = bisect_left(xs, x)
        if j < len(xs) and x == xs[j]:
            if stock[j] > 0.0:
                ratios[j] = max(ratios[j], min(1.0, r / (stock[j]*inradius)))
            continue
        i = j - 1
        t = (x-xs[i]) / (xs[j]-xs[i])
        interpolated = ((1.0-t)*stock[i] + t*stock[j]) * inradius
        if interpolated > 0.0:
            reserve = min(1.0, r/interpolated)
            if stock[i] > 0.0:
                ratios[i] = max(ratios[i], reserve)
            if stock[j] > 0.0:
                ratios[j] = max(ratios[j], reserve)

    # Independent fit buckets can create a saw-tooth as mesh density rises.
    # Raise interior valleys to the smallest concave profile above all
    # section targets and coverage floors. This keeps the outline convex,
    # preserving contact-friendly transitions; it does not fit concavities.
    proposed = [radius * ratio for radius, ratio in zip(stock, ratios)]
    target = _concave_majorant(xs, proposed)
    scale_by_x = {
        x: 1.0 - strength * (1.0 - min(1.0, fitted/radius)) if radius else 1.0
        for x, radius, fitted in zip(xs, stock, target)
    }
    return [(x, y*scale_by_x[x], z*scale_by_x[x]) for x, y, z in vertices]
