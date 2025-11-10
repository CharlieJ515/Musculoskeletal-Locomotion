from typing import Optional
from dataclasses import dataclass

import opensim


@dataclass(frozen=True)
class IndexBundle:
    joint: dict[str, int]
    body: dict[str, int]
    muscle: dict[str, int]
    force: dict[str, int]
    marker: dict[str, int]
    force_label: dict[tuple[str, str], int]
    probe: dict[str, int]


_index_bundle: Optional[IndexBundle] = None


def build_index_bundle(model: opensim.Model) -> IndexBundle:
    global _index_bundle

    joint_set = model.getJointSet()
    body_set = model.getBodySet()
    muscle_set = model.getMuscles()
    force_set = model.getForceSet()
    marker_set = model.getMarkerSet()
    probe_set = model.getProbeSet()

    joint_index = {joint_set.get(i).getName(): i for i in range(joint_set.getSize())}
    body_index = {body_set.get(i).getName(): i for i in range(body_set.getSize())}
    muscle_index = {muscle_set.get(i).getName(): i for i in range(muscle_set.getSize())}
    force_index = {force_set.get(i).getName(): i for i in range(force_set.getSize())}
    marker_index = {marker_set.get(i).getName(): i for i in range(marker_set.getSize())}
    probe_index = {probe_set.get(i).getName(): i for i in range(probe_set.getSize())}

    force_label_index: dict[tuple[str, str], int] = {}
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        fname = f.getName()
        labels = f.getRecordLabels()
        for j in range(labels.getSize()):
            force_label_index[(fname, labels.get(j))] = j

    index_bundle = IndexBundle(
        joint=joint_index,
        body=body_index,
        muscle=muscle_index,
        force=force_index,
        marker=marker_index,
        force_label=force_label_index,
        probe=probe_index,
    )

    _index_bundle = index_bundle
    return index_bundle


def get_index_bundle() -> IndexBundle:
    global _index_bundle
    if _index_bundle is None:
        raise RuntimeError("Index bundle is not initialized. Call build_index_bundle")
    return _index_bundle


def joint_index(name: str) -> int:
    try:
        return get_index_bundle().joint[name]
    except KeyError:
        raise KeyError(f"Unknown joint '{name}'")


def body_index(name: str) -> int:
    try:
        return get_index_bundle().body[name]
    except KeyError:
        raise KeyError(f"Unknown body '{name}'")


def muscle_index(name: str) -> int:
    try:
        return get_index_bundle().muscle[name]
    except KeyError:
        raise KeyError(f"Unknown muscle '{name}'")


def force_index(name: str) -> int:
    try:
        return get_index_bundle().force[name]
    except KeyError:
        raise KeyError(f"Unknown force '{name}'")


def marker_index(name: str) -> int:
    try:
        return get_index_bundle().marker[name]
    except KeyError:
        raise KeyError(f"Unknown marker '{name}'")


def probe_index(name: str) -> int:
    try:
        return get_index_bundle().probe[name]
    except KeyError:
        raise KeyError(f"Unknown probe '{name}'")


def force_label_index(force_name: str, label: str) -> int:
    try:
        return get_index_bundle().force_label[(force_name, label)]
    except KeyError:
        raise KeyError(
            f"No record label '{label}' for force '{force_name}'. "
            f"Inspect labels via model.getForceSet().get('{force_name}').getRecordLabels()."
        )
