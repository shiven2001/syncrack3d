# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from dataclasses import field

import numpy as np

from .expression import ScalarExpression, nodedataclass
from .set_reasoning import ObjectSetExpression


@nodedataclass()
class center_stable_surface_dist(ScalarExpression):
    objs: ObjectSetExpression


@nodedataclass()
class accessibility_cost(ScalarExpression):
    objs: ObjectSetExpression
    others: ObjectSetExpression
    normal: np.array = field(default_factory=lambda: np.array([1, 0, 0]))
    dist: float = 1.0

    def __post_init__(self):
        if isinstance(self.normal, (list, tuple)):
            self.normal = np.array(self.normal)
        assert isinstance(self.normal, np.ndarray)


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class distance(ScalarExpression):
    objs: ObjectSetExpression
    others: ObjectSetExpression
    others_tags: set = field(default_factory=set)

    def __post_init__(self):
        assert isinstance(self.objs, ObjectSetExpression)
        assert isinstance(self.others, ObjectSetExpression)
        assert isinstance(self.others_tags, set)


@nodedataclass()
class min_distance_internal(ScalarExpression):
    objs: ObjectSetExpression


@nodedataclass()
class focus_score(ScalarExpression):
    objs: ObjectSetExpression
    others: ObjectSetExpression


@nodedataclass()
class angle_alignment_cost(ScalarExpression):
    objs: ObjectSetExpression
    others: ObjectSetExpression
    others_tags: set = None

    def __post_init__(self):
        if self.others_tags is None:
            self.others_tags = set()
        assert isinstance(self.others_tags, set), type(self.others_tags)


@nodedataclass()
class freespace_2d(ScalarExpression):
    objs: ObjectSetExpression
    others: ObjectSetExpression


@nodedataclass()
class min_dist_2d(ScalarExpression):
    objs: ObjectSetExpression
    others: ObjectSetExpression


@nodedataclass()
class rotational_asymmetry(ScalarExpression):
    objs: ObjectSetExpression


@nodedataclass()
class reflectional_asymmetry(ScalarExpression):
    objs: ObjectSetExpression
    others: ObjectSetExpression
    use_long_plane: bool = True


@ObjectSetExpression.register_postfix_func
@nodedataclass()
class volume(ScalarExpression):
    objs: ObjectSetExpression
    dims: int | tuple = 3


@nodedataclass()
class coplanarity_cost(ScalarExpression):
    objs: ObjectSetExpression


@nodedataclass()
class wall_plane_offset(ScalarExpression):
    """Perpendicular distance from each obj's own front plane to the nearest other.

    A wall fixture and an opening cut into the same wall are separated only by
    the wall's half thickness plus whatever asymmetry the cutter's extrusion has
    - about 0.2 m in practice. A fixture on the perpendicular return wall is
    separated by however far along the wall the opening sits, which is metres.
    That difference is what distinguishes "beside the door" from "round the
    corner from the door".

    Neither existing node can see it. `distance` measures to the cutter box,
    which is deliberately extruded straight through the wall, so a switch round
    the corner reads as being just as close as one on the door's own wall.
    `angle_alignment_cost` compares the object's axis against the closest edge
    of the other's XY projection, and the cutter's through-wall sides are edges
    too, so it scores a correctly placed switch as badly as a misplaced one.
    """

    objs: ObjectSetExpression
    others: ObjectSetExpression
