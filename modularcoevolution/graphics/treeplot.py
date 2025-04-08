#  Copyright 2025 BONSAI Lab at Auburn University
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2025, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

# Adapted from https://llimllib.github.io/pymag-trees/
import warnings
from typing import Literal

import matplotlib.patches
import matplotlib.text
import matplotlib.transforms
from matplotlib import pyplot

from modularcoevolution.genotypes.geneticprogramming.gpnode import GPNode
from modularcoevolution.genotypes.geneticprogramming.gptree import GPTree


class DrawTree:
    def __init__(self, node: GPNode, parent=None, depth=0, sibling_index=1):
        self.x = -1.0
        self.y = depth
        self.node = node
        self.children = [DrawTree(child, self, depth + 1, i + 1)
                         for i, child in enumerate(node.input_nodes)]
        self.parent = parent
        self.thread = None
        self.mod = 0
        self.ancestor = self
        self.change = 0
        self.shift = 0
        self._leftmost_sibling = None
        self.sibling_index = sibling_index

        if parent is None:
            _buchheim_algorithm(self)

    def left(self):
        return self.thread or len(self.children) and self.children[0]

    def right(self):
        return self.thread or len(self.children) and self.children[-1]

    def left_sibling(self):
        sibling = None
        if self.parent:
            for node in self.parent.children:
                if node == self:
                    return sibling
                sibling = node
        return sibling

    @property
    def leftmost_sibling(self):
        if not self._leftmost_sibling and self.parent and self != self.parent.children[0]:
            self._leftmost_sibling = self.parent.children[0]
        return self._leftmost_sibling


def _buchheim_algorithm(tree):
    draw_tree = _first_walk(tree)
    min_x = _second_walk(draw_tree)
    if min_x < 0:
        _third_walk(draw_tree, -min_x)
    return draw_tree


def _first_walk(draw_tree, distance=1.0):
    if len(draw_tree.children) == 0:
        if draw_tree.leftmost_sibling:
            draw_tree.x = draw_tree.left_sibling().x + distance
        else:
            draw_tree.x = 0.0
    else:
        default_ancestor = draw_tree.children[0]
        for child in draw_tree.children:
            _first_walk(child)
            default_ancestor = _apportion(child, default_ancestor, distance)
        _execute_shifts(draw_tree)

        midpoint = (draw_tree.children[0].x + draw_tree.children[-1].x) / 2

        left_sibling = draw_tree.left_sibling()
        if left_sibling:
            draw_tree.x = left_sibling.x + distance
            draw_tree.mod = draw_tree.x - midpoint
        else:
            draw_tree.x = midpoint
    return draw_tree


def _apportion(draw_tree, default_ancestor, distance):
    left_sibling = draw_tree.left_sibling()
    if left_sibling:
        inner_tree = outer_tree = draw_tree
        inner_left_tree = left_sibling
        outer_left_tree = draw_tree.leftmost_sibling
        inner_tree_mod = outer_tree_mod = draw_tree.mod
        inner_left_tree_mod = inner_left_tree.mod
        outer_left_tree_mod = outer_left_tree.mod
        while inner_left_tree.right() and inner_tree.left():
            inner_left_tree = inner_left_tree.right()
            inner_tree = inner_tree.left()
            outer_left_tree = outer_left_tree.left()
            outer_tree = outer_tree.right()
            outer_tree.ancestor = draw_tree
            shift = (inner_left_tree.x + inner_left_tree_mod) - (inner_tree.x + inner_tree_mod) + distance
            if shift > 0:
                ancestor = _find_ancestor(inner_left_tree, draw_tree, default_ancestor)
                _move_subtree(ancestor, draw_tree, shift)
                inner_tree_mod += shift
                outer_tree_mod += shift
            inner_left_tree_mod += inner_left_tree.mod
            inner_tree_mod += inner_tree.mod
            outer_left_tree_mod += outer_left_tree.mod
            outer_tree_mod += outer_tree.mod
        if inner_left_tree.right() and not outer_tree.right():
            outer_tree.thread = inner_left_tree.right()
            outer_tree.mod += inner_left_tree_mod - outer_tree_mod
        elif inner_tree.left() and not outer_left_tree.left():
            outer_left_tree.thread = inner_tree.left()
            outer_left_tree.mod += inner_tree_mod - outer_left_tree_mod
            default_ancestor = draw_tree
    return default_ancestor


def _move_subtree(left_subtree, right_subtree, shift):
    num_subtrees = right_subtree.sibling_index - left_subtree.sibling_index
    right_subtree.change -= shift / num_subtrees
    right_subtree.shift += shift
    left_subtree.change += shift / num_subtrees
    right_subtree.x += shift
    right_subtree.mod += shift


def _execute_shifts(draw_tree):
    shift = change = 0
    for child in reversed(draw_tree.children):
        child.x += shift
        child.mod += shift
        change += child.change
        shift += child.shift + change


def _find_ancestor(inner_left_tree, draw_tree, default_ancestor):
    if inner_left_tree.ancestor in draw_tree.parent.children:
        return inner_left_tree.ancestor
    return default_ancestor


def _second_walk(draw_tree, mod_sum=0, depth=0, min_x=None):
    draw_tree.x += mod_sum
    draw_tree.y = depth

    if min_x is None or draw_tree.x < min_x:
        min_x = draw_tree.x

    for child in draw_tree.children:
        min_x = _second_walk(child, mod_sum + draw_tree.mod, depth + 1, min_x)

    return min_x


def _third_walk(draw_tree, shift):
    draw_tree.x += shift
    for child in draw_tree.children:
        _third_walk(child, shift)


class TreePlotData:
    axes: pyplot.axes
    sub_axes: list[pyplot.axes]
    axes_map: dict[GPNode, pyplot.axes]
    text_scales: dict[pyplot.Text, float]

    def __init__(self, axes: pyplot.axes, axes_map: dict[GPNode, pyplot.axes], text_scales: dict[pyplot.Text, float]):
        self.axes = axes
        self.sub_axes = list(axes_map.values())
        self.axes_map = axes_map
        self.text_scales = text_scales


def plot_tree(
        tree: GPTree,
        axes: pyplot.axes,
        node_scale: float = 0.5,
        node_aspect: float = 1,
        direction: Literal['up', 'left', 'down', 'right'] | float = 'down'
):
    draw_tree = DrawTree(tree.root)

    transform = axes.transData
    match direction:
        case 'up':
            rotation = 0
        case 'left':
            rotation = 90
        case 'down':
            rotation = 180
        case 'right':
            rotation = 270
        case float():
            rotation = float(direction)
        case _:
            raise ValueError(f"Invalid direction: {direction}")
    transform = matplotlib.transforms.Affine2D().rotate_deg(rotation) + transform

    axes_map = _plot_node(draw_tree, axes, node_scale=node_scale, transform=transform)
    axes.set_aspect(node_aspect)
    axes.autoscale_view()

    text_scales = _initialize_tree_text(axes, axes_map)
    return TreePlotData(axes, axes_map, text_scales)


def _plot_node(
        draw_node: DrawTree,
        axes: pyplot.axes,
        node_scale: float = 0.5,
        text_location: Literal['top', 'bottom'] = 'top',
        transform: matplotlib.transforms.Transform = None
) -> dict[GPNode, pyplot.axes]:
    if transform is None:
        transform = axes.transData
    width = node_scale
    height = node_scale
    x = draw_node.x - width / 2
    y = draw_node.y - height / 2

    sub_axes: pyplot.axes = axes.inset_axes([x, y, width, height], transform=transform)
    if text_location == 'top':
        node_name = sub_axes.text(0.5, 1.05, draw_node.node.function_id, ha='center', va='bottom', transform=sub_axes.transAxes)
    else:
        node_name = sub_axes.text(0.5, 0.95, draw_node.node.function_id, ha='center', va='top', transform=sub_axes.transAxes)
    # rescale_text(sub_axes, node_name)
    sub_axes.axis('off')
    _plot_node_data(draw_node.node, sub_axes)

    node_rectangle = pyplot.Rectangle((x, y), width, height, edgecolor='black', facecolor='none', transform=transform)
    node_rectangle.set_zorder(10)
    axes.add_patch(node_rectangle)

    axes_map = {draw_node.node: sub_axes}

    if draw_node.children:
        branch_top = y + height
        branch_bottom = draw_node.y + 0.5
        branch_left = draw_node.children[0].x
        branch_right = draw_node.children[-1].x
        branch_down_line = pyplot.Line2D([draw_node.x, draw_node.x], [branch_top, branch_bottom], color='black', transform=transform)
        branch_across_line = pyplot.Line2D([branch_left, branch_right], [branch_bottom, branch_bottom], color='black', transform=transform)
        axes.add_line(branch_down_line)
        axes.add_line(branch_across_line)

        for child in draw_node.children:
            axes_map.update(_plot_node(child, axes, node_scale=node_scale, text_location=text_location, transform=transform))
            stem_top = child.y - 0.5
            stem_bottom = child.y - width / 2
            stem_line = pyplot.Line2D([child.x, child.x], [stem_top, stem_bottom], color='black', transform=transform)
            axes.add_line(stem_line)

    return axes_map


def _plot_node_data(gp_node: GPNode, axes: pyplot.axes, update=False):
    if gp_node.saved_value is None:
        warnings.warn("This tree has not been executed with \"save_values = True\"")
        return

    if gp_node.can_render_type():
        gp_node.render_saved_value(axes, update=update)
        return

    if not update:
        match gp_node.saved_value:
            case float():
                axes.text(0.5, 0.5, f"{gp_node.saved_value:.2f}", ha='center', va='center')
            case str():
                axes.text(0.5, 0.5, gp_node.saved_value, ha='center', va='center')
            case _:
                axes.text(0.5, 0.5, str(gp_node.saved_value), ha='center', va='center')
    else:
        match gp_node.saved_value:
            case float():
                axes.texts[1].set_text(f"{gp_node.saved_value:.2f}")
            case str():
                axes.texts[1].set_text(gp_node.saved_value)
            case _:
                axes.texts[1].set_text(str(gp_node.saved_value))
        # for text in axes.texts:
        #     rescale_text(axes, text)


def _scale_text(area, text: pyplot.Text, width_margin: float = 0.1, height_margin: float = 0.1) -> float:
    area_extent = area.get_window_extent()
    text_extent = text.get_window_extent().expanded(1 + width_margin, 1 + height_margin)
    # TODO: Allow scaling up without also scaling text above relative default, somehow
    scale = min((area_extent.height * 0.25) / text_extent.height, area_extent.width / text_extent.width)
    text.set_size(text.get_size() * scale)
    return scale


def _initialize_tree_text(axes, axes_map: dict[GPNode, pyplot.axes]) -> dict[pyplot.Text, float]:
    text_scales = {}
    axes.figure.draw_without_rendering()
    for sub_axes in axes_map.values():
        for text in sub_axes.texts:
            scale = _scale_text(sub_axes, text)
            text_scales[text] = scale
    return text_scales


def update_tree_text(plot_data: TreePlotData):
    plot_data.axes.figure.draw_without_rendering()
    new_scales = {}
    for sub_axes in plot_data.axes_map.values():
        for text in sub_axes.texts:
            new_scale = _scale_text(sub_axes, text)
            new_scales[text] = new_scale
    # print(f"Old scales: {plot_data.text_scales}")
    # print(f"New scales: {new_scales}")


def update_tree_plot(plot_data: TreePlotData):
    for node, axes in plot_data.axes_map.items():
        _plot_node_data(node, axes, update=True)


def remap_tree(axes_map: dict[GPNode, pyplot.axes], new_tree: GPTree):
    new_axes_map = {}
    for old_node, new_node in zip(axes_map.keys(), new_tree.get_node_list()):
        new_axes_map[new_node] = axes_map[old_node]
    return new_axes_map
