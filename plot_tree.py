import matplotlib.pyplot as plt

#plots decision tree recursively, can give an optional max depth to limit the depth of the plot
def plot_tree(node, depth=0, pos=(0, 0), ax=None, max_depth=None):
    x_offset=3
    y_step=3
    if ax is None:
        fig, ax = plt.subplots(figsize=(40, 20))
        ax.set_axis_off()

    # Set the position for the current node
    x, y = pos
    if max_depth is not None and depth >= max_depth:
        ax.text(
            x, y, "... (truncated)", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray")
        )
        return

    # Plot the current node
    if node.leaf_class is not None:
        ax.text(x, y, f"Leaf: {node.leaf_class}", ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    else:
        ax.text(x, y, f"x{node.attribute} < {node.value}", ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        dx = x_offset / (1.75 ** depth)
        dy = y_step

        # Plot left and right subtrees
        plot_tree(node.l_tree, depth + 1, pos=(x - dx, y - dy), ax=ax, max_depth=max_depth)
        ax.plot([x, x - dx], [y - 0.2, y - dy + 0.2], 'k-')

        plot_tree(node.r_tree, depth + 1, pos=(x + dx, y - dy), ax=ax, max_depth=max_depth)
        ax.plot([x, x + dx], [y - 0.2, y - dy + 0.2], 'k-')

    if depth == 0:
        plt.show()
