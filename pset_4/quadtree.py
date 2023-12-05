class QuadTreeNode:
    """Constructs a new node with the given x and y ranges and points. If node
    has more than 4 points, it will be split into 4 child nodes."""
    def __init__(self, x_range, y_range, points=None, parent=None):
        self.x_range = x_range  # (x_min, x_max)
        self.y_range = y_range  # (y_min, y_max)
        self.points = points if points else []  # Points in this node
        self.parent = parent  # Parent node
        self.children = []  # Child nodes

        # If this node has more than 4 points, split it into 4 child nodes
        if len(self.points) > 4:
            self.split()

    """Splits this node into 4 child nodes based on quadrants defined by the
    midpoints of the x and y ranges. Points are assigned to child nodes based
    on which quadrant they fall into."""
    def split(self):
        # Find midpoints of x and y ranges
        x_mid = (self.x_range[0] + self.x_range[1]) / 2
        y_mid = (self.y_range[0] + self.y_range[1]) / 2

        # Quadrants are defined as ((x_min, x_max), (y_min, y_max))
        quadrants = [
            ((self.x_range[0], x_mid), (self.y_range[0], y_mid)),  # Bottom left
            ((x_mid, self.x_range[1]), (self.y_range[0], y_mid)),  # Bottom right
            ((self.x_range[0], x_mid), (y_mid, self.y_range[1])),  # Top left
            ((x_mid, self.x_range[1]), (y_mid, self.y_range[1]))  # Top right
        ]

        # Create a child node for each quadrant
        for quadrant in quadrants:
            child_points = [point for point in self.points if self.point_in_range(point, quadrant)]
            child_node = QuadTreeNode(quadrant[0], quadrant[1], child_points, self)
            self.children.append(child_node)

        self.points = []  # Delete points from this node since they are now in child nodes

    """Returns true if the given point is within the given range, false
    otherwise."""
    def point_in_range(self, point, range):
        # range is ((x_min, x_max), (y_min, y_max))
        in_range = range[0][0] <= point[0] <= range[0][1] and range[1][0] <= point[1] <= range[1][1]
        return in_range

    """Returns true if this node contains the given point (x, y), false
    otherwise."""
    def contains(self, x, y):
        return self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]

    """Returns the smallest node that contains the given point."""
    def small_containing_quadtree(self, x, y):
        # If this node is a leaf or does not contain the given point, return this node
        if self.is_leaf() or not self.contains(x, y):
            return self
        # Otherwise, check if any child nodes contain the given point
        for child in self.children:
            if child.contains(x, y):
                return child.small_containing_quadtree(x, y)
        return self

    """Returns the distance between the given point and the closest point in
    this node or its child nodes."""
    @staticmethod
    def euclidean_distance(point_1, point_2):
        return ((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)**0.5

    """Returns true if the given point is within the given distance of this
    node or its child nodes, false otherwise."""
    def within_distance(self, x, y, d):
        closest_x = max(self.x_range[0], min(x, self.x_range[1]))
        closest_y = max(self.y_range[0], min(y, self.y_range[1]))
        return self.euclidean_distance((x, y), (closest_x, closest_y)) <= d

    """Returns all leaves within a given distance of a point (x, y)."""
    def leaves_within_distance(self, x, y, d):
        # If this node is a leaf, return this node if it is within the given distance
        if self.is_leaf():
            return [self] if self.within_distance(x, y, d) else []
        # Otherwise, return all leaves within the given distance from this node's children
        return [leaf for child in self.children for leaf in child.leaves_within_distance(x, y, d)]

    """Returns true if this node is a leaf (no children), false otherwise."""
    def is_leaf(self):
        return len(self.children) == 0

    """Parent tree asks for all points in this node and its children."""
    def all_points(self):
        # If this node is a leaf, return its points
        if self.is_leaf():
            return self.points
        # Otherwise, return all points in this node's children
        return [point for child in self.children for point in child.all_points()]
