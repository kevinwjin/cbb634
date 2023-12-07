class QuadTreeNode:
    """Constructs a new node with the given x and y ranges and points. If node
    has more than 4 points, it will be split into 4 child nodes."""
    def __init__(self, x_range, y_range, points=None, parent=None):
        self.x_range = x_range  # (x_min, x_max)
        self.y_range = y_range  # (y_min, y_max)
        self.points = points if points else []  # Distribute points to this node if given
        self.parent = parent  # Parent node
        self.children = []  # Child nodes

        # If this node has more than 4 points, split it into 4 child nodes
        if len(self.points) > 4:
            self.split()

    """Splits this node into 4 child nodes based on bounding box quadrants
    defined by the midpoints of x and y. Points are assigned to child nodes
    based on which quadrant of the bounding box they fall into."""
    def split(self):
        # Find midpoints of x and y ranges
        x_mid = (self.x_range[0] + self.x_range[1]) / 2
        y_mid = (self.y_range[0] + self.y_range[1]) / 2

        # Bounding box quadrants = ((x_min, x_max), (y_min, y_max))
        quadrants = [
            ((self.x_range[0], x_mid), (self.y_range[0], y_mid)),  # Bottom left quadrant
            ((x_mid, self.x_range[1]), (self.y_range[0], y_mid)),  # Bottom right quadrant
            ((self.x_range[0], x_mid), (y_mid, self.y_range[1])),  # Top left quadrant
            ((x_mid, self.x_range[1]), (y_mid, self.y_range[1]))  # Top right quadrant
        ]

        # Create a child node for each bounding box quadrant
        for quadrant in quadrants:
            # Find points that fall within the quadrant
            child_points = [point for point in self.points if self.point_in_range(point, quadrant)]
            # Create a child node for the quadrant
            child_node = QuadTreeNode(quadrant[0], quadrant[1], child_points, self)
            # Add the child node to this node's children
            self.children.append(child_node)

        self.points = []  # Purge this node's points list since they are now in child nodes

    """Checks if the point (x, y) is within this node's bounds."""
    def contains(self, x, y):
        return self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]

    """Checks if the given point is within the given range."""
    def point_in_range(self, point, quadrant):
        # Unpack quadrant into x and y ranges
        (x_min, x_max), (y_min, y_max) = quadrant
        # Unpack point into x and y coordinates
        x, y, _ = point
        # Check if point is within the given range
        return x_min <= x <= x_max and y_min <= y <= y_max

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

    """Calculates the distance between two points in this node or its child nodes."""
    @staticmethod
    def euclidean_distance(point_1, point_2):
        return ((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)**0.5

    """Checks if the given point (x, y) is within the given distance d of this node
    or its child nodes."""
    def within_distance(self, x, y, d):
        closest_x = max(self.x_range[0], min(x, self.x_range[1]))
        closest_y = max(self.y_range[0], min(y, self.y_range[1]))
        return self.euclidean_distance((x, y), (closest_x, closest_y)) <= d

    """Checks if this node is a leaf (does not have any children)."""
    def is_leaf(self):
        return len(self.children) == 0

    """Returns all leaves within a given distance of the given point (x, y)."""
    def leaves_within_distance(self, x, y, d):
        # Return this leaf if it is within the given distance d
        if self.is_leaf():
            return [self] if self.within_distance(x, y, d) else []
        # Otherwise, recurse through this node's children and return all leaves within the given distance d
        return [leaf for child in self.children for leaf in child.leaves_within_distance(x, y, d)]

    """Return all points in this node and its children."""
    def all_points(self):
        # Checks if this node is a leaf; return its points if so
        if self.is_leaf():
            return self.points
        # Otherwise, recurse through this node's children and return all points within them
        return [point for child in self.children for point in child.all_points()]
